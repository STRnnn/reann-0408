import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import time
import numpy as np
import torch.distributed as dist
import math

class KFOptimizerWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        atoms_selected: int,
        atoms_per_group: int,
        epoch: int,
        is_distributed: bool = False,
        distributed_backend: str = "torch",  # torch or horovod
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.atoms_selected = atoms_selected  # 24
        self.atoms_per_group = atoms_per_group  # 6
        self.epoch = epoch
        self.is_distributed = is_distributed
        self.distributed_backend = distributed_backend

    def update_energy(
        self, cart: torch.Tensor, numatoms: torch.Tensor, species: torch.Tensor, atom_index: torch.Tensor, shifts: torch.Tensor,
        Etot_label: torch.Tensor, update_prefactor: float = 1
    ) -> torch.Tensor:
        Etot_predict, _ = self.model(
            cart, numatoms, species, atom_index, shifts, create_graph=True
        )
        natoms_sum = numatoms[0]
        self.optimizer.set_grad_prefactor(natoms_sum)

        self.optimizer.zero_grad()
        bs = Etot_label.shape[0]
        error = Etot_label - Etot_predict
        error = error / natoms_sum
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        if self.is_distributed:
            if self.distributed_backend == "horovod":
                import horovod as hvd
                error = hvd.torch.allreduce(error)
            elif self.distributed_backend == "torch":
                dist.all_reduce(error)
                error /= dist.get_world_size()

        Etot_predict = update_prefactor * Etot_predict
        Etot_predict[mask] = -1 * Etot_predict[mask]

        Etot_predict.sum().backward()
        error = error * math.sqrt(bs)
        # if error > 1 or error.isnan():
        #     error = 1
        self.optimizer.step(error)
        return Etot_predict

    def update_force(
        self, cart: torch.Tensor, numatoms: torch.Tensor, species: torch.Tensor, atom_index: torch.Tensor, shifts: torch.Tensor,
        Force_label: torch.Tensor, update_prefactor: float = 1
    ) -> tuple:
        natoms_sum = numatoms[0]
        bs = Force_label.shape[0]
        self.optimizer.set_grad_prefactor(natoms_sum * self.atoms_per_group * 3)

        index = self.__sample(self.atoms_selected, self.atoms_per_group, natoms_sum)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            Etot_predict, force_predict_flat = self.model(
                cart, numatoms, species, atom_index, shifts
            )
            # [B, N*3]-->[B, N, 3]
            B = force_predict_flat.shape[0]
            force_predict = force_predict_flat.view(B, -1, 3)
            force_label = Force_label.view(B, -1, 3)
            
            error_tmp = force_label[:, index[i]] - force_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum

            if self.is_distributed:
                if self.distributed_backend == "horovod":
                    import horovod as hvd
                    error = hvd.torch.allreduce(error)
                elif self.distributed_backend == "torch":
                    dist.all_reduce(error)
                    error /= dist.get_world_size()

            tmp_force_predict = force_predict[:, index[i]] * update_prefactor
            tmp_force_predict[mask] = -1 * tmp_force_predict[mask]

            # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
            (tmp_force_predict.sum() + Etot_predict.sum() * 0).backward()
            error = error * math.sqrt(bs)
            # if error > 1 or error.isnan():
            #  error = 1
            self.optimizer.step(error)
        return Etot_predict, force_predict_flat

    def __sample(
        self, atoms_selected: int, atoms_per_group: int, natoms: int
    ) -> np.ndarray:
        if atoms_selected % atoms_per_group:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, atoms_selected).reshape(-1, atoms_per_group)
        # res = np.array([i for i in range(natoms)])
        # res = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24]])
        return res

# with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False) as prof:
#     the code u wanna profile
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
