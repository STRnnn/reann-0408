import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from src.KFWrapper import KFOptimizerWrapper


def Optimize_KF(fout,prop_ceff,nprop,train_nele,test_nele,lr,init_f,final_f,decay_factor,start_lr,end_lr,patience_epoch,print_epoch,Epoch,\
data_train,data_test,Prop_class,loss_fn,optim,restart,PES_Normal,device,PES_Lammps=None): 

    KFOptWrapper = KFOptimizerWrapper(
        Prop_class, optim, atoms_selected=24, atoms_per_group=6, epoch=0, is_distributed=False, distributed_backend="torch"
    )

    rank=dist.get_rank()
    best_loss=1e30*torch.ones(1,device=device)    
    patience=30
    for iepoch in range(Epoch): 
        # set the model to train
       Prop_class.train()
       lossprop=torch.zeros(nprop,device=device)        
       for data in data_train:
          abProp,cart,numatoms,species,atom_index,shifts=data
          ab_list = list(abProp)
        #   loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts),ab_list)
        #   lossprop+=loss.detach()
        #   loss=torch.sum(torch.mul(loss,prop_ceff[0:nprop]))
          Etot_label = ab_list[0] if len(ab_list)>0 else None
          Force_label = ab_list[1] if len(ab_list)>1 else None

          Etot_pred = KFOptWrapper.update_energy(cart,numatoms,species,atom_index,shifts, Etot_label)
          Etot_pred, Force_pred = KFOptWrapper.update_force(cart,numatoms,species,atom_index,shifts, Force_label)

          loss = loss_fn((Etot_pred,Force_pred),ab_list)
          lossprop+=loss.detach()
          loss=torch.sum(torch.mul(loss,prop_ceff[0:nprop]))
    
       #  print the error of vailadation and test each print_epoch
    #    if np.mod(iepoch,print_epoch)==0:
       if np.mod(iepoch,1)==0:
          # apply the EMA parameters to evaluate
          # ema.apply_shadow()
          # set the model to eval for used in the model
          Prop_class.eval()
          # all_reduce the rmse form the training process 
          # here we dont need to recalculate the training error for saving the computation
          dist.all_reduce(lossprop,op=dist.ReduceOp.SUM)
          loss=torch.sum(lossprop)
          
          # get the current rank and print the error in rank 0
          if rank==0:
              lossprop=torch.sqrt(lossprop.detach().cpu()/train_nele)
              fout.write("{:5} {:4} {:15} {:5e}  {} ".format("Epoch=",iepoch,"learning rate",lr,"train error:"))
              for error in lossprop:
                  fout.write('{:10.5f} '.format(error))
            
          # calculate the test error
          lossprop=torch.zeros(nprop,device=device)
          for data in data_test:
             abProp,cart,numatoms,species,atom_index,shifts=data
             loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts,\
             create_graph=False),abProp)
             lossprop=lossprop+loss.detach()

          # all_reduce the rmse
          dist.all_reduce(lossprop,op=dist.ReduceOp.SUM)
          loss=torch.sum(torch.mul(lossprop,prop_ceff[0:nprop]))

        #   scheduler.step(loss)
        #   lr=optim.param_groups[0]["lr"]
        #   f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
        #   prop_ceff[1]=f_ceff
          f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
          prop_ceff[1]=f_ceff

          #  save the best model
          if loss<best_loss[0]:
             best_loss[0]=loss
             patience=30
             if rank == 0:
                 state = {'reannparam': Prop_class.state_dict(), 'optimizer': optim.state_dict()}
                 torch.save(state, "./REANN.pth")
                 PES_Normal.jit_pes()
                 if PES_Lammps:
                     PES_Lammps.jit_pes()
          else:
             patience-=1
             if patience==0:
                lr=lr*decay_factor
                patience=30
          # restore the model for continue training
          # ema.restore()
          # back to the best error
          if loss>25*best_loss[0] or loss.isnan():
              restart(Prop_class,"REANN.pth")
        #       optim.param_groups[0]["lr"]=optim.param_groups[0]["lr"]*decay_factor
              lr=lr*decay_factor
          if rank==0:
              lossprop=torch.sqrt(lossprop.detach().cpu()/test_nele)
              fout.write('{} '.format("test error:"))
              for error in lossprop:
                 fout.write('{:10.5f} '.format(error))
              # if stop criterion
              fout.write("\n")
              fout.flush()
          if lr<end_lr: break
