#!/bin/bash
#SBATCH -J test1
####SBATCH -N 1    
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -p a100   ###队列名称
#SBATCH --output=%J.out
#SBATCH --error=%J.err

source ~/.bashrc
# conda environment
conda_env=pytorch-1.12.1-gpu-11.3

echo Begin Time: `date`
echo Dirretory is: $PWD
echo This job run on the nodes:
cat $SLURM_JOB_NODELIST

#path of REANN package `REANN-github-0408`
path="/public/home/dat-share/xjf/reann-0408/REANN-github-0408/"
#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
WORLD_SIZE=`expr $SLURM_JOB_NUM_NODES \* $NPROC_PER_NODE`
COMMAND="$path"
MASTER=`/bin/hostname -s`
id_master=`expr $(($RANDOM%10000)) + 10000`
echo $id_master '|' `date` '|' `pwd` > id_master
source activate $conda_env
cd $SLURM_SUBMIT_DIR
#python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$SLURM_JOB_NUM_NODES --standalone $COMMAND > out
python -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$id_master $COMMAND >out
echo End Time: `date`