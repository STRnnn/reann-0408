#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=Train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g04
#SBATCH --output=/share/home/liuchuandong/reann_log/reann-h2o.out
#SBATCH --error=/share/home/liuchuandong/reann_log/reann-h2o.err

module purge
module load cuda/11.8
module load gcc/9.5.0
module load openmpi/4.1.6
module load cmake/3.28

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
conda deactivate
source activate $conda_env
cd $SLURM_SUBMIT_DIR
#python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$SLURM_JOB_NUM_NODES --standalone $COMMAND > out
python -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$id_master $COMMAND >out
echo End Time: `date`
