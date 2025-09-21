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

export PATH="/share/app/intel2020/intelpython3/bin:$PATH"
export CONDA_PREFIX="/share/home/liuchuandong/.conda/envs/reann"

echo Begin Time: `date`
echo Directory is: $PWD
echo This job run on the nodes:
if [ -n "$SLURM_JOB_NODELIST" ]; then
    echo $SLURM_JOB_NODELIST
else
    echo "SLURM_JOB_NODELIST is empty"
fi

#path of REANN package `REANN-github-0408`
path="/share/home/liuchuandong/reann-0408/REANN-github-0408/"
#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
WORLD_SIZE=`expr $SLURM_JOB_NUM_NODES \* $NPROC_PER_NODE`
COMMAND="$path"
MASTER=`/bin/hostname -s`
id_master=`expr $(($RANDOM%10000)) + 10000`
echo $id_master '|' `date` '|' `pwd` > id_master

PYTHON_PATH="/share/home/liuchuandong/.conda/envs/reann/bin/python"

if [ -f "$PYTHON_PATH" ]; then
    echo "Using Python: $PYTHON_PATH"
else
    echo "Error: Python not found at $PYTHON_PATH"
    exit 1
fi

cd /share/home/liuchuandong/reann-0408/examp-reann-h2o

$PYTHON_PATH -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$id_master $COMMAND > out 2>&1

echo End Time: `date`
