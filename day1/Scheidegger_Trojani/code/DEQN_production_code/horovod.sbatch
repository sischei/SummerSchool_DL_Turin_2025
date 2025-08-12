#!/bin/bash
#SBATCH --job-name="horovod"
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
# SBATCH --gres=gpu:1
#SBATCH --partition=intq
#SBATCH --output="/beegfs/swift/alphacruncher.net/salia/nuvolos.cloud/rselab/deep_equilibrium_nets/andras_sali/files/hpc_logs/deepnet_%j.out"

module load cudnn7.6-cuda10.1/7.6.5.32
module load openmpi/gcc/64/3.1.2

source ~/miniconda3/bin/activate ~/horovod/env

horovodrun --mpi -np 3 bash /beegfs/swift/alphacruncher.net/salia/nuvolos.cloud/rselab/deep_equilibrium_nets/andras_sali/files/DSGE_DEQ/src/sudden_stop/run_horovod.sh
