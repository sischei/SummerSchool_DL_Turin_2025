#!/bin/bash
#SBATCH --job-name="deepnet"
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=intq
#SBATCH --output="/beegfs/swift/alphacruncher.net/salia/nuvolos.cloud/rselab/deep_equilibrium_nets/andras_sali/files/hpc_logs/deepnet_%j.out"

module load cudnn7.6-cuda10.1/7.6.5.32

source activate deepnet 

python --version

export HOME=/beegfs/swift/alphacruncher.net/salia/nuvolos.cloud/rselab/deep_equilibrium_nets/andras_sali

cd $HOME/files/DSGE_DEQ/src/sudden_stop/ && python run_deepnet.py hydra.run.dir=$HOME/files/runs/sudden_stop net=sudden_stop STARTING_POINT=LATEST learning_rate=0.0000001 N_minibatch_size=10000
