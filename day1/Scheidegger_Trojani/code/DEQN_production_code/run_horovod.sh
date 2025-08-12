#!/bin/bash

export HOME=/beegfs/swift/alphacruncher.net/salia/nuvolos.cloud/rselab/deep_equilibrium_nets/andras_sali/files
cd $HOME/DSGE_DEQ/src/sudden_stop
python run_deepnet.py hydra.run.dir=$HOME/runs/sudden_stop 