#!/bin/bash

#SBATCH -J gop
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/gopModelsTraining
#SBATCH --gres=gpu:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/gopModelsTraining/out/gop.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/gopModelsTraining/out/gop.%N.%J.%u.err # STDERR

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/gopModelsTraining/training_scripts/hpcDLScripts/gop_model_train.py 0

