#!/bin/bash

#SBATCH -J emb_rnn
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/phoneEmbeddingModelsTraining
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb.%N.%J.%u.err # STDERR

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/phoneEmbeddingModelsTraining/training_scripts/hpcDLScriptsPhoneEmbedding/embedding_rnn_model_train.py

