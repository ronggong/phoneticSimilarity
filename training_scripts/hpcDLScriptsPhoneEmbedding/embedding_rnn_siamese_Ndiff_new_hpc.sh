#!/bin/bash

#SBATCH -J emb_nd_5
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/phoneEmbeddingModelsTraining
#SBATCH --nodelist=node021
#--gres=gpu:maxwell:1
#SBATCH --mem=40G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_siamese_nd_5.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_siamese_nd_5.%N.%J.%u.err # STDERR

module load Tensorflow/1.5.0-foss-2017a-Python-2.7.12

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/phoneEmbeddingModelsTraining/training_scripts/hpcDLScriptsPhoneEmbedding/embedding_rnn_siamese_train_Ndiff.py

