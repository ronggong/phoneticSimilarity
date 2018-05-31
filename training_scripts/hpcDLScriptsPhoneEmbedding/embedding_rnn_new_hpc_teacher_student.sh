#!/bin/bash

#SBATCH -J att_2
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/phoneEmbeddingModelsTraining
# --gres=gpu:1
# --nodelist=node020
#SBATCH --mem=80G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_rnn_ts_2_class_att.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_rnn_ts_2_class_att.%N.%J.%u.err # STDERR

module load Tensorflow/1.5.0-foss-2017a-Python-2.7.12

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/phoneEmbeddingModelsTraining/training_scripts/hpcDLScriptsPhoneEmbedding/embedding_rnn_model_train_teacher_student.py

