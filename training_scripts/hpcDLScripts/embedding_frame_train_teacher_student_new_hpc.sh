#!/bin/bash

#SBATCH -J emb_f_2
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
#SBATCH -o /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_frame_ts_2_class.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_frame_ts_2_class.%N.%J.%u.err # STDERR

module load libsndfile/1.0.28-foss-2017a

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/phoneEmbeddingModelsTraining/training_scripts/hpcDLScripts/embedding_frame_train_teacher_student.py
