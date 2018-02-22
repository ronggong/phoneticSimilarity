#!/bin/bash

export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/phoneEmbeddingModelsTraining ]; then
        rm -Rf /scratch/phoneEmbeddingModelsTraining
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/phoneEmbeddingModelsTraining


#$ -N emb_frame
#$ -q default.q
#$ -l debian8
#$ -l h=node11

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_frame.$JOB_ID.out
#$ -e /homedtic/rgong/phoneEmbeddingModelsTraining/out/emb_frame.$JOB_ID.err

python /homedtic/rgong/phoneEmbeddingModelsTraining/training_scripts/hpcDLScripts/embedding_frame_train.py

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/phoneEmbeddingModelsTraining ]; then
        rm -Rf /scratch/phoneEmbeddingModelsTraining
fi
printf "Job done. Ending at `date`\n"
