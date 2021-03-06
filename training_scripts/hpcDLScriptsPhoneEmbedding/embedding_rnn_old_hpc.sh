#!/bin/bash

export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate keras_env

#printf "Removing local scratch directories if exist...\n"
#if [ -d /scratch/gopModelsTraining ]; then
#        rm -Rf /scratch/gopModelsTraining
#fi

## Second, replicate the structure of the experiment's folder:
## -----------------------------------------------------------
#mkdir /scratch/gopModelsTraining
#
#
#printf "Copying feature files into scratch directory...\n"
## Third, copy the experiment's data:
## ----------------------------------
#start=`date +%s`
#cp -rp /homedtic/rgong/gopModelsTraining/dataset/feature_gop.h5 /scratch/gopModelsTraining/
#end=`date +%s`
#
#printf "Finish copying feature files into scratch directory...\n"
#printf $((end-start))


#$ -N embed_rnn
#$ -q default.q
#$ -l h=node11

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/phoneEmbeddingModelsTraining/out/embed.$JOB_ID.out
#$ -e /homedtic/rgong/phoneEmbeddingModelsTraining/out/embed.$JOB_ID.err

python /homedtic/rgong/phoneEmbeddingModelsTraining/training_scripts/hpcDLScriptsPhoneEmbedding/embedding_rnn_model_train.py 1

## Clean the crap:
## ---------------
#printf "Removing local scratch directories...\n"
#if [ -d /scratch/gopModelsTraining ]; then
#        rm -Rf /scratch/gopModelsTraining
#fi
#printf "Job done. Ending at `date`\n"
