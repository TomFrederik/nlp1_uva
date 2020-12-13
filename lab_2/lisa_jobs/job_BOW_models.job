#!/bin/bash

#SBATCH --job-name="tom_bow"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:45:00
#SBATCH --mem=20000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --array=1-4%2
#SBATCH --out=./job_outputs/bow_%A_%a.out


module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12


cd $HOME/nlp1_uva/lab_2/

source activate dl2020
pip install nltk

HPARAMS_FILE=./lisa_params/BOW_models_hparams.txt

srun python3 -u train.py $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
