#!/bin/bash
#SBATCH --job-name=unet-training
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
module load anaconda
conda activate torch

srun python3 unetpipeline.py
