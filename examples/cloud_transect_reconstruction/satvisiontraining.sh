#!/bin/bash
#SBATCH --job-name=satvision-$1
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
module load anaconda
conda activate torch

srun python3 satvisionpipeline.py $1 $2