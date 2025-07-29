#!/bin/bash
#SBATCH --job-name=satvision-training
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --partition=grace
module load miniforge
conda activate 3dclouddownstream

srun python3 3dcloudpipeline.py $1 $2
