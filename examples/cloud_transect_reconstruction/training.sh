#!/bin/bash
#SBATCH --job-name=satvision-training
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=grace

mv /explore/nobackup/people/szhang16/checkpoints/* /explore/nobackup/people/szhang16/checkpoints-old

module load miniforge
conda activate 3dclouddownstream

srun python3 3dcloudpipeline.py $1 $2
