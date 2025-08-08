#!/bin/bash
#SBATCH --job-name=chip-cropping
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10240
module load anaconda
conda activate ilab-pytorch

python3 cropChips.py $1
