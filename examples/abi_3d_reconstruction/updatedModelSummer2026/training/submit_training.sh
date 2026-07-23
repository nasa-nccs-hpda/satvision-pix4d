#!/bin/bash
#SBATCH --job-name=unet3d-summer
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10240
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.err
#SBATCH --export=ALL

echo "Starting training job on $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# --- Container Setup ---
# The project's official Singularity container has all dependencies pre-installed
# (pytorch-lightning, torch, torchmetrics, tensorboard, etc.)
# 
# Build it once on a login node with:
#   module load singularity
#   singularity build --sandbox /lscratch/$USER/container/satvision-pix4d \
#     docker://nasanccs/satvision-pix4d:latest
#
# See the repo README.md for full instructions.

CONTAINER="/lscratch/$USER/container/satvision-pix4d"
WORK_DIR="/home/aliewehr/satvision-pix4d/examples/abi_3d_reconstruction/updatedModelSummer2026"
REPO_ROOT="/home/aliewehr/satvision-pix4d"

# Verify the container exists before trying to run
if [ ! -d "$CONTAINER" ]; then
    echo "ERROR: Singularity container not found at $CONTAINER"
    echo "Build it first with:"
    echo "  module load singularity"
    echo "  singularity build --sandbox $CONTAINER docker://nasanccs/satvision-pix4d:latest"
    exit 1
fi

module load singularity

# Run training inside the Singularity container
#   --nv          : Enable NVIDIA GPU support (passes through CUDA drivers)
#   --env         : Set PYTHONPATH so imports from the repo work
#   -B            : Bind-mount the filesystem paths needed for data and code
#   --pwd         : Set working directory inside the container
singularity exec \
  --nv \
  --env PYTHONPATH="$REPO_ROOT" \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css \
  --pwd "$WORK_DIR" \
  "$CONTAINER" \
  python3 3dcloudpipeline.py

echo "Training completed at $(date)"
