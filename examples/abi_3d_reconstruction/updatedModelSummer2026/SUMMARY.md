# 2026 Cloud Transect Reconstruction Pipeline Summary

This document summarizes the complete overhaul of the `cloud_transect_reconstruction` pipeline to support the new 2026 satellite chip format.

## Overview of Changes

The original pipeline processed 14-channel, 2D chips (128x128). The pipeline has been completely rewritten to support the new 2026 chips which are **4D Tensors: `(7 timesteps, 512 Height, 512 Width, 16 Channels)`**.

### 1. Simplified Normalization (`transforms.py`)
*   **Action**: Stripped out the complex, physics-based radiometric calibration (ESUN, Planck constants) which was hardcoded for only 14 channels.
*   **Result**: Replaced with a fast, deep-learning standard `SimpleMinMaxScale`.
*   **Note**: We built a high-performance multiprocessing Jupyter notebook (`calculate_min_max.ipynb`) to scan all 21,638 chips to find the true global minimums and maximums across the dataset to plug into this transform.

### 2. Dataloader & On-The-Fly Downsampling (`abidatamodule.py`)
*   **Data Aggregation**: Automatically globs all 21k `.npz` chips recursively from 12 distinct month directories and handles the 80/10/10 Train/Val/Test splitting dynamically.
*   **Key Updates**: Extracts the new 16-channel `ABI/chip` matrix and the 9-class `CloudSat/cloud_class` mask. 
*   **Dynamic Transect Downsampling Algorithm**: 
    *   Instead of rewriting terabytes of `.npz` files, the dataloader performs in-memory downsampling.
    *   It identifies overlapping CloudSat footprints (caused by ABI pixel distortion) using the `CloudSat/abi_row` and `CloudSat/abi_column` arrays.
    *   It logically drops these exact (row, col) duplicates, and if necessary, drops an evenly spaced subset of remaining footprints to bring the transect length to exactly **`478`**.

### 3. Model Architecture (`models.py`)
*   **Action**: Upgraded the standard 2D U-Net to a baseline **3D U-Net (`UNET3D`)**.
*   **Architecture**: Uses `Conv3d` and `MaxPool3d` to ingest the temporal (`T=7`) dimension. A final 3D convolution collapses the time dimension before passing the features to a 2D prediction head, outputting a spatial mask of `(478, 40)`.
*   **Multi-Class Segmentation**: 
    *   Switched from binary predictions to 9-class segmentation (`num_classes=9`).
    *   Utilizes `nn.CrossEntropyLoss(ignore_index=-1)` and `JaccardIndex(task="multiclass", ignore_index=-1)` to accurately penalize the model while ignoring any "-1" (missing retrieval) CloudSat labels.

### 4. Training & Execution (`3dcloudpipeline.py`)
*   A clean PyTorch Lightning training script that stitches the updated datamodule and `UNET3D` model together.
*   **OOM Prevention**: The script is configured with `BATCH_SIZE = 1` to prevent CUDA Out Of Memory errors when passing the massive 4D tensors through the 3D convolutions on 32GB GPUs. It uses `accumulate_grad_batches=4` in PyTorch Lightning to simulate an effective batch size of 4 for training stability without the memory overhead.
*   **Logging**: Uses `CSVLogger` (omitted TensorBoard to avoid dependency issues on the cluster) to log epoch metrics directly into `./checkpoints/unet3d_baseline/`.
*   **Imports**: Upgraded all `import lightning` syntax to `import pytorch_lightning` to support the specific older Lightning package installed in the ADAPT environment.

### 5. Slurm Submission (`submit_training.sh`)
*   A custom Slurm script engineered for the NASA ADAPT cluster.
*   **Environment**: Actively bypasses the Singularity container (which wasn't built locally) in favor of seamlessly loading a pre-configured, working Conda environment (`ilab-pytorch`).
*   **Architecture Matching**: Intentionally avoids the `grace` partition (which uses ARM architecture incompatible with the x86 `ilab-pytorch` python binary) and allows Slurm to auto-assign a standard x86 GPU node instead.
*   **Direct Execution**: Runs `python3` natively inside the activated shell (avoiding `srun`) to guarantee the Conda environment variables are correctly inherited by the training process.
