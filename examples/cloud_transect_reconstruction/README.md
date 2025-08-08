# Table of Contents

- [Table of Contents](#table-of-contents)
- [3D Cloud Downstream Task](#3d-cloud-downstream-task)
  - [3dcloudpipeline.py](#3dcloudpipelinepy)
    - [Usage](#usage)
    - [Configuration](#configuration)
    - [Logging](#logging)
  - [3dcloudtesting.py](#3dcloudtestingpy)
    - [Usage](#usage-1)
    - [Configuration](#configuration-1)
  - [models.py](#modelspy)
  - [transforms.py](#transformspy)
  - [abidatamodule.py](#abidatamodulepy)
  - [notebooks/](#notebooks)
  - [visualization/](#visualization)
  - [cropping/cropChips.py](#croppingcropchipspy)
- [Data Paths on ADAPT](#data-paths-on-adapt)
  - [Abi Chips](#abi-chips)
  - [CloudSAT Data](#cloudsat-data)
  - [Sample ABI Reconstructions](#sample-abi-reconstructions)
- [Tropics Step-by-step](#tropics-step-by-step)
  - [Data Preparation](#data-preparation)
    - [Chip cropping](#chip-cropping)
    - [Chip Classification](#chip-classification)
  - [Model Training](#model-training)
  - [Model Testing](#model-testing)

# 3D Cloud Downstream Task

## 3dcloudpipeline.py
This script is for training the models.

### Usage
Load the environment
```bash
ssh gpulogin1 # connect to Prism
cd <path to project folder>

# TO CREATE A SLURM JOB TO RUN THE MODEL TRAINING
sbatch training.sh <model_name>

# TO TRAIN THE MODEL LIVE IN TERMINAL
salloc -N1 -G1 -p grace # only add -p grace if you want to use the hopper arm nodes
module load <miniforge|conda> # miniforge or conda depending on if x86 vs arm
conda activate <environmentname> # torch for x86, 3dcclouddownstream for arm
python3 3dcloudpipeline.py <model_name>
```

where `<model_name>` must begin with "sat" or "unet" to use SatVision or U-Net models respectively.
The `<model_name>` will also be the name of the folder in the `checkpoints` folder you save to.

Example:
```bash
python3 3dcloudpipeline.py satfull
```

### Configuration

`SAVE_EVERY_N_EPOCHS` represents how often checkpoints will be saved (5 means 1 checkpoint per 5 epoch).

For loading datasets:
`traindatapath` is the folder where all the training data is.
`TRAINING_SPLIT` will be the percentages of the folder to use.

Example:
`TRAINING_SPLIT = (0, 0.8)` means the first 80% of the data in `traindatapath` will be used as the training dataset.

The same goes for the validation and testing datasets.

`checkpointpath` is where all the checkpoints will be saved.

### Logging

In order to see model results with tensorboard while the model is training, simply use:
```bash
tensorboard --logdir <checkpoint_path>
```
Metrics will also be saved in a csv file in the checkpoints folder.

## 3dcloudtesting.py

Testing suite for the various models trained.

### Usage
Run with `python3 3dcloudtesting.py`

### Configuration
All the configurable parameters are in the file. Change `MODEL_NAMES` to be the list of models you want to test.

`checkpoint_path` will need to be the directory where all the checkpoints are saved.

The script will automatically find the best checkpoint (the one with best in the filename). It will also save a `metrics.csv` file with all the per epoch metrics which can help with producing figures. This file will be under a folder named `version_<verison_number>` in the model checkpoint folder.

## models.py
Contains all the code for the SatVision and UNet model. Any tweaks to model architecture should be done here.

## transforms.py
Contains all the transformations/unit conversions. Any changes done to min-max scaling or unit conversion should be done here.

## abidatamodule.py
Contains all the code for data preprocessing/loading. This, combined with `transforms.py`, will be the files changed if the data preprocessing is ever changed.

## notebooks/
Contains jupyter notebooks with some sample scripts such as SatVision-TOA reconstruction with ABI chips.

## visualization/
Contains jupyter notebooks with some visualizations such as boxplots and predictions.

## cropping/cropChips.py
This is the cropping script, all the directories you need to change are in the top of the file

# Data Paths on ADAPT

## Abi Chips
Saved at `/explore/nobackup/projects/pix4dcloud/szhang16/abiChips`

24000 GOES-16 chips (days 0,1, 5,6, ...) saved in `GOES-16`. Among these 24000 were split into `GOES-16-Tropics` and `GOES-16-MidLatitude`.

200,000+ GOES-16 chips from all days saved in `GOES-16-Redone`

## CloudSAT Data
Saved at `/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat`

## Sample ABI Reconstructions
Saved at `/explore/nobackup/projects/pix4dcloud/szhang16/satvision-toa-reconstructions`

# Tropics Step-by-step

Full step-by-step instructions for training models on the tropics dataset on Prism

## Data Preparation

### Chip cropping

This should already be done by me under `/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16-Redone` which contains 250,000+ chips from GOES-16. If the cropping did not finish all the way this is how to submit a slurm job to crop the chips:

```bash
sbatch cropping/cropChips.sh <year to crop> <day to start on>
```

For example to crop all of 2020:

```bash
sbatch cropping/cropChips.sh 2020 0
```

All the parameters are and should be configured in `cropping/cropChips.py` including the data directories and save directories.

### Chip Classification

All the classification logic is in `tropics/filterChips.py`, it's actually pretty simple once you look at the code.

To classify the chips, simply specify the directory where all the chips are, and the names of the folders you want to save the chips to (in the `filterChips.py` file).

## Model Training

All the model training logic is in `3dcloudpipeline.py`.

Change all the data training paths, more information can be found here: [3dcloudpipeline.py](#3dcloudpipelinepy).

Also make sure to update the training splits. For example if you're using the full testing folder for testing the `TEST_SPLIT = (0, 1)` or if you're using only the first half of the testing folder `TEST_SPLIT = (0, 0.5)`.

## Model Testing

All the model testing logic is in `3dcloudtesting.py`.

You'll want to change `MODEL_NAMES` to be the list of models you want to test, as well as `checkpoint_path` to be the directory where all the checkpoints are saved. More details can be found here: [3dcloudtesting.py](#3dcloudtestingpy).