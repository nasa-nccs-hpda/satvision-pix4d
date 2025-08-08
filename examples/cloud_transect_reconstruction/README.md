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

# Data Paths on ADAPT

## Abi Chips
Saved at `/explore/nobackup/projects/pix4dcloud/szhang16/abiChips`

## CloudSAT Data
Saved at `/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat`

## Sample ABI Reconstructions
Saved at `/explore/nobackup/projects/pix4dcloud/szhang16/satvision-toa-reconstructions`
