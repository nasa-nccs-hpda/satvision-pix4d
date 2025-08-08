## Directory Contents

Everything found in this GitHub folder can also be found at /explore/nobackup/projects/pix4dcloud/sjaddu.

* `BESTUNET.ckpt`: A model checkpoint file containing the saved weights of the best-performing U-Net model (not in GitHub).
* `bins+statistics.ipynb`: A Jupyter Notebook for analyzing and visualizing statistics, specifically counting and showing the number of pixels of each class type of all the cloud vertical structure masks in a given directory.
* `confusion_matrix.ipynb`: A Jupyter Notebook used to generate and display the confusion matrix.
* `png_visualization.ipynb`: A Jupyter Notebook for showing PNGs.
* `slurm-BESTUNET.out`: The Slurm output file of the best U-Net model I trained, includes the confusion matrix and mIoU/per-class IoUs.
* `test_images/`: A directory containing sample reconstructions or inference masks of the model. They are all PNGs.
* `unet-satvision_training_pipeline-inference.py`: A Python script that defines the complete pipeline for training and inference of the U-Net model with the SatVision encoder.
* `u_net_training_pipeline-inference.py`: A Python script implementing the training and inference pipeline for the normal U-Net model.

