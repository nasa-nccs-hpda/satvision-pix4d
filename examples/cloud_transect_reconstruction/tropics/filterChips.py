import numpy as np
import os
import shutil

CHIP_DIR = "/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/"
TROPIC_SAVE_DIR = "/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16-Tropics/"
MIDLATITUDE_SAVE_DIR = "/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16-MidLatitude/"

for entry in os.scandir(CHIP_DIR):
    if entry.is_file():
        chip = np.load(entry.path, allow_pickle=True)
        chip_data = chip['data']
        lat_data = chip_data.item()['Latitude']
        lat = lat_data[45] #get the middle latitude

        if lat >= -25 and lat <= 25:
            # Tropic
            shutil.copyfile(entry.path, TROPIC_SAVE_DIR + entry.name)

        elif (lat >= -40 and lat <= -25) or (lat >= 25 and lat <= 40):
            shutil.copyfile(entry.path, MIDLATITUDE_SAVE_DIR + entry.name)