import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from transforms import SimpleMinMaxScale


class AbiToaTransform:
    """
    Transforms the input imagery (7, 512, 512, 16) into a format 
    suitable for a 3D CNN: (16, 7, 512, 512)
    """
    def __init__(self):
        self.transform_img = SimpleMinMaxScale()

    def __call__(self, img):
        # Apply scaling
        img = self.transform_img(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).float()
        
        # Permute from (Time, Height, Width, Channels) to (Channels, Time, Height, Width)
        # (7, 512, 512, 16) -> (16, 7, 512, 512)
        img = img.permute(3, 0, 1, 2)
        
        return img


class AbiChipDataset(Dataset):
    def __init__(self, chip_paths):
        self.chip_paths = chip_paths
        self.transform = AbiToaTransform()

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):
        chip = np.load(self.chip_paths[idx], allow_pickle=True)
        # Use 'ABI/chip' key based on the new chip format, fallback to 'chip' just in case
        image = chip['ABI/chip'] if 'ABI/chip' in chip else chip['chip']
        
        if self.transform is not None:
            image = self.transform(image)
            
        # Extract necessary arrays for downsampling
        # Use cloud_class instead of binary_mask for 9-class prediction
        mask = torch.from_numpy(chip['CloudSat/cloud_class']).long()
        rows = chip['CloudSat/abi_row']
        cols = chip['CloudSat/abi_column']
        
        current_len = mask.shape[0]
        target_len = 478
        num_to_drop = current_len - target_len
        
        if num_to_drop > 0:
            seen = set()
            dup_indices = []
            unique_indices = []
            
            for i in range(current_len):
                pair = (rows[i], cols[i])
                if pair in seen:
                    dup_indices.append(i)
                else:
                    seen.add(pair)
                    unique_indices.append(i)
            
            indices_to_drop = []
            
            if len(dup_indices) >= num_to_drop:
                # We have more duplicates than we need to drop. Drop a subset of dupes.
                drop_idx = np.round(np.linspace(0, len(dup_indices) - 1, num_to_drop)).astype(int)
                indices_to_drop = [dup_indices[i] for i in drop_idx]
            else:
                # Drop all duplicates
                indices_to_drop.extend(dup_indices)
                
                # Drop remaining footprints from unique footprints
                remaining_to_drop = num_to_drop - len(dup_indices)
                if remaining_to_drop > 0:
                    extra_drop_idx = np.round(np.linspace(0, len(unique_indices) - 1, remaining_to_drop)).astype(int)
                    indices_to_drop.extend([unique_indices[i] for i in extra_drop_idx])
            
            # Create list of indices to keep
            indices_to_drop = set(indices_to_drop)
            keep_indices = [i for i in range(current_len) if i not in indices_to_drop]
            
            # Apply to mask
            mask = mask[keep_indices]
        
        # We do NOT add a channel dimension. 
        # CrossEntropyLoss expects targets of shape (H, W) with class integers.

        return {"chip": image, "mask": mask}


class AbiDataModule(L.LightningDataModule):
    def __init__(self, data_path, 
                 train_split=(0, 0.8), val_split=(0.8, 0.9), test_split=(0.9, 1.0), 
                 batch_size=4, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Glob all .npz files from all subdirectories (jan_chips, feb_chips, etc.)
        all_chips = sorted(glob.glob(self.data_path + "/*/*.npz"))
        
        train_start = int(len(all_chips) * self.train_split[0])
        train_end = int(len(all_chips) * self.train_split[1])
        self.train_dataset = AbiChipDataset(all_chips[train_start:train_end])

        val_start = int(len(all_chips) * self.val_split[0])
        val_end = int(len(all_chips) * self.val_split[1])
        self.val_dataset = AbiChipDataset(all_chips[val_start:val_end])

        test_start = int(len(all_chips) * self.test_split[0])
        test_end = int(len(all_chips) * self.test_split[1])
        self.test_dataset = AbiChipDataset(all_chips[test_start:test_end])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
