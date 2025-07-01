import logging
import multiprocessing
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from lightning.pytorch import LightningDataModule

# placeholder for testing
from satvision_pix4d.datasets.abi_temporal_dataset import ABITemporalToyDataset


class ABITemporalDataModule(LightningDataModule):
    def __init__(self, config) -> None:

        super().__init__()

        self.config = config
        self.batch_size = config.DATA.BATCH_SIZE
        self.shuffle = config.DATA.SHUFFLE
        self.num_workers = config.DATA.NUM_WORKERS
        self.persistent_workers = config.DATA.PERSISTENT_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.in_chans = config.MODEL.MAE_VIT.IN_CHANS
        self.train_data_paths = config.DATA.DATA_PATHS
        self.train_data_length = config.DATA.LENGTH
        self.pin_memory = config.DATA.PIN_MEMORY
        self.drop_last = config.DATA.DROP_LAST

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.RandomCrop(self.img_size),
            ]
        )

        self.trainset = None
        self.validset = None

        #print("> Init datasets")
        #self.trainset = ABITemporalToyDataset(
        #    self.data_paths,
        #    split="train",
        #    transform=self.transform,
        #    img_size=img_size,
        #)
        #self.validset = ABITemporalToyDataset(
        #    self.data_paths,
        #    split="valid",
        #    transform=self.transform,
        #    img_size=img_size,
        #)

        #print("Done init datasets")
        #(
        #    self.trainsampler,
        #    self.validsampler,
        #) = self.setup_samplers()

    #def setup_samplers(
    #    self,
    #):
    #    trainsampler = DistributedSampler(
    #        self.trainset,
    #        num_replicas=dist.get_world_size(),
    #        rank=dist.get_rank(),
    #        shuffle=True,
    #    )

    #    validsampler = DistributedSampler(
    #        self.validset,
    #        num_replicas=dist.get_world_size(),
    #        rank=dist.get_rank(),
    #        shuffle=False,
    #    )

    #    return (
    #        trainsampler,
    #        validsampler,
    #    )

    def setup(self, stage=None):
        # This is called after Lightning sets up distributed
        logging.info("> Init datasets")
        self.trainset = ABITemporalToyDataset(
            self.train_data_paths,
            split="train",
            transform=self.transform,
            img_size=self.img_size,
            in_chans=self.in_chans
        )
        self.validset = ABITemporalToyDataset(
            self.train_data_paths,
            split="valid",
            transform=self.transform,
            img_size=self.img_size,
            in_chans=self.in_chans
        )
        logging.info("Done init datasets")
        return

    def train_dataloader(
        self,
    ):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(
        self,
    ):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            # sampler=self.validsampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def plot(*args, **kwargs):
        return None


if __name__ == "__main__":

    toa_module = ABITemporalDataModule(data_path=[])
