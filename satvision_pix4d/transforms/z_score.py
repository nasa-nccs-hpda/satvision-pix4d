import torch

class PerChannelStandardize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)  # (1,C,1,1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    def __call__(self, x):
        # x is shape (T,C,H,W) as in your dataset
        # apply per-channel standardization
        return (x - self.mean) / self.std
