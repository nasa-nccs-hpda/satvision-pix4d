import torch


class PerTileMinMaxNormalize:
    def __init__(self, clip=True):
        self.clip = clip

    def __call__(self, x: torch.Tensor):
        # x shape: (T, C, H, W)
        # Flatten spatial dims per timestep and channel
        x_min = x.amin(dim=(-2, -1), keepdim=True)  # per time & channel
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        if self.clip:
            x = torch.clamp(x, 0.0, 1.0)
        return x

