import torch.nn as nn

class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        # Ensure input is float32 for numerical stability
        orig_dtype = x.dtype
        return super().forward(x.float()).to(orig_dtype)
