# Workarounds for running SpikingJelly models on p150a via torch_xla (pjrt-plugin-tt 1.5.0.dev20260831000501).
import torch
from torch import nn


class ManualBN2d(nn.Module):
    """Training-mode BatchNorm2d written with mean/var, to avoid the native batch_norm op."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        xhat = (x - mean) * torch.rsqrt(var + self.eps)
        return xhat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
