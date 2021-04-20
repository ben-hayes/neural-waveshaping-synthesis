import math
from typing import Union

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[int, tuple],
        n_sincs: int,
        fir_window_fn: int = torch.blackman_window,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_sincs = n_sincs

        self.register_buffer("window", fir_window_fn(kernel_size))
        self.register_buffer(
            "time_axis",
            (torch.linspace(0, kernel_size - 1, kernel_size) - kernel_size // 2)
            / kernel_size,
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.width = nn.Parameter(torch.randn(n_sincs, out_channels, in_channels, 1))
        self.amp = nn.Parameter(torch.randn(n_sincs, out_channels, in_channels, 1))

    def forward(self, x: torch.Tensor):
        filters = torch.sinc(self.width * self.time_axis + 1e-6) / self.kernel_size
        filters = self.amp * filters
        filters = filters.sum(0) / self.n_sincs
        filters = filters * self.window
        return F.conv1d(x, filters, bias=self.bias, padding=self.padding)
