from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPad(nn.Module):
    def __init__(self, pad_size: int, lookahead: int = 0):
        super().__init__()
        self.pad_size = pad_size
        self.lookahead = lookahead

    def forward(self, x: torch.Tensor):
        return F.pad(x, (self.pad_size - self.lookahead, self.lookahead))


class Identity(nn.Module):
    def forward(self, x):
        return x


class View(nn.Module):
    def __init__(self, *out_shape):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(*self.out_shape)


class LearnableUpsampler(nn.Module):
    def __init__(
        self,
        window_size: int,
        hop_length: int,
        channels: int = 1,
        window_fn: Callable = torch.hann_window,
    ):
        super().__init__()
        self.window_size = window_size
        self.hop_length = hop_length
        self.channels = channels
        self.window = nn.Parameter(
            window_fn(window_size).view(1, 1, -1).expand(channels, -1, -1) / window_size
        )
        self.padding = (window_size - hop_length) // 2
        # self.pad = CausalPad(window_size - 1)

    def forward(self, x: torch.Tensor):
        # x = self.pad(x)
        return F.conv_transpose1d(
            x,
            self.window,
            stride=self.hop_length,
            groups=self.channels,
            padding=self.padding,
        )
