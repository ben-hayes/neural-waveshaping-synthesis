from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import CausalPad, View


class FiLM(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta


class TimeDistributedMLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, depth: int = 3):
        super().__init__()
        assert depth >= 3, "Depth must be at least 3"
        layers = []
        for i in range(depth):
            layers.append(
                nn.Conv1d(
                    in_size if i == 0 else hidden_size,
                    hidden_size if i < depth - 1 else out_size,
                    1,
                )
            )
            if i < depth - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Dynamic1dConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        conditioning_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        filter_size = in_channels * out_channels * kernel_size

        self.filter_net = TimeDistributedMLP(
            conditioning_size, conditioning_size, filter_size
        )
        self.causal_pad = CausalPad(kernel_size - 1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        filters = self.filter_net(conditioning)
        filters = filters.view(
            conditioning.shape[0],
            conditioning.shape[-1],
            self.out_channels,
            self.in_channels * self.kernel_size,
        )

        x_padded = self.causal_pad(x)
        x_unfolded = F.unfold(x_padded.unsqueeze(2), (1, self.kernel_size))
        x_unfolded = x_unfolded.view(
            x.shape[0], x.shape[-1], self.in_channels * self.kernel_size, -1
        )

        out = torch.matmul(filters, x_unfolded)
        return out.squeeze(3).transpose(1, 2)


class DynamicFFTConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hop_length: int,
        conditioning_size: int,
        window_fn: Callable = torch.hann_window,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.freq_bins = kernel_size // 2 + 1
        self.hop_length = hop_length
        self.register_buffer("window", window_fn(self.kernel_size))

        self.filter_net = nn.Sequential(
            nn.Conv1d(
                conditioning_size,
                conditioning_size * 2,
                kernel_size,
                hop_length,
                kernel_size // 2,
            ),
            nn.Conv1d(
                conditioning_size * 2,
                2 * in_channels * (kernel_size // 2 + 1),
                1,
            ),
        )

        self.tdd = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        filters = self.filter_net(conditioning).view(
            x.shape[0], x.shape[1], self.freq_bins, -1, 2
        )

        X = torch.stft(
            x.view(x.shape[0] * x.shape[1], -1),
            self.kernel_size,
            self.hop_length,
            window=self.window,
            return_complex=False,
        )
        X = X.view(x.shape[0], x.shape[1], self.freq_bins, -1, 2)

        X = X * filters
        out = torch.istft(
            X.view(x.shape[0] * x.shape[1], self.freq_bins, -1, 2),
            self.kernel_size,
            self.hop_length,
            window=self.window,
        )
        out = out.view(x.shape[0], x.shape[1], -1)
        return self.tdd(out)