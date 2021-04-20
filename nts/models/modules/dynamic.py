import math
from typing import Callable

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from .utils import CausalPad, View


class FiLM(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta


class TimeDistributedLayerNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size)
    
    def forward(self, x):
        return self.layer_norm(x.transpose(1, 2)).transpose(1, 2)


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
                layers.append(TimeDistributedLayerNorm(hidden_size))
                layers.append(nn.LeakyReLU())
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer("window", window_fn(self.kernel_size))

        self.bias = nn.Parameter(
            torch.randn(1, self.out_channels, 1), requires_grad=True
        )

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
                2 * in_channels * out_channels * self.freq_bins,
                1,
            ),
        )

    def _reshape_filters(self, filters: torch.Tensor):
        filters = torch.stack(torch.split(filters, self.freq_bins, dim=1), dim=1)
        filters = torch.stack(torch.split(filters, self.in_channels, dim=1), dim=1)
        filters = torch.stack(torch.split(filters, self.out_channels, dim=1), dim=-1)
        return filters

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        filters = self.filter_net(conditioning)
        filters = self._reshape_filters(filters)

        X = torch.stft(
            x.view(x.shape[0] * x.shape[1], -1),
            self.kernel_size,
            self.hop_length,
            window=self.window,
            return_complex=False,
        )
        X = X.view(x.shape[0], 1, self.in_channels, self.freq_bins, -1, 2)

        X = X * filters
        X = X.sum(2)

        out = torch.istft(
            X.view(x.shape[0] * self.out_channels, self.freq_bins, -1, 2),
            self.kernel_size,
            self.hop_length,
            window=self.window,
        )
        out = out.view(x.shape[0], self.out_channels, -1)
        return out + self.bias


class DynamicSincConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hop_length: int,
        n_sincs: int,
        conditioning_size: int,
        ola_window_fn: Callable = torch.hann_window,
        fir_window_fn: Callable = torch.blackman_window,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.freq_bins = kernel_size // 2 + 1
        self.hop_length = hop_length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_sincs = n_sincs
        self.epsilon = eps
        self.register_buffer("ola_window", ola_window_fn(self.kernel_size))
        self.register_buffer("fir_window", fir_window_fn(self.kernel_size))

        self.register_buffer(
            "time_axis",
            (torch.linspace(0, kernel_size - 1, kernel_size) - kernel_size // 2)
            / kernel_size,
        )

        self.bias = nn.Parameter(
            torch.randn(1, self.out_channels, 1), requires_grad=True
        )

        self.filter_net = nn.Sequential(
            nn.Conv1d(
                conditioning_size,
                conditioning_size,
                kernel_size,
                hop_length,
                kernel_size // 2,
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                conditioning_size,
                in_channels * out_channels * n_sincs * 2,
                1,
            ),
            nn.Tanh(),
        )

    def _reshape_params(self, params: torch.Tensor):
        params = torch.stack(torch.split(params, self.n_sincs, dim=1), dim=1)
        params = torch.stack(torch.split(params, self.in_channels, dim=1), dim=1)
        params = torch.stack(torch.split(params, self.out_channels, dim=1), dim=-1)
        return params

    def _generate_filters(self, sinc_params: torch.Tensor):
        amplitude, width = torch.split(sinc_params, 1, dim=-1)

        filters = (
            torch.sinc(width * self.time_axis + self.epsilon) / self.kernel_size
        )
        filters = amplitude * filters
        filters = filters.sum(3)
        filters = filters * self.fir_window / self.n_sincs

        fft_filters = torch.fft.rfft(filters, dim=-1).transpose(-1, -2)
        return fft_filters

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        sinc_params = self.filter_net(conditioning)
        sinc_params = self._reshape_params(sinc_params)
        filters = self._generate_filters(sinc_params)

        X = torch.stft(
            x.view(x.shape[0] * x.shape[1], -1),
            self.kernel_size,
            self.hop_length,
            window=self.ola_window,
            return_complex=True,
        )
        X = X.view(x.shape[0], 1, self.in_channels, self.freq_bins, -1)

        X = X * filters
        X = X.sum(2)

        out = torch.istft(
            X.view(x.shape[0] * self.out_channels, self.freq_bins, -1),
            self.kernel_size,
            self.hop_length,
            window=self.ola_window,
        )
        out = out.view(x.shape[0], self.out_channels, -1)
        return out + self.bias