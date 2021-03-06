import gin
import torch
import torch.nn as nn

from .activations import MultiActivationBank
from .dynamic import DynamicFFTConv1d, DynamicSincConv1d, FiLM, TimeDistributedMLP
from .filters import SincConv1d
from .generators import AdditiveNoise, ParallelNoise
from .utils import CausalPad


class Sine(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.sin(x)


class TrainableNonlinearity(nn.Module):
    def __init__(self, channels, width, nonlinearity=nn.ELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels * width, 1, groups=channels),
            nonlinearity(),
            nn.Conv1d(channels * width, channels * width, 1, groups=channels),
            nonlinearity(),
            nn.Conv1d(channels * width, channels, 1, groups=channels),
            nn.Tanh(),
        )

    def set_requires_grad(self, value: bool):
        for layer in self.net:
            if type(layer) != nn.Tanh:
                for p in layer.parameters():
                    p.requires_grad = value

    def forward(self, x):
        return self.net(x)


@gin.configurable
class NoiseSaturateFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_taps: int = 128,
        control_embedding_size: int = 32,
        skip_connection: bool = True,
        filter_type: str = "static",
        **filter_params,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_scale = nn.Parameter(torch.randn(1, out_channels, 1))

        self.conditioning = nn.Conv1d(control_embedding_size, in_channels * 2, 1)
        self.noise = AdditiveNoise(in_channels, init_range=0.001)
        self.film1_size = in_channels
        self.film1 = FiLM()
        self.saturate = TrainableNonlinearity(in_channels, 4)
        self.padding = CausalPad(filter_taps - 1)

        self.dynamic_filter = False
        if filter_type == "dynamic":
            self.dynamic_filter = True
            self.filter = DynamicSincConv1d(
                in_channels,
                out_channels,
                filter_taps,
                filter_taps // 2,
                6,
                control_embedding_size,
                **filter_params,
            )
        elif filter_type == "sinc":
            self.filter = nn.Sequential(
                CausalPad(filter_taps - 1 if filter_taps % 2 == 1 else filter_taps),
                SincConv1d(in_channels, out_channels, filter_taps, 0, **filter_params),
            )
        elif filter_type == "static":
            self.filter = nn.Sequential(
                CausalPad(filter_taps - 1),
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    filter_taps,
                    **filter_params,
                ),
            )
        else:
            raise ValueError("Filter type %s not known." % filter_type)

    def _get_conditioning(self, control_embedding):
        conditioning = self.conditioning(control_embedding)
        gamma1 = conditioning[:, : self.film1_size]
        beta1 = conditioning[:, self.film1_size :]
        return gamma1, beta1

    def forward(self, x: torch.Tensor, control_embedding: torch.Tensor):
        g1, b1 = self._get_conditioning(control_embedding)

        y = self.noise(x)
        y = self.film1(y, g1, b1)
        y = self.saturate(y)
        if self.dynamic_filter:
            # x = self.filter(x, torch.cat((control_embedding, x), dim=1))
            y = self.filter(y, control_embedding)
        else:
            y = self.filter(y)

        return x * self.skip_scale + y if self.skip_connection else y
        