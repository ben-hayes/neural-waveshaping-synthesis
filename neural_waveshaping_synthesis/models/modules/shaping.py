import math

import gin
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from .activations import MultiActivationBank
from .dynamic import DynamicFFTConv1d, DynamicSincConv1d, FiLM, TimeDistributedMLP
from .filters import SincConv1d
from .generators import AdditiveNoise, ParallelNoise
from .utils import CausalPad, LearnableUpsampler


class Sine(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.sin(x)


@gin.configurable
class TrainableNonlinearity(nn.Module):
    def __init__(
        self, channels, width, nonlinearity=nn.ReLU, final_nonlinearity=Sine, depth=3
    ):
        super().__init__()
        self.input_scale = nn.Parameter(torch.randn(1, channels, 1) * 10)
        layers = []
        for i in range(depth):
            layers.append(
                nn.Conv1d(
                    channels if i == 0 else channels * width,
                    channels * width if i < depth - 1 else channels,
                    1,
                    groups=channels,
                )
            )
            layers.append(nonlinearity() if i < depth - 1 else final_nonlinearity())

        self.net = nn.Sequential(*layers)

    # def set_requires_grad(self, value: bool):
    #     for layer in self.net:
    #         if type(layer) != nn.Tanh:
    #             for p in layer.parameters():
    #                 p.requires_grad = value

    def forward(self, x):
        return self.net(self.input_scale * x)


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

        self.upsample = LearnableUpsampler(256, 128, control_embedding_size)
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
        control = self.upsample(control_embedding)
        conditioning = self.conditioning(control)
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


@gin.configurable
class NEWT(nn.Module):
    def __init__(
        self,
        n_waveshapers: int,
        control_embedding_size: int,
        shaping_fn_size: int = 16,
        out_channels: int = 1,
    ):
        super().__init__()

        self.n_waveshapers = n_waveshapers

        self.mlp = TimeDistributedMLP(
            control_embedding_size, control_embedding_size, n_waveshapers * 4, depth=4
        )

        self.waveshaping_index = FiLM()
        self.shaping_fn = TrainableNonlinearity(
            n_waveshapers, shaping_fn_size, nonlinearity=Sine
        )
        self.normalising_coeff = FiLM()

        self.mixer = nn.Sequential(
            nn.Conv1d(n_waveshapers, out_channels, 1),
        )

    def forward(self, exciter, control_embedding):
        film_params = self.mlp(control_embedding)
        film_params = F.upsample(film_params, exciter.shape[-1], mode="linear")
        gamma_index, beta_index, gamma_norm, beta_norm = torch.split(
            film_params, self.n_waveshapers, 1
        )

        x = self.waveshaping_index(exciter, gamma_index, beta_index)
        x = self.shaping_fn(x)
        x = self.normalising_coeff(x, gamma_norm, beta_norm)

        # return x
        return self.mixer(x)


class FastNEWT(NEWT):
    def __init__(
        self,
        newt: NEWT,
        table_size: int = 4096,
        table_min: float = -3.0,
        table_max: float = 3.0,
    ):
        super().__init__()
        self.table_size = table_size
        self.table_min = table_min
        self.table_max = table_max

        self.n_waveshapers = newt.n_waveshapers
        self.mlp = newt.mlp

        self.waveshaping_index = newt.waveshaping_index
        self.normalising_coeff = newt.normalising_coeff
        self.mixer = newt.mixer

        self.lookup_table = self._init_lookup_table(
            newt, table_size, self.n_waveshapers, table_min, table_max
        )

    def _init_lookup_table(
        self,
        newt: NEWT,
        table_size: int,
        n_waveshapers: int,
        table_min: float,
        table_max: float,
    ):
        sample_values = torch.linspace(table_min, table_max, table_size).expand(
            1, n_waveshapers, table_size
        )
        lookup_table = newt.shaping_fn(sample_values)[0]
        return nn.Parameter(lookup_table)

    def _lookup(self, idx):
        return torch.stack(
            [
                torch.stack(
                    [
                        self.lookup_table[shaper, idx[batch, shaper]]
                        for shaper in range(idx.shape[1])
                    ],
                    dim=0,
                )
                for batch in range(idx.shape[0])
            ],
            dim=0,
        )

    def shaping_fn(self, x):
        idx = self.table_size * (x - self.table_min) / (self.table_max - self.table_min)

        lower = torch.floor(idx).long()
        lower[lower < 0] = 0
        lower[lower >= self.table_size] = self.table_size - 1

        upper = lower + 1
        upper[upper >= self.table_size] = self.table_size - 1

        fract = idx - lower
        lower_v = self._lookup(lower)
        upper_v = self._lookup(upper)

        output = (upper_v - lower_v) * fract + lower_v
        return output


@gin.configurable
class Reverb(nn.Module):
    def __init__(self, length_in_seconds, sr):
        super().__init__()
        self.ir = nn.Parameter(torch.randn(1, sr * length_in_seconds - 1) * 1e-6)
        self.register_buffer("initial_zero", torch.zeros(1, 1))
        # self.register_buffer("time", torch.arange(length_in_samples) / sr)

    def forward(self, x):
        ir_ = torch.cat(
            (self.initial_zero, self.ir), dim=-1
        )  # * torch.exp(-self.time * torch.exp(-self.decay))
        if x.shape[-1] > ir_.shape[-1]:
            ir_ = F.pad(ir_, (0, x.shape[-1] - ir_.shape[-1]))
            x_ = x
        else:
            x_ = F.pad(x, (0, ir_.shape[-1] - x.shape[-1]))
        # return (1 - self.wet) * x + self.wet * torch.fft.irfft(
        #     torch.fft.rfft(x_) * torch.fft.rfft(ir_)
        # )
        return (
            x
            + torch.fft.irfft(torch.fft.rfft(x_) * torch.fft.rfft(ir_))[
                ..., : x.shape[-1]
            ]
        )
