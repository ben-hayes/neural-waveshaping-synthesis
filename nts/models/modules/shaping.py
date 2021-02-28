import torch
import torch.nn as nn

from .activations import MultiActivationBank
from .dynamic import DynamicFFTConv1d, FiLM, TimeDistributedMLP
from .generators import ParallelNoise
from .utils import CausalPad


class NoiseSaturateFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_taps: int = 128,
        noise_channels: int = 1,
        control_embedding_size: int = 32,
        dynamic_filter: bool = False,
    ):
        super().__init__()
        self.conditioning = TimeDistributedMLP(
            control_embedding_size,
            control_embedding_size,
            (in_channels * 2 + noise_channels) * 2,
            depth=3,
        )
        self.film1_size = in_channels
        self.film2_size = in_channels + noise_channels

        self.film1 = FiLM()
        self.saturate = MultiActivationBank(
            (torch.tanh, nn.PReLU(in_channels), torch.abs), in_channels
        )
        self.noise = ParallelNoise(noise_channels)
        self.film2 = FiLM()
        self.padding = CausalPad(filter_taps - 1)

        self.dynamic_filter = dynamic_filter
        if dynamic_filter:
            self.filter = DynamicFFTConv1d(
                in_channels + noise_channels,
                out_channels,
                filter_taps,
                filter_taps // 2,
                control_embedding_size,
            )
        else:
            self.filter = nn.Sequential(
                CausalPad(filter_taps - 1),
                nn.Conv1d(
                    in_channels + noise_channels,
                    out_channels,
                    filter_taps,
                ),
            )

    def _get_conditioning(self, control_embedding):
        conditioning = self.conditioning(control_embedding)
        gamma1 = conditioning[:, : self.film1_size]
        beta1 = conditioning[:, self.film1_size : self.film1_size * 2]
        gamma2 = conditioning[
            :, self.film1_size * 2 : self.film1_size * 2 + self.film2_size
        ]
        beta2 = conditioning[:, self.film1_size * 2 + self.film2_size :]
        return gamma1, beta1, gamma2, beta2

    def forward(self, x: torch.Tensor, control_embedding: torch.Tensor):
        g1, b1, g2, b2 = self._get_conditioning(control_embedding)

        x = self.film1(x, g1, b1)
        x = self.saturate(x)
        x = self.noise(x)
        x = self.film2(x, g2, b2)
        if self.dynamic_filter:
            x = self.filter(x, control_embedding)
        else:
            x = self.filter(x)

        return x