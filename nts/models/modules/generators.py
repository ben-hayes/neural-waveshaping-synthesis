import math

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wavetable(nn.Module):
    def __init__(
        self,
        wavetable_length: int,
        n_wavetables: int = 1,
        sample_rate: float = 16000,
        initialisation: str = "sine",
    ):
        super().__init__()
        self.wavetable_length = wavetable_length
        self.sr = sample_rate

        if initialisation == "sine":
            sinusoid = torch.sin(
                math.tau
                * torch.ones(wavetable_length, n_wavetables).cumsum(0)
                / wavetable_length
            )
            self.wavetable = nn.Parameter(sinusoid[None, :, :, None])
        else:
            self.wavetable = nn.Parameter(
                torch.rand(1, wavetable_length, n_wavetables, 1)
            )

    def forward(self, f0: torch.Tensor):
        phase = (f0.cumsum(-1) / self.sr)[:, :, None, :] % 1
        wt_phase = torch.linspace(0.0, 1.0, self.wavetable_length, device=f0.device)[
            None, :, None, None
        ]
        diff = torch.abs(phase - wt_phase) * self.wavetable_length
        weights = F.relu(1 - diff)
        weighted_wavetable = self.wavetable * weights
        return weighted_wavetable.sum(1)


class ParallelNoise(nn.Module):
    def __init__(self, noise_channels: int = 1, noise_type: str = 'gaussian'):
        super().__init__()
        self.noise_channels = noise_channels
        self.noise_fn = torch.randn if noise_type == "gaussian" else torch.rand

    def forward(self, x: torch.Tensor):
        noise = self.noise_fn(
            1,
            self.noise_channels,
            x.shape[-1],
            device=x.device,
            requires_grad=x.requires_grad,
        ).expand(x.shape[0], -1, -1)
        noise = noise * 2 - 1
        noise = noise
        return torch.cat((noise, x), dim=1)


class AdditiveNoise(nn.Module):
    def __init__(self, channels: int = 1, noise_type: str = 'gaussian', init_range: float = 0.1):
        super().__init__()
        self.channels = channels
        self.noise_fn = torch.randn_like if noise_type == "gaussian" else torch.rand_like
        self.scale = nn.Parameter(torch.randn(1, channels, 1) * init_range)
        self.shift = nn.Parameter(torch.randn(1, channels, 1) * init_range)
    
    def forward(self, x: torch.Tensor):
        noise = self.noise_fn(x) * self.scale + self.shift
        return x + noise