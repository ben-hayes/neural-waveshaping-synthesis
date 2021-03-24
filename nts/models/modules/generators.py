import math
from typing import Callable

import gin
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
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
    def __init__(self, noise_channels: int = 1, noise_type: str = "gaussian"):
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
    def __init__(
        self, channels: int = 1, noise_type: str = "gaussian", init_range: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.noise_fn = (
            torch.randn_like if noise_type == "gaussian" else torch.rand_like
        )
        self.scale = nn.Parameter(torch.randn(1, channels, 1) * init_range)
        self.shift = nn.Parameter(torch.randn(1, channels, 1) * init_range)

    def forward(self, x: torch.Tensor):
        noise = self.noise_fn(x) * self.scale + self.shift
        return x + noise


class FIRNoiseSynth(nn.Module):
    def __init__(
        self, ir_length: int, hop_length: int, window_fn: Callable = torch.hann_window
    ):
        super().__init__()
        self.ir_length = ir_length
        self.hop_length = hop_length
        self.register_buffer("window", window_fn(ir_length))

    def forward(self, H_re):
        H_im = torch.zeros_like(H_re)
        H_z = torch.complex(H_re, H_im)

        h = torch.fft.irfft(H_z.transpose(1, 2))
        h = h.roll(self.ir_length // 2, -1)
        h = h * self.window.view(1, 1, -1)
        H = torch.fft.rfft(h)

        noise = torch.rand(self.hop_length * H_re.shape[-1] - 1, device=H_re.device)
        X = torch.stft(noise, self.ir_length, self.hop_length, return_complex=True)
        X = X.unsqueeze(0)
        Y = X * H.transpose(1, 2)
        y = torch.istft(Y, self.ir_length, self.hop_length, center=False)
        return y.unsqueeze(1)