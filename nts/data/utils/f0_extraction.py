from functools import partial
from typing import Callable, Optional, Sequence, Union

import gin
import librosa
import numpy as np
import torch
import torchcrepe

from .upsampling import linear_interpolation
from ...utils import apply


@gin.configurable
def extract_f0_with_crepe(
    audio: np.ndarray,
    sample_rate: float,
    hop_length_in_samples: int = 128,
    minimum_frequency: float = 50.0,
    maximum_frequency: float = 550.0,
    full_model: bool = True,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
    interpolate_fn: Optional[Callable] = linear_interpolation,
):
    # convert to torch tensor with channel dimension (necessary for CREPE)
    audio = torch.tensor(audio).unsqueeze(0)
    f0, confidence = torchcrepe.predict(
        audio,
        sample_rate,
        hop_length_in_samples,
        minimum_frequency,
        maximum_frequency,
        "full" if full_model else "tiny",
        batch_size=batch_size,
        device=device,
        decoder=torchcrepe.decode.weighted_argmax,
    )

    f0, confidence = f0.numpy(), confidence.numpy()

    if interpolate_fn:
        f0 = interpolate_fn(
            f0, sample_rate, frame_length, hop_length, original_length=audio.size
        )
        confidence = interpolate_fn(
            confidence,
            sample_rate,
            frame_length,
            hop_length,
            original_length=audio.size,
        )

    return f0, confidence


@gin.configurable
def extract_f0_with_pyin(
    audio: np.ndarray,
    sample_rate: float,
    minimum_frequency: float = 65.0,  # recommended minimum freq from librosa docs
    maximum_frequency: float = 2093.0,  # recommended maximum freq from librosa docs
    frame_length: int = 1024,
    hop_length: int = 128,
    fill_na: Optional[float] = None,
    interpolate_fn: Optional[Callable] = linear_interpolation,
):
    f0, _, voiced_prob = librosa.pyin(
        audio,
        sr=sample_rate,
        fmin=minimum_frequency,
        fmax=maximum_frequency,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=fill_na,
    )

    if interpolate_fn:
        f0 = interpolate_fn(
            f0, sample_rate, frame_length, hop_length, original_length=audio.size
        )
        voiced_prob = interpolate_fn(
            voiced_prob,
            sample_rate,
            frame_length,
            hop_length,
            original_length=audio.size,
        )

    return f0, voiced_prob
