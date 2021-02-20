from functools import partial
from typing import Optional, Sequence, Union

import librosa
import numpy as np
import torch
import torchcrepe

from ...utils import apply


def extract_f0_with_crepe(
    audios: np.ndarray,
    sample_rate: float,
    hop_length_in_samples: int = 128,
    minimum_frequency: float = 50.0,
    maximum_frequency: float = 550.0,
    full_model: bool = True,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
):
    # convert to torch tensor with channel dimension (necessary for CREPE)
    audio = torch.tensor(audio).unsqueeze(0)
    results = torchcrepe.predict(
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

    return results


def extract_f0_with_pyin(
    audios: Sequence[np.ndarray],
    sample_rate: float,
    minimum_frequency: float = 65.0,  # recommended minimum freq from librosa docs
    maximum_frequency: float = 2093.0,  # recommended maximum freq from librosa docs
    frame_length: int = 1024,
    hop_length: int = 128,
    fill_na: Optional[float] = None,
):
    results = [
        librosa.pyin(
            audio,
            sr=sample_rate,
            fmin=minimum_frequency,
            fmax=maximum_frequency,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=fill_na,
        )
        for audio in audios
    ]

    return [(f0, voiced_prob) for f0, _, voiced_prob in results]
