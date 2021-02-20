from typing import Sequence, Union

import numpy as np
import torch
import torchcrepe

from ...utils import apply


def extract_f0_with_crepe(
    audios: Sequence[np.ndarray],
    sample_rate: float,
    hop_length_in_samples: int = 128,
    minimum_frequency: float = 50.,
    maximum_frequency: float = 550.,
    full_model: bool = True,
    batch_size: int = 2048,
    device: Union[str, torch.device] = 'cpu',
):
    audios = apply(torch.tensor, audios)
    results = [
        torchcrepe.predict(
            audio,
            sample_rate,
            hop_length_in_samples,
            minimum_frequency,
            maximum_frequency,
            "full" if full_model else "tiny",
            batch_size=batch_size,
            device=device,
            decoder=torchcrepe.decode.weighted_argmax,
        ) for audio in audios
    ]

    return results