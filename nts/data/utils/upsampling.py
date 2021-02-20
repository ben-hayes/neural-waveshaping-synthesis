from typing import Optional

import numpy as np


def get_padded_length(frames: int, window_length: int, hop_length: int):
    return frames * hop_length + window_length - hop_length


def linear_interpolation(
    signal: np.ndarray,
    target_sr: float,
    window_length: int,
    hop_length: int,
    original_length: Optional[int] = None,
):
    padded_length = get_padded_length(signal.size, window_length, hop_length)
    source_x = np.linspace(0, signal.size - 1, signal.size)
    target_x = np.linspace(0, signal.size - 1, padded_length)

    interpolated = np.interp(target_x, source_x, signal)
    if original_length:
        interpolated = interpolated[window_length // 2:]
        interpolated = interpolated[:original_length]

    return interpolated
