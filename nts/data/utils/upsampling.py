from typing import Optional

import gin
import numpy as np
import scipy.interpolate


def get_padded_length(frames: int, window_length: int, hop_length: int):
    return frames * hop_length + window_length - hop_length


def get_source_target_axes(frames: int, window_length: int, hop_length: int):
    padded_length = get_padded_length(frames, window_length, hop_length)
    source_x = np.linspace(0, frames - 1, frames)
    target_x = np.linspace(0, frames - 1, padded_length)
    return source_x, target_x


@gin.configurable
def linear_interpolation(
    signal: np.ndarray,
    target_sr: float,
    window_length: int,
    hop_length: int,
    original_length: Optional[int] = None,
):
    source_x, target_x = get_source_target_axes(signal.size, window_length, hop_length)

    interpolated = np.interp(target_x, source_x, signal)
    if original_length:
        interpolated = interpolated[window_length // 2 :]
        interpolated = interpolated[:original_length]

    return interpolated


@gin.configurable
def cubic_spline_interpolation(
    signal: np.ndarray,
    target_sr: float,
    window_length: int,
    hop_length: int,
    original_length: Optional[int] = None,
):
    source_x, target_x = get_source_target_axes(signal.size, window_length, hop_length)

    interpolant = scipy.interpolate.interp1d(source_x, signal, kind="cubic")
    interpolated = interpolant(target_x)
    if original_length:
        interpolated = interpolated[window_length // 2 :]
        interpolated = interpolated[:original_length]

    return interpolated