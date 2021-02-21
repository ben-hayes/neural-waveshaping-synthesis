from typing import Optional

import gin
import numpy as np
import scipy.interpolate
import scipy.signal.windows


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


@gin.configurable
def overlap_add_upsample(
    signal: np.ndarray,
    window_length: int,
    hop_length: int,
    window_fn: str = "hann",
    window_scale: int = 2,
    original_length: Optional[int] = None,
):
    window = scipy.signal.windows.get_window(window_fn, hop_length * window_scale)
    padded_length = get_padded_length(signal.size, window_length, hop_length)
    padded_output = np.zeros(padded_length)

    for i, value in enumerate(signal):
        window_start = i * hop_length
        window_end = window_start + hop_length * window_scale
        padded_output[window_start:window_end] += window * value

    if original_length:
        output = padded_output[(padded_length - original_length) // 2:]
        output = output[:original_length]
    else:
        output = padded_output

    return output
