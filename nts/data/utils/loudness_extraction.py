from functools import partial
from typing import Callable, Optional, Sequence
import warnings

import gin
import librosa
import numpy as np

from .upsampling import linear_interpolation
from ...utils import apply


def compute_power_spectrogram(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: str,
    epsilon: float,
):
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    magnitude_spectrogram = np.abs(spectrogram)
    power_spectrogram = librosa.amplitude_to_db(
        magnitude_spectrogram, ref=np.max, amin=epsilon
    )
    return power_spectrogram


def perform_perceptual_weighting(
    power_spectrogram_in_db: np.ndarray, sample_rate: float, n_fft: int
):
    centre_frequencies = librosa.fft_frequencies(sample_rate, n_fft)

    # We know that we will get a log(0) warning here due to the DC component -- we can
    # safely ignore as it is clipped to the default min dB value of -80.0 dB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = librosa.A_weighting(centre_frequencies)

    weights = np.expand_dims(weights, axis=1)
    weighted_spectrogram = power_spectrogram_in_db #+ weights
    return weighted_spectrogram


@gin.configurable
def extract_perceptual_loudness(
    audio: np.ndarray,
    sample_rate: float = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    epsilon: float = 1e-5,
    interpolate_fn: Optional[Callable] = linear_interpolation,
    normalise: bool = True,
):
    power_spectrogram = compute_power_spectrogram(
        audio, n_fft=n_fft, hop_length=hop_length, window=window, epsilon=epsilon
    )
    perceptually_weighted_spectrogram = perform_perceptual_weighting(
        power_spectrogram, sample_rate=sample_rate, n_fft=n_fft
    )
    loudness = np.mean(perceptually_weighted_spectrogram, axis=0)
    if interpolate_fn:
        loudness = interpolate_fn(
            loudness, n_fft, hop_length, original_length=audio.size
        )
    
    if normalise:
        loudness = (loudness + 80) / 80

    return loudness


@gin.configurable
def extract_rms(
    audio: np.ndarray,
    window_size: int = 2048,
    hop_length: int = 512,
    sample_rate: Optional[float] = 16000.,
    interpolate_fn: Optional[Callable] = linear_interpolation,
):
    # pad audio to centre frames
    padded_audio = np.pad(audio, (window_size // 2, window_size // 2))
    frames = librosa.util.frame(padded_audio, window_size, hop_length)
    squared = frames ** 2
    mean = np.mean(squared, axis=0)
    root = np.sqrt(mean)
    if interpolate_fn:
        assert sample_rate is not None, "Must provide sample rate if upsampling"
        root = interpolate_fn(
            root, window_size, hop_length, original_length=audio.size
        )
    
    return root