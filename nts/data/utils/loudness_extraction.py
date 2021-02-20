from functools import partial
from typing import Sequence
import warnings

import librosa
import numpy as np

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
    weighted_spectrogram = power_spectrogram_in_db + weights
    return weighted_spectrogram


def extract_perceptual_loudness(
    audios: Sequence[np.ndarray],
    sample_rate: float = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    epsilon: float = 1e-5,
):
    power_spec_fn = partial(
        compute_power_spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        epsilon=epsilon,
    )
    power_spectrograms = apply(power_spec_fn, audios)

    perceptual_weighting_fn = partial(
        perform_perceptual_weighting,
        sample_rate=sample_rate,
        n_fft=n_fft,
    )
    perceptually_weighted_spectrograms = apply(
        perceptual_weighting_fn, power_spectrograms
    )

    mean_energy_fn = partial(np.mean, axis=0)
    loudness = apply(mean_energy_fn, perceptually_weighted_spectrograms)

    return loudness


def extract_rms(audio: np.ndarray, window_size: int, hop_length: int):
    frames = librosa.util.frame(audio, window_size, hop_length)
    squared = frames ** 2
    mean = np.mean(squared, axis=0)
    root = np.sqrt(mean)
    return root


def extract_rms_amplitude(
    audios: Sequence[np.ndarray], window_size: int = 2048, hop_length: int = 512
):
    rms_fn = partial(extract_rms, window_size=window_size, hop_length=hop_length)
    rms = apply(rms_fn, audios)
    return rms