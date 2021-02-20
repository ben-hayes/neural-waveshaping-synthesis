from functools import partial
from typing import Callable

import gin
import librosa
import numpy as np
import resampy
import scipy.io.wavfile as wavfile

from .f0_extraction import extract_f0_with_crepe, extract_f0_with_pyin
from .loudness_extraction import extract_perceptual_loudness, extract_rms
from ...utils import apply, apply_unpack, unzip


def read_audio_files(files: list):
    rates_and_audios = apply(wavfile.read, files)
    return unzip(rates_and_audios)


def convert_to_float32_audio(audio: np.ndarray):
    if audio.dtype == np.float32:
        return audio

    max_sample_value = np.iinfo(audio.dtype).max
    floating_point_audio = audio / max_sample_value
    return floating_point_audio.astype(np.float32)


def make_monophonic(audio: np.ndarray, strategy: str = "keep_left"):
    # deal with non stereo array formats
    if len(audio.shape) == 1:
        return audio
    elif len(audio.shape) != 2:
        raise ValueError("Unknown audio array format.")

    # deal with single audio channel
    if audio.shape[0] == 1:
        return audio[0]
    elif audio.shape[1] == 1:
        return audio[:, 0]
    # deal with more than two channels
    elif audio.shape[0] != 2 and audio.shape[1] != 2:
        raise ValueError("Expected stereo input audio but got too many channels.")

    # put channel first
    if audio.shape[1] == 2:
        audio = audio.T

    # make stereo audio monophonic
    if strategy == "keep_left":
        return audio[0]
    elif strategy == "keep_right":
        return audio[1]
    elif strategy == "sum":
        return np.mean(audio, axis=0)
    elif strategy == "diff":
        return audio[0] - audio[1]


def resample_audio(audio: np.ndarray, original_sr: float, target_sr: float):
    return resampy.resample(audio, original_sr, target_sr)


def segment_audio(
    audio: np.ndarray,
    sample_rate: float,
    segment_length_in_seconds: float,
    hop_length_in_seconds: float,
):
    segment_length_in_samples = int(sample_rate * segment_length_in_seconds)
    hop_length_in_samples = int(sample_rate * hop_length_in_seconds)
    segments = librosa.util.frame(
        audio, segment_length_in_samples, hop_length_in_samples
    )
    return segments


@gin.configurable
def preprocess_audio(
    files: list,
    target_sr: float = 16000,
    segment_length_in_seconds: float = 4.0,
    hop_length_in_seconds: float = 2.0,
    f0_extractor: Callable = extract_f0_with_crepe,
    loudness_extractor: Callable = extract_perceptual_loudness,
):
    print("Loading audio files...")
    rates, audios = read_audio_files(files)
    audios = apply(convert_to_float32_audio, audios)
    audios = apply(make_monophonic, audios)

    print("Resampling audio files...")
    resample_to_target = partial(resample_audio, target_sr=target_sr)
    audios = apply_unpack(resample_to_target, list(zip(audios, rates)))

    print("Extracting f0 with extractor '%s'" % f0_extractor.__name__)
    f0s_and_confidences = apply(f0_extractor, audios)
    f0s, confidences = unzip(f0s_and_confidences)

    print("Extracting loudness with extractor '%s'" % loudness_extractor.__name__)
    loudness = apply(loudness_extractor, audios)

    print("Segmenting audio...")
    segment_fn = partial(
        segment_audio,
        sample_rate=target_sr,
        segment_length_in_seconds=segment_length_in_seconds,
        hop_length_in_seconds=hop_length_in_seconds,
    )
    segmented_audio = apply(segment_fn, audios)
