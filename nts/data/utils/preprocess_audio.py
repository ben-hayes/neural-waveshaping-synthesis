from functools import partial

import numpy as np
import resampy
import scipy.io.wavfile as wavfile

from .f0_extraction import extract_f0_with_crepe, extract_f0_with_pyin
from .loudness_extraction import extract_perceptual_loudness, extract_rms
from ...utils import apply, apply_unpack, unzip


def read_audio_files(files: list):
    rates_and_audios = [wavfile.read(file) for file in files]
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


def resample_audio(audio, original_sr, target_sr):
    return resampy.resample(audio, original_sr, target_sr)


def preprocess_audio(
    files: list,
    target_sr: float = 16000,
    f0_extractor: str = "crepe",
    f0_params: dict = {},
    loudness_extractor: str = "perceptual",
    loudness_params: dict = {},
):
    print("Loading audio files...")
    rates, audios = read_audio_files(files)
    audios = apply(convert_to_float32_audio, audios)
    audios = apply(make_monophonic, audios)

    print("Resampling audio files...")
    resample_to_target = partial(resample_audio, target_sr=target_sr)
    audios = apply_unpack(resample_to_target, list(zip(audios, rates)))

    if f0_extractor == "crepe":
        f0_fn = extract_f0_with_crepe
    elif f0_extractor == "pyin":
        f0_fn = extract_f0_with_pyin
    else:
        raise ValueError(
            "Unknown f0 extractor. Supported extractor strings are ('crepe', 'pyin')"
        )
    print("Extracting f0 with extractor '%s'" % f0_extractor)
    extract_f0 = partial(f0_fn, sample_rate=target_sr, **f0_params)
    f0s_and_confidences = apply(extract_f0, audios)

    if loudness_extractor == "perceptual":
        loudness_fn = extract_perceptual_loudness
    elif loudness_extractor == "rms":
        loudness_fn = extract_rms
    else:
        raise ValueError(
            "Unknown loudness extractor. Supported extractor strings are ('perceptual', 'rms')"
        )
    print("Extracting loudness with extractor '%s'" % loudness_extractor)
    extract_loudness = partial(loudness_fn, **loudness_params)
    loudness = apply(extract_loudness, audios)
