from functools import partial
from typing import Callable, Sequence

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

def normalise_signal(audio: np.ndarray, epsilon: float = 1e-8):
    max_value = max(np.abs(audio).max(), epsilon)
    return audio / max_value


def resample_audio(audio: np.ndarray, original_sr: float, target_sr: float):
    return resampy.resample(audio, original_sr, target_sr)


def segment_signal(
    signal: np.ndarray,
    sample_rate: float,
    segment_length_in_seconds: float,
    hop_length_in_seconds: float,
):
    segment_length_in_samples = int(sample_rate * segment_length_in_seconds)
    hop_length_in_samples = int(sample_rate * hop_length_in_seconds)
    segments = librosa.util.frame(
        signal, segment_length_in_samples, hop_length_in_samples
    )
    return segments


def filter_segments(
    threshold: float,
    key_segments: np.ndarray,
    segments: Sequence[np.ndarray],
):
    mean_keys = key_segments.mean(axis=0)
    mask = mean_keys > threshold
    filtered_segments = apply(lambda x: x[:, mask], segments)
    return filtered_segments


def preprocess_single_audio_file(
    file: str,
    target_sr: float = 16000.0,
    segment_length_in_seconds: float = 4.0,
    hop_length_in_seconds: float = 2.0,
    confidence_threshold: float = 0.85,
    f0_extractor: Callable = extract_f0_with_crepe,
    loudness_extractor: Callable = extract_perceptual_loudness,
    normalise_audio: bool = False,
):
    print("Loading audio file: %s..." % file)
    original_sr, audio = wavfile.read(file)
    audio = convert_to_float32_audio(audio)
    audio = make_monophonic(audio)

    if normalise_audio:
        audio = normalise_signal(audio)

    print("Resampling audio file: %s..." % file)
    audio = resample_audio(audio, original_sr, target_sr)

    print("Extracting f0 with extractor '%s': %s..." % (f0_extractor.__name__, file))
    f0, confidence = f0_extractor(audio)

    print(
        "Extracting loudness with extractor '%s': %s..."
        % (loudness_extractor.__name__, file)
    )
    loudness = loudness_extractor(audio)

    print("Segmenting audio file: %s..." % file)
    segmented_audio = segment_signal(
        audio, target_sr, segment_length_in_seconds, hop_length_in_seconds
    )

    print("Segmenting control signals: %s..." % file)
    segmented_f0 = segment_signal(
        f0, target_sr, segment_length_in_seconds, hop_length_in_seconds
    )
    segmented_confidence = segment_signal(
        confidence, target_sr, segment_length_in_seconds, hop_length_in_seconds
    )
    segmented_loudness = segment_signal(
        loudness, target_sr, segment_length_in_seconds, hop_length_in_seconds
    )

    filtered_audio, filtered_f0, filtered_loudness = filter_segments(
        confidence_threshold,
        segmented_confidence,
        (segmented_audio, segmented_f0, segmented_loudness),
    )

    split = lambda x: [e.squeeze() for e in np.split(x, x.shape[-1], -1)]
    audio_split = split(filtered_audio)
    f0_split = split(filtered_f0)
    loudness_split = split(filtered_loudness)

    return audio_split, f0_split, loudness_split


@gin.configurable
def preprocess_audio(
    files: list,
    target_sr: float = 16000,
    segment_length_in_seconds: float = 4.0,
    hop_length_in_seconds: float = 2.0,
    confidence_threshold: float = 0.85,
    f0_extractor: Callable = extract_f0_with_crepe,
    loudness_extractor: Callable = extract_perceptual_loudness,
    normalise_audio: bool = False,
):
    processor = partial(
        preprocess_single_audio_file,
        target_sr=target_sr,
        segment_length_in_seconds=segment_length_in_seconds,
        hop_length_in_seconds=hop_length_in_seconds,
        f0_extractor=f0_extractor,
        loudness_extractor=loudness_extractor,
        normalise_audio=normalise_audio,
    )
    for file in files:
        yield processor(file)
