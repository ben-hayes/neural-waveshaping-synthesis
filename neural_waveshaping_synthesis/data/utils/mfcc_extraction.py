import gin
import librosa
import numpy as np


@gin.configurable
def extract_mfcc(
    audio: np.ndarray, sample_rate: float, n_fft: int, hop_length: int, n_mfcc: int
):
    mfcc = librosa.feature.mfcc(
        audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc