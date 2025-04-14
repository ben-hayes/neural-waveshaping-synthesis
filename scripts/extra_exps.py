import csv
import os
import random
from pathlib import Path
from typing import Dict, Generator, Literal, Optional, Tuple

import fire
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as Fta
from loguru import logger
from tqdm import tqdm

####################
# Utilities
####################


def interpolate_params(params: torch.Tensor, n_samples: int) -> torch.Tensor:
    return nn.functional.interpolate(params[None, None], n_samples, mode="linear")[0, 0]


####################
# Define model
####################


class Sin(nn.Module):
    def __init__(self, bandwidth: float = 1.0):
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x * self.bandwidth)


class NEWT(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int = 1,
        initial_bandwidth: float = 1.0,
        later_bandwidth: float = 1.0,
        oversample: float = 1.0,
        act: Literal["sin", "relu", "sinrelu"] = "sin",
    ):
        super().__init__()

        initial_act = nn.ReLU() if act == "relu" else Sin(initial_bandwidth)
        later_act = Sin(later_bandwidth) if act == "sin" else nn.ReLU()

        hiddens = []
        for _ in range(depth):
            hiddens.append(nn.Linear(width, width))
            hiddens.append(later_act)

        self.net = nn.Sequential(
            nn.Linear(1, width),
            initial_act,
            *hiddens,
            nn.Linear(width, 1),
        )
        self.oversample = oversample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.oversample > 1.0:
            x = Fta.resample(
                waveform=x,
                orig_freq=1.0,
                new_freq=self.oversample,
            )
            x = self.net(x[..., None])[..., 0]
            x = Fta.resample(
                waveform=x,
                orig_freq=self.oversample,
                new_freq=1.0,
            )

        else:
            # Native rate pass
            x = self.net(x[..., None])[..., 0]
        return x


class LearnableWaveshaper(nn.Module):
    """
    A learnable waveshaper that:
      1. Interpolates per-segment scaling/offset parameters across the input signal.
      2. Optionally oversamples the input, feeds it through a NEWT network,
         then downsamples back.
      3. Applies per-segment scaling/offset to the output as well.
    """

    def __init__(
        self,
        signal_length_samples: int,
        hop_length: int = 128,
        width: int = 128,
        depth: int = 4,
        initial_bandwidth: float = 1.0,
        later_bandwidth: float = 1.0,
        oversample: float = 1.0,
        activation: Literal["sin", "relu", "sinrelu"] = "sin",
    ):
        """
        Args:
            signal_length_samples (int): The total number of samples in the signal.
            window_length (int): Used to compute how many parameter segments we'll have.
            hop_length (int): Stride used to define each segment (for interpolation).
            initial_bandwidth (float): Bandwidth for the first Sin in NEWT.
            later_bandwidth (float): Bandwidth for subsequent Sins in NEWT.
            oversample (float): If > 1, we'll oversample the signal using torchaudio.resample
                                before sending it through NEWT.
        """
        super().__init__()
        self.signal_length_samples = signal_length_samples
        self.hop_length = hop_length

        # The number of segments used for parameter interpolation
        seq_length = signal_length_samples // hop_length

        # Learnable per-segment parameters
        self.alpha_in = nn.Parameter(torch.ones(seq_length))
        self.beta_in = nn.Parameter(torch.zeros(seq_length))
        self.alpha_out = nn.Parameter(torch.ones(seq_length))
        self.beta_out = nn.Parameter(torch.zeros(seq_length))

        # The shaping network
        self.newt = NEWT(
            width=width,
            depth=depth - 2,
            initial_bandwidth=initial_bandwidth,
            later_bandwidth=later_bandwidth,
            oversample=oversample,
            act=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input signal, shape (batch, time) or (time,).

        Returns:
            torch.Tensor: Waveshaped signal, same shape as input.
        """
        # Interpolate alpha_in, beta_in
        alpha_in_full = interpolate_params(self.alpha_in, self.signal_length_samples)
        beta_in_full = interpolate_params(self.beta_in, self.signal_length_samples)

        # Apply input scaling/offset
        x = x * alpha_in_full + beta_in_full

        x = self.newt(x)

        # Interpolate alpha_out, beta_out
        alpha_out_full = interpolate_params(self.alpha_out, self.signal_length_samples)
        beta_out_full = interpolate_params(self.beta_out, self.signal_length_samples)

        # Apply output scaling/offset
        x = x * alpha_out_full + beta_out_full

        return x


####################
# Data loading
####################

DATA_ROOT = "/import/c4dm-datasets/URMP/synth-dataset/4s-dataset/"


def load_segment(
    instrument: str, index: Optional[int] = None, data_root: str = DATA_ROOT
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_path = Path(data_root)
    instrument_path = data_path / instrument
    f0_mean = np.load(instrument_path / "data_mean.npy")[0]
    f0_std = np.load(instrument_path / "data_std.npy")[0]

    audio_path = instrument_path / "test/audio"
    audio_files = list(audio_path.glob("*.npy"))
    audio_files = sorted(audio_files)  # for determinism

    if index is not None:
        audio_file = audio_files[index]
    else:
        audio_file = random.choice(audio_files)

    audio_file = str(audio_file)

    control_file = audio_file.replace("audio", "control")
    audio = np.load(str(audio_file))

    f0 = np.load(control_file)[0]
    f0 = f0 * f0_std + f0_mean
    audio = torch.from_numpy(audio).float()
    f0 = torch.from_numpy(f0).float()

    return audio, f0


def make_input_signal(audio: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    f0 = nn.functional.interpolate(f0[None, None], audio.shape[-1], mode="linear")[0, 0]
    phase = torch.cumsum(f0 / 16000, dim=-1)
    x = torch.sin(2 * np.pi * phase)

    return x


def iterate_segments(
    instrument: str, n_segments: int, data_root: str = DATA_ROOT
) -> Generator:
    for i in range(n_segments):
        audio, f0 = load_segment(instrument, index=i, data_root=data_root)
        input_signal = make_input_signal(audio, f0)
        yield audio, f0, input_signal


####################
# Test metrics
####################

MEL_PARAMS = [
    (10, 5, 32),
    (25, 10, 64),
    (100, 50, 128),
]


def compute_mel_specs(y: np.ndarray, sample_rate: float = 16000.0):
    mel_specs = []
    for window_size, hop_size, n_mels in MEL_PARAMS:
        window_size = int(window_size * sample_rate / 1000.0)
        hop_size = int(hop_size * sample_rate / 1000.0)

        spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=window_size,
            hop_length=hop_size,
            window="hann",
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        mel_specs.append(spec_db)

    return mel_specs


def compute_mss(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 16000.0
) -> float:
    logger.info("Computing MSS...")
    target_specs = compute_mel_specs(target, sample_rate)
    pred_specs = compute_mel_specs(pred, sample_rate)

    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += np.mean(np.abs(target_spec - pred_spec))

    dist = dist / len(target_specs)
    return dist


def compute_mfcc(target: np.ndarray, sample_rate: float = 16000.0) -> np.ndarray:
    window_length = int(0.05 * sample_rate)
    hop_length = int(0.01 * sample_rate)

    mfcc = librosa.feature.mfcc(
        y=target,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=window_length,
        hop_length=hop_length,
        n_mels=128,
    )

    return mfcc


def compute_mfcc_distance(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 16000.0
) -> float:
    logger.info("Computing MFCC distance...")
    target_mfcc = compute_mfcc(target, sample_rate)
    pred_mfcc = compute_mfcc(pred, sample_rate)

    dist = np.mean(np.abs(target_mfcc - pred_mfcc))
    return dist


def compute_rms(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 16000.0
) -> float:
    logger.info("Computing amp env...")
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.025 * sample_rate)

    target_rms = librosa.feature.rms(
        y=target, frame_length=win_length, hop_length=hop_length
    )
    pred_rms = librosa.feature.rms(
        y=pred, frame_length=win_length, hop_length=hop_length
    )

    target_norm = np.linalg.vector_norm(target_rms, axis=-1, ord=2)
    pred_norm = np.linalg.vector_norm(pred_rms, axis=-1, ord=2)

    cosine_sim = np.dot(target_rms[0], pred_rms[0]) / (target_norm * pred_norm)

    return cosine_sim.mean()


@torch.no_grad()
def compute_tm_and_td(
    target: np.ndarray,
    model: nn.Module,
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
    f0_min: float,
    f0_max: float,
    frame_length: int = 1024,
    sample_rate: float = 16000.0,
    n_samples: int = 10_000,
    batch_size: int = 1024,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    alphas = torch.empty(n_samples, device=device).uniform_(alpha_min, alpha_max)
    betas = torch.empty(n_samples, device=device).uniform_(beta_min, beta_max)
    f0s = torch.empty(n_samples, device=device).uniform_(f0_min, f0_max)

    input_phase = (
        torch.arange(frame_length, device=device)[None]
        * 2
        * torch.pi
        * f0s[:, None]
        / sample_rate
    )
    input_signals = torch.sin(input_phase)
    input_signals = input_signals * alphas[:, None] + betas[:, None]

    batches = input_signals.split(batch_size)
    output_signals = torch.cat(
        [model(batch) for batch in tqdm(batches, desc="Inference")], dim=0
    )
    output_signals = output_signals.detach().cpu().numpy()

    target_mfcc = librosa.feature.mfcc(
        y=target,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=frame_length,
        hop_length=int(frame_length // 4),
        n_mels=128,
    ).transpose(-1, -2)
    output_mfcc = librosa.feature.mfcc(
        y=output_signals,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=frame_length,
        hop_length=frame_length,
        center=False,
        n_mels=128,
    )[..., 0]

    l1_dists = np.abs(target_mfcc[:, None] - output_mfcc[None, :]).mean(axis=-1)
    min_l1_dists = np.min(l1_dists, axis=0)
    tm = np.mean(min_l1_dists)

    S = np.argmin(l1_dists, axis=0)
    S_set = np.unique(S)
    td = S_set.size / target_mfcc.shape[0]

    return tm, td


def compute_tc(
    model: nn.Module,
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
    f0: float,
    frame_length: int = 1024,
    sample_rate: float = 16000.0,
    n_samples: int = 10_000,
    device: str = "cuda",
):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=20,
        log_mels=True,
        melkwargs={"n_fft": frame_length, "center": False},
    )

    alphas = torch.empty(n_samples, device=device).uniform_(alpha_min, alpha_max)
    betas = torch.empty(n_samples, device=device).uniform_(beta_min, beta_max)
    alphabetas = torch.stack((alphas, betas), dim=-1).requires_grad_(True)
    input_phase = (
        torch.arange(frame_length, device=device)[None]
        * 2
        * torch.pi
        * f0
        / sample_rate
    )
    input_signals = torch.sin(input_phase)

    def func(alphabetas):
        alphas = alphabetas[..., 0]
        betas = alphabetas[..., 1]

        x = input_signals * alphas + betas

        output_signals = model(x)
        mfccs = transform(output_signals).squeeze()
        return mfccs

    jac = torch.func.vmap(torch.func.jacrev(func))(alphabetas)
    _, S, _ = torch.linalg.svd(jac)
    tc = S.max(dim=-1).values.mean().detach()

    return tc.item()


####################
# Loss function
####################


def compute_stft(
    x: torch.Tensor, n_fft: int = 1024, hop_length: int = 128
) -> torch.Tensor:
    return torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True).abs()


def stft_loss(
    pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024, hop_length: int = 128
) -> torch.Tensor:
    Y = compute_stft(target, n_fft, hop_length)
    Y_pred = compute_stft(pred, n_fft, hop_length)
    return nn.functional.l1_loss(Y, Y_pred)


####################
# Training logic
####################


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    input_signal: torch.Tensor,
):
    pred = model(input_signal)
    loss = stft_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def training_loop(
    model: nn.Module,
    input_signal: torch.Tensor,
    target: torch.Tensor,
    max_steps: int = 5_000,
    patience: int = 250,
    lr: float = 1e-3,
    logging_interval: int = 100,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    steps_since_best = 0

    for step in range(max_steps):
        loss = training_step(model, optimizer, target, input_signal)

        if step % logging_interval == 0:
            print(f"Step {step}: loss = {loss}")

        if loss < best_loss:
            best_loss = loss
            steps_since_best = 0
        else:
            steps_since_best += 1

        if steps_since_best > patience:
            break

    return best_loss


####################
# Testing
####################


def test(
    model: nn.Module,
    target: torch.Tensor,
    input_signal: torch.Tensor,
    f0_signal: torch.Tensor,
    sample_rate: float = 16000.0,
    n_samples: int = 10_000,
    frame_length: int = 1024,
    device: str = "cuda",
):
    # MSS, MFCC, RMS
    with torch.no_grad():
        pred = model(input_signal)

    target_cpu = target.detach().cpu().numpy()
    pred_cpu = pred.detach().cpu().numpy()

    mss = compute_mss(target_cpu, pred_cpu, sample_rate)
    mfcc = compute_mfcc_distance(target_cpu, pred_cpu, sample_rate)
    rms = compute_rms(target_cpu, pred_cpu, sample_rate)

    # TM, TD, TC
    alpha_min = model.alpha_in.min().item()
    alpha_max = model.alpha_in.max().item()
    beta_min = model.beta_in.min().item()
    beta_max = model.beta_in.max().item()
    f0_min = f0_signal.min().item()
    f0_max = f0_signal.max().item()
    f0_median = f0_signal.median().item()

    tm, td = compute_tm_and_td(
        target_cpu,
        model.newt,
        alpha_min,
        alpha_max,
        beta_min,
        beta_max,
        f0_min,
        f0_max,
        frame_length,
        sample_rate=sample_rate,
        n_samples=n_samples,
        device=device,
    )

    tc = compute_tc(
        target_cpu,
        model.newt,
        alpha_min,
        alpha_max,
        beta_min,
        beta_max,
        f0_median,
        frame_length,
        n_samples,
        sample_rate=sample_rate,
        device=device,
    )

    return dict(
        mss=mss,
        mfcc=mfcc,
        rms=rms,
        tm=tm,
        td=td,
        tc=tc,
    )


####################
# Experiments
####################


def experiment(
    instrument: str = "fl",
    n_segments: int = 10,
    hop_length: int = 128,
    width: int = 128,
    depth: int = 4,
    initial_bandwidth: float = 1.0,
    later_bandwidth: float = 1.0,
    oversample: float = 1.0,
    activation: Literal["sin", "relu", "sinrelu"] = "sin",
    max_steps: int = 5000,
    patience: int = 250,
    lr: float = 1e-3,
    device: str = "cuda",
    results_dir: str = "results",
):
    os.makedirs(results_dir, exist_ok=True)
    results = []

    for idx, (target, f0, input_signal) in enumerate(
        iterate_segments(instrument, n_segments)
    ):
        print(f"Running segment {idx+1}/{n_segments}")

        target = target.to(device)
        input_signal = input_signal.to(device)
        f0 = f0.to(device)

        model = LearnableWaveshaper(
            signal_length_samples=target.shape[-1],
            hop_length=hop_length,
            width=width,
            depth=depth,
            initial_bandwidth=initial_bandwidth,
            later_bandwidth=later_bandwidth,
            oversample=oversample,
            activation=activation,
        ).to(device)

        training_loop(
            model=model,
            input_signal=input_signal,
            target=target,
            max_steps=max_steps,
            patience=patience,
            lr=lr,
        )

        metrics: Dict[str, float] = test(
            model=model,
            target=target,
            input_signal=input_signal,
            f0_signal=f0,
            sample_rate=16000.0,
            n_samples=10000,
            frame_length=1024,
            device=device,
        )

        metrics.update(
            {
                "segment_index": idx,
                "initial_bandwidth": initial_bandwidth,
                "later_bandwidth": later_bandwidth,
                "oversample": oversample,
                "instrument": instrument,
            }
        )

        results.append(metrics)

    # Save results to CSV
    hparam_str = (
        f"{width}{depth}{initial_bandwidth}{later_bandwidth}{oversample}{activation}"
    )
    save_path = os.path.join(results_dir, f"{instrument}_{hparam_str}.csv")
    keys = results[0].keys()
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Experiment results saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(experiment)


# outer loop (over segments)

# input: hyperparams, instrument (and all other config params that need to be passed down.)
# output: csv file with a column for each hyperparam, and a row for each segment
