# translated from colab/NEWT_Timbre_Transfer.ipynb

import subprocess
import shutil
import tempfile
import os
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import gin
import numpy as np
from scipy.io import wavfile
import torch
import cog

from neural_waveshaping_synthesis.data.utils.loudness_extraction import (
    extract_perceptual_loudness,
)
from neural_waveshaping_synthesis.data.utils.f0_extraction import extract_f0_with_crepe
from neural_waveshaping_synthesis.data.utils.preprocess_audio import (
    convert_to_float32_audio,
    make_monophonic,
    resample_audio,
)
from neural_waveshaping_synthesis.models.modules.shaping import FastNEWT
from neural_waveshaping_synthesis.models.neural_waveshaping import NeuralWaveshaping

gin.constant("device", "cuda")
gin.parse_config_file("gin/models/newt.gin")
gin.parse_config_file("gin/data/urmp_4second_crepe.gin")

checkpoints = {"Violin": "vn", "Flute": "fl", "Trumpet": "tpt"}


class Predictor(cog.Predictor):
    def setup(self):
        self.use_gpu = True  # @param {"type": "boolean"}
        self.device = torch.device("cuda")
        self.models = {}
        self.original_newts = {}
        self.data_means = {}
        self.data_stds = {}
        for instrument, checkpoint_key in checkpoints.items():
            print(f"Loading {instrument} model...")
            checkpoint_path = os.path.join("checkpoints/nws", checkpoint_key)
            model = NeuralWaveshaping.load_from_checkpoint(
                os.path.join(checkpoint_path, "last.ckpt")
            ).to(self.device)
            self.models[instrument] = model
            self.original_newts[instrument] = model.newt
            model.eval()
            self.data_means[instrument] = np.load(
                os.path.join(checkpoint_path, "data_mean.npy")
            )
            self.data_stds[instrument] = np.load(
                os.path.join(checkpoint_path, "data_std.npy")
            )

    @cog.input(
        "youtube_url",
        type=str,
        default=None,
        help="Youtube URL. Use either this or audio_file",
    )
    @cog.input(
        "youtube_start_in_seconds",
        type=float,
        default=None,
        help="Youtube clip start (seconds). Use together with youtube_url",
    )
    @cog.input(
        "youtube_length_in_seconds",
        type=float,
        default=None,
        help="Youtube clip length (seconds). Use together with youtube_url",
    )
    @cog.input(
        "audio_file",
        type=Path,
        default=None,
        help="Audio file. Use either this or youtube_url",
    )
    @cog.input(
        "instrument",
        type=str,
        default="Violin",
        options=checkpoints.keys(),
    )
    @cog.input(
        "use_full_crepe_model", type=bool, default=True, help="Use the full CREPE model"
    )
    @cog.input(
        "use_fastnewt",
        type=bool,
        default=False,
        help="Set the parameter below to decide whether to use the original NEWT or the FastNEWT optimisation. For more info see the paper.",
    )
    @cog.input(
        "octave_shift",
        type=int,
        min=-4,
        max=4,
        default=1,
        help="Scale the input octave to match training data. Play with this parameter to fit pitch into an appropriate range.",
    )
    @cog.input(
        "loudness_scale",
        type=float,
        min=0,
        max=2,
        default=0.5,
        help="Scale the loudness of the input to match training data. Play with this parameter to fit loudness into an appropriate range.",
    )
    @cog.input(
        "loudness_floor", type=float, min=0, max=1, default=0, help="Experimental"
    )
    @cog.input(
        "loudness_conf_filter",
        type=float,
        min=0,
        max=0.5,
        default=0,
        help="Experimental",
    )
    @cog.input(
        "pitch_conf_filter", type=float, min=0, max=0.5, default=0, help="Experimental"
    )
    @cog.input(
        "pitch_smoothing",
        type=float,
        min=0,
        max=100,
        default=0,
        help="Very experimental, but can produce some fun wacky sounds",
    )
    @cog.input(
        "loudness_smoothing",
        type=float,
        min=0,
        max=100,
        default=0,
        help="Very experimental, but can produce some fun wacky sounds",
    )
    def predict(
        self,
        youtube_url,
        youtube_start_in_seconds,
        youtube_length_in_seconds,
        audio_file,
        instrument,
        use_full_crepe_model,
        use_fastnewt,
        octave_shift,
        loudness_scale,
        loudness_floor,
        loudness_conf_filter,
        pitch_conf_filter,
        pitch_smoothing,
        loudness_smoothing,
    ):
        # validate inputs
        if youtube_url is not None and audio_file is not None:
            raise ValueError("You must only specify one of youtube_url or audio_file")
        if youtube_url is None and audio_file is None:
            raise ValueError("You must specify one of youtube_url or audio_file")
        if youtube_url is not None and (
            youtube_start_in_seconds is None or youtube_length_in_seconds is None
        ):
            raise ValueError(
                "You must specify youtube_start_in_seconds and youtube_length_in_seconds if you use a youtube_url"
            )
        if youtube_url is None and (
            youtube_start_in_seconds is not None
            or youtube_length_in_seconds is not None
        ):
            raise ValueError(
                "You can only specify youtube_start_in_seconds and youtube_length_in_seconds if you use youtube_url"
            )

        model = self.models[instrument]
        original_newt = self.original_newts[instrument]
        data_mean = self.data_means[instrument]
        data_std = self.data_stds[instrument]

        if youtube_url is not None:
            print("Downloading youtube video...")
            audio = self.youtube_dl(
                youtube_url, youtube_start_in_seconds, youtube_length_in_seconds, model
            )
        else:
            print("Reading input file...")
            audio = self.read_audio_file(audio_file, model)

        print("Extracting F0 with crepe...")
        with torch.no_grad():
            f0, confidence = extract_f0_with_crepe(
                audio,
                sample_rate=model.sample_rate,
                full_model=use_full_crepe_model,
                maximum_frequency=1000,
            )
            loudness = extract_perceptual_loudness(audio)

        print("Computing control signals...")
        f0_t, control = self.compute_control_signals(
            f0,
            confidence,
            loudness,
            pitch_conf_filter,
            loudness_conf_filter,
            octave_shift,
            loudness_floor,
            loudness_scale,
            data_mean,
            data_std,
            pitch_smoothing,
            loudness_smoothing,
        )

        if use_fastnewt:
            model.newt = FastNEWT(original_newt)
        else:
            model.newt = original_newt

        print("Running model...")
        with torch.no_grad():
            start_time = time.time()
            out = model(f0_t.expand(1, 1, -1), control.unsqueeze(0))
            run_time = time.time() - start_time
        rtf = (audio.shape[-1] / model.sample_rate) / run_time
        print(
            "Audio generated in %.2f seconds. That's %.1f times faster than the real time threshold!"
            % (run_time, rtf)
        )

        print("Saving result...")
        synthesized_audio = out.detach().cpu().numpy()
        scaled = np.int16(
            synthesized_audio / np.max(np.abs(synthesized_audio)) * 32767
        ).T
        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "out.wav"
        mp3_path = out_dir / "out.mp3"
        try:
            wavfile.write(str(out_path), rate=model.sample_rate, data=scaled)
            self.ffmpeg_convert(out_path, mp3_path)
        finally:
            out_path.unlink()
        return mp3_path

    def youtube_dl(
        self,
        youtube_url,
        youtube_start_in_seconds,
        youtube_length_in_seconds,
        model,
    ):
        out_dir = Path(tempfile.mkdtemp())
        out_prefix = out_dir / "yt_audio"
        out_path = out_prefix.with_suffix(".wav")
        try:
            cmd = [
                "youtube-dl",
                "--format=bestaudio[ext=m4a]",
                "--extract-audio",
                "--audio-format=wav",
                f"--output={out_prefix}.%(ext)s",
                youtube_url,
            ]
            print("Running: " + " ".join(cmd))
            subprocess.check_output(cmd)
            rate, audio = wavfile.read(str(out_path))
            audio = convert_to_float32_audio(make_monophonic(audio))
            audio = audio[
                int(rate * youtube_start_in_seconds) : int(
                    rate * (youtube_start_in_seconds + youtube_length_in_seconds)
                )
            ]
            audio = resample_audio(audio, rate, model.sample_rate)
            return audio
        finally:
            shutil.rmtree(out_dir)

    def read_audio_file(self, audio_path, model):
        temp_dir = Path(tempfile.mkdtemp())
        wav_path = temp_dir / "input.wav"
        try:
            self.ffmpeg_convert(audio_path, wav_path)
            rate, audio = wavfile.read(str(wav_path))
            audio = convert_to_float32_audio(make_monophonic(audio))
            audio = resample_audio(audio, rate, model.sample_rate)
            return audio
        finally:
            shutil.rmtree(temp_dir)

    def compute_control_signals(
        self,
        f0,
        confidence,
        loudness,
        pitch_conf_filter,
        loudness_conf_filter,
        octave_shift,
        loudness_floor,
        loudness_scale,
        data_mean,
        data_std,
        pitch_smoothing,
        loudness_smoothing,
    ):
        with torch.no_grad():
            f0_filtered = f0 * (confidence > pitch_conf_filter)
            loudness_filtered = loudness * (confidence > loudness_conf_filter)
            f0_shifted = f0_filtered * (2 ** octave_shift)
            loudness_floored = (
                loudness_filtered * (loudness_filtered > loudness_floor)
                - loudness_floor
            )
            loudness_scaled = loudness_floored * loudness_scale
            # loudness = loudness * (confidence > 0.4)

            loud_norm = (loudness_scaled - data_mean[1]) / data_std[1]

            f0_t = torch.tensor(f0_shifted, device=self.device).float()
            loud_norm_t = torch.tensor(loud_norm, device=self.device).float()

            if pitch_smoothing != 0:
                f0_t = torch.nn.functional.conv1d(
                    f0_t.expand(1, 1, -1),
                    torch.ones(1, 1, pitch_smoothing * 2 + 1, device=self.device)
                    / (pitch_smoothing * 2 + 1),
                    padding=pitch_smoothing,
                ).squeeze()
            f0_norm_t = torch.tensor(
                (f0_t.cpu() - data_mean[0]) / data_std[0], device=self.device
            ).float()

            if loudness_smoothing != 0:
                loud_norm_t = torch.nn.functional.conv1d(
                    loud_norm_t.expand(1, 1, -1),
                    torch.ones(1, 1, loudness_smoothing * 2 + 1, device=self.device)
                    / (loudness_smoothing * 2 + 1),
                    padding=loudness_smoothing,
                ).squeeze()
            f0_norm_t = torch.tensor(
                (f0_t.cpu() - data_mean[0]) / data_std[0], device=self.device
            ).float()

            control = torch.stack((f0_norm_t, loud_norm_t), dim=0)

        return f0_t, control

    def ffmpeg_convert(self, input_path, output_path):
        subprocess.check_output(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                str(output_path),
            ],
        )
