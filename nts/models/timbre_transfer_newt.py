import math

import auraloss
import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .modules.dynamic import TimeDistributedMLP
from .modules.generators import Wavetable, FIRNoiseSynth, HarmonicOscillator
from .modules.loss import WaveformStatisticLoss
from .modules.shaping import NEWT, Reverb
from .modules.tcn import CausalTCN
from .modules.utils import LearnableUpsampler

gin.external_configurable(nn.GRU, module="torch.nn")
gin.external_configurable(nn.Conv1d, module="torch.nn")


@gin.configurable
class ControlModule(nn.Module):
    def __init__(self, control_size: int, hidden_size: int, embedding_size: int):
        super().__init__()
        self.gru = nn.GRU(control_size, hidden_size, batch_first=True)
        self.proj = nn.Conv1d(hidden_size, embedding_size, 1)
    
    def forward(self, x):
        x, _ = self.gru(x.transpose(1, 2))
        return self.proj(x.transpose(1, 2))


@gin.configurable
class TimbreTransferNEWT(pl.LightningModule):
    def __init__(
        self,
        n_waveshapers: int,
        control_hop: int,
        sample_rate: float = 16000,
        learning_rate: float = 1e-3,
        lr_decay: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.control_hop = control_hop

        self.sample_rate = sample_rate

        self.embedding = ControlModule()

        self.osc = HarmonicOscillator()
        self.harmonic_mixer = nn.Conv1d(self.osc.n_harmonics, n_waveshapers, 1)

        self.newt = NEWT()

        with gin.config_scope("noise_synth"):
            self.h_generator = TimeDistributedMLP()
            self.noise_synth = FIRNoiseSynth()

        self.reverb = Reverb()

    def render_exciter(self, f0):
        sig = self.osc(f0[:, 0])
        sig = self.harmonic_mixer(sig)
        return sig

    def get_embedding(self, control):
        f0, other = control[:, 0:1], control[:, 1:2]
        control = torch.cat((f0, other), dim=1)
        return self.embedding(control)

    def forward(self, f0, control):
        f0_upsampled = F.upsample(f0, f0.shape[-1] * self.control_hop, mode="linear")
        x = self.render_exciter(f0_upsampled)

        control_embedding = self.get_embedding(control)

        x = self.newt(x, control_embedding)

        H = self.h_generator(control_embedding)
        noise = self.noise_synth(H)

        x = torch.cat((x, noise), dim=1)
        x = x.sum(1)

        x = self.reverb(x)

        return x

    def configure_optimizers(self):
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _run_step(self, batch):
        audio = batch["audio"].float()
        f0 = batch["f0"].float()
        control = batch["control"].float()

        recon = self(f0, control)

        loss = self.stft_loss(recon, audio)
        return loss, recon, audio

    def _log_audio(self, name, audio):
        wandb.log(
            {
                "audio/%s"
                % name: wandb.Audio(audio, sample_rate=self.sample_rate, caption=name)
            },
            commit=False,
        )

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._run_step(batch)
        self.log(
            "train/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, audio = self._run_step(batch)
        self.log(
            "val/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if batch_idx == 0:
            self._log_audio("original", audio[0].detach().cpu().squeeze())
            self._log_audio("recon", recon[0].detach().cpu().squeeze())
        return loss
