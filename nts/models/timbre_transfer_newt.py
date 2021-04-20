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


@gin.configurable
class TimbreTransferNEWT(pl.LightningModule):
    def __init__(
        self,
        embedding_network,
        post_network,
        n_waveshapers: int,
        newts: int,
        control_hop: int,
        sample_rate: float = 16000,
        target_shift: int = 0,
        learning_rate: float = 1e-3,
        patience: int = 25,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.patience = patience
        self.target_shift = target_shift
        self.control_hop = control_hop

        self.sample_rate = sample_rate

        # self.embedding = embedding_network()
        self.embedding = nn.GRU(2, 128, batch_first=True)
        self.embedding_proj = nn.Conv1d(128, 128, 1)

        # self.wavetable = Wavetable()
        self.osc = HarmonicOscillator()
        self.harmonic_mixer = nn.Conv1d(self.osc.n_harmonics, n_waveshapers, 1)

        # self.newt = NEWT()
        self.newts = nn.ModuleList([NEWT() for _ in range(newts)])

        self.h_generator = TimeDistributedMLP(128, 128, 129, 4)
        self.noise_synth = FIRNoiseSynth(control_hop * 2, control_hop)

        self.reverb = Reverb(sample_rate * 2, sample_rate)

        # self.tcn = post_network()

    def render_exciter(self, f0):
        sig = self.osc(f0[:, 0])
        sig = self.harmonic_mixer(sig)
        return sig

    def get_embedding(self, control):
        f0, other = control[:, 0:1], control[:, 1:2]
        control = torch.cat((f0, other), dim=1).transpose(1, 2)
        emb, _ = self.embedding(control)
        emb = self.embedding_proj(emb.transpose(1, 2))
        return emb

    def forward(self, f0, control):
        f0_upsampled = F.upsample(
            f0, f0.shape[-1] * self.control_hop, mode="linear"
        )
        x = self.render_exciter(f0_upsampled)

        control_embedding = self.get_embedding(control)
        embedding_upsampled = F.upsample(
            control_embedding, control.shape[-1] * self.control_hop, mode="linear"
        )

        # x = self.newt(x, control_embedding)
        for newt in self.newts:
            x = newt(x, control_embedding)


        H = self.h_generator(control_embedding)
        noise = self.noise_synth(H)

        # x = self.tcn(torch.cat((x, noise, embedding_upsampled), dim=1))
        x = torch.cat((x, noise), dim=1)
        x = x.sum(1)

        x = self.reverb(x)

        return x

    def configure_optimizers(self):
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _run_step(self, batch):
        audio = batch["audio"].float()
        f0 = batch["f0"].float()
        control = batch["control"].float()

        recon = self(f0, control)
        recon_shifted = recon[:, self.target_shift :]
        audio_shifted = (
            audio[:, : -self.target_shift] if self.target_shift > 0 else audio
        )

        loss = self.stft_loss(recon_shifted, audio_shifted)
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
