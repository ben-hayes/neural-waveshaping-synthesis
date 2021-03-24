import math

import auraloss
import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .modules.dynamic import TimeDistributedMLP
from .modules.generators import Wavetable, FIRNoiseSynth
from .modules.loss import WaveformStatisticLoss
from .modules.shaping import NEWT
from .modules.tcn import CausalTCN
from .modules.utils import LearnableUpsampler


@gin.configurable
class TimbreTransferNEWT(pl.LightningModule):
    def __init__(
        self,
        embedding_network,
        post_network,
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

        self.embedding = embedding_network()

        self.wavetable = Wavetable()

        self.newt = NEWT()

        self.h_generator = TimeDistributedMLP(32, 64, 129)
        self.noise_synth = FIRNoiseSynth(control_hop * 2, control_hop)

        self.tcn = post_network()

    def render_exciter(self, control):
        f0, amp = control[:, 0].unsqueeze(1), control[:, 1].unsqueeze(1)
        # sig = self.wavetable(f0) * amp

        phase = torch.cumsum(math.tau * f0 / self.sample_rate, dim=-1)
        sig = torch.sin(phase)

        return sig

    def get_embedding(self, control):
        f0, other = control[:, 0:1], control[:, 1:2]
        f0 = f0 / 2000
        control = torch.cat((f0, other), dim=1)
        return self.embedding(control)

    def forward(self, control):
        control_upsampled = F.upsample(
            control, control.shape[-1] * self.control_hop, mode="linear"
        )
        x = self.render_exciter(control_upsampled)

        control_embedding = self.get_embedding(control)
        embedding_upsampled = F.upsample(
            control_embedding, control.shape[-1] * self.control_hop, mode="linear"
        )

        x = self.newt(x, control_embedding)

        H = self.h_generator(control_embedding)
        noise = self.noise_synth(H)

        x = self.tcn(torch.cat((x, noise, embedding_upsampled), dim=1))

        return x.sum(1)

    def configure_optimizers(self):
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss()
        # self.stat_loss = WaveformStatisticLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.316228, patience=self.patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def _run_step(self, batch):
        audio = batch["audio"].float()
        control = batch["control"].float()

        recon = self(control)
        recon_shifted = recon[:, self.target_shift :]
        audio_shifted = (
            audio[:, : -self.target_shift] if self.target_shift > 0 else audio
        )

        loss = self.stft_loss(recon_shifted, audio_shifted)
        # + self.stat_loss(
        #     recon_shifted, audio_shifted
        # )
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
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
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
        )
        if batch_idx == 0:
            self._log_audio("original", audio[0].detach().cpu().squeeze())
            self._log_audio("recon", recon[0].detach().cpu().squeeze())
        return loss