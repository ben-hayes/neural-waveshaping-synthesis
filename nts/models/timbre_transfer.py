import auraloss
import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from .modules.generators import Wavetable
from .modules.loss import WaveformStatisticLoss
from .modules.shaping import NoiseSaturateFilter
from .modules.tcn import CausalTCN


@gin.configurable
class TimbreTransfer(pl.LightningModule):
    def __init__(
        self,
        control_embedding_size: int = 16,
        filterbank_channels: int = 14,
        fir_taps: int = 128,
        filterbank_depth: int = 3,
        tcn_channels: int = 24,
        tcn_kernel_size: int = 8,
        tcn_dilation: int = 8,
        tcn_depth: int = 4,
        sample_rate: float = 16000,
        wavetable_dim: int = 256,
        num_wavetables: int = 1,
        wavetable_init: str = "sine",
        target_shift: int = 0,
        learning_rate: float = 1e-3,
        patience: int = 25,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.patience = patience
        self.target_shift = target_shift

        self.sample_rate = sample_rate

        self.filterbank_channels = filterbank_channels

        self.embedding = CausalTCN(
            2,
            control_embedding_size,
            control_embedding_size,
            tcn_kernel_size,
            tcn_dilation,
            tcn_depth,
        )

        self.wavetable = Wavetable(
            wavetable_dim, num_wavetables, sample_rate, initialisation=wavetable_init
        )

        self.noise_saturate_filter = nn.ModuleList(
            [
                NoiseSaturateFilter(
                    filterbank_channels,
                    filterbank_channels,
                    fir_taps,
                    control_embedding_size,
                )
                for _ in range(filterbank_depth)
            ]
        )

        self.tcn = CausalTCN(
            filterbank_channels,
            tcn_channels,
            1,
            tcn_kernel_size,
            tcn_dilation,
            tcn_depth,
        )
        # self.tcn = nn.Conv1d(filterbank_channels, 1, 1)

    def render_exciter(self, control):
        f0, amp = control[:, 0].unsqueeze(1), control[:, 1].unsqueeze(1)
        amp = amp * 80 - 80
        amp = 1000 * 10 ** (amp / 20)
        sig = self.wavetable(f0) * amp

        return sig

    def get_embedding(self, control):
        f0, amp = control[:, 0], control[:, 1]
        f0 = f0 / 2000
        control = torch.stack((f0, amp), dim=1)
        return self.embedding(control)

    def forward(self, control):
        x = self.render_exciter(control)
        x = x.expand(-1, self.filterbank_channels, -1)
        control_embedding = self.get_embedding(control)
        for nsf in self.noise_saturate_filter:
            x = nsf(x, control_embedding)
        x = self.tcn(x)
        return x.sum(1)

    def configure_optimizers(self):
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss()
        self.stat_loss = WaveformStatisticLoss()
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

        loss = self.stft_loss(recon_shifted, audio_shifted) + self.stat_loss(
            recon_shifted, audio_shifted
        )
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