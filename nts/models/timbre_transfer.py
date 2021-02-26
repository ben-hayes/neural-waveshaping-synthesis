import auraloss
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from .modules.generators import Wavetable
from .modules.loss import WaveformStatisticLoss
from .modules.shaping import NoiseSaturateFilter
from .modules.tcn import CausalTCN


class TimbreTransfer(pl.LightningModule):
    def __init__(
        self,
        control_embedding_size: int = 16,
        filterbank_channels: int = 14,
        fir_taps: int = 128,
        noise_channels: int = 1,
        tcn_channels: int = 24,
        tcn_kernel_size: int = 8,
        tcn_dilation: int = 8,
        tcn_depth: int = 4,
        sample_rate: float = 16000,
        wavetable_dim: int = 256,
        num_wavetables: int = 1,
        wavetable_init: str = "sine",
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate

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
                    num_wavetables,
                    filterbank_channels,
                    fir_taps,
                    noise_channels,
                    control_embedding_size,
                ),
                NoiseSaturateFilter(
                    filterbank_channels,
                    filterbank_channels,
                    fir_taps,
                    noise_channels,
                    control_embedding_size,
                ),
                NoiseSaturateFilter(
                    filterbank_channels,
                    filterbank_channels,
                    fir_taps,
                    noise_channels,
                    control_embedding_size,
                ),
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

    def render_exciter(self, control):
        f0, amp = control[:, 0].unsqueeze(1), control[:, 1].unsqueeze(1)
        sig = self.wavetable(f0) * amp

        return sig

    def get_embedding(self, control):
        f0, amp = control[:, 0], control[:, 1]
        f0 = f0 / 2000
        control = torch.stack((f0, amp), dim=1)
        return self.embedding(control)

    def forward(self, control):
        x = self.render_exciter(control)
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
            optimizer, factor=0.316228, patience=True
        )
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val/loss"}

    def _run_step(self, batch):
        audio = batch["audio"].float()
        control = batch["control"].float()

        recon = self(control)

        loss = self.stft_loss(recon, audio) + self.stat_loss(recon, audio)
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
            "train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, audio = self._run_step(batch)
        self.log(
            "val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == 0:
            self._log_audio("original", audio[0].detach().cpu().squeeze())
            self._log_audio("recon", recon[0].detach().cpu().squeeze())
        return loss