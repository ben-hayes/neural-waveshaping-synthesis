import auraloss
import pytorch_lightning as pl

from .modules.generators import Wavetable
from .modules.loss import WaveformStatisticLoss
from .modules.shaping import NoiseSaturateFilter
from .modules.tcn import CausalTCN


class TimbreTransfer(pl.LightningModule):
    def __init__(
        self,
        control_embedding_size: int = 32,
        filterbank_channels: int = 16,
        fir_taps: int = 128,
        noise_channels: int = 1,
        tcn_channels: int = 32,
        tcn_kernel_size: int = 8,
        tcn_dilation: int = 8,
        tcn_depth: int = 5,
        sample_rate: float = 16000,
        wavetable_dim: int = 256,
        num_wavetables: int = 1,
        wavetable_init: str = "sine",
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.embedding = CausalTCN(
            2,
            tcn_channels,
            control_embedding_size,
            tcn_kernel_size,
            tcn_dilation,
            tcn_depth,
        )

        self.wavetable = Wavetable(
            wavetable_dim, num_wavetables, sample_rate, init=wavetable_init
        )

        self.noise_saturate_filter = nn.Sequential(
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

    def forward(self, control):
        x = self.render_exciter(control)
        control_embedding = self.control_embedding(control)
        x = self.noise_saturate_filter(x, control_embedding)
        x = self.tcn(x)
        return x.sum(1)

    def configure_optimizers(self):
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss()
        self.stat_loss = WaveformStatisticLoss()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def run_step(self, batch):
        audio = batch["audio"]
        control = batch["control"]

        recon = self(control)

        loss = self.stft_loss(recon, audio) + self.stat_loss(recon, audio)
        return loss

    def train_step(self, batch, batch_idx):
        loss = run_step(batch)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def val_step(self, batch, batch_idx):
        loss = run_step(batch)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss