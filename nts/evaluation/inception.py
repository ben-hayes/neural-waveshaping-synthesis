from typing import Union

import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import Inception3, BasicConv2d


class InceptionSingleChannel(Inception3):
    def __init__(self, **kwargs):
        super().__init__(init_weights=True, **kwargs)

        initialised_kernel = self.Conv2d_1a_3x3.conv.weight[:, 0:1]
        self.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        with torch.no_grad():
            self.Conv2d_1a_3x3.conv.weight.copy_(initialised_kernel)


@gin.configurable
class InceptionInstrumentClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 3,
        aux_logits: bool = True,
        learning_rate: float = 0.001,
        betas: Union[float] = [0, 0.99],
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas

        self.net = InceptionSingleChannel(
            num_classes=num_classes, aux_logits=aux_logits, transform_input=False
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate  # , betas=self.betas
        )
        # return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, eps=1.0)

    def training_step(self, batch, batch_idx):
        inputs = batch["spectrogram"].unsqueeze(1)
        labels = batch["label"]

        # Scale spectrogram to match Inception3 input dimensions
        inputs_scaled = F.interpolate(inputs, (299, 299))
        logits, aux_logits = self(inputs_scaled)
        loss = F.cross_entropy(logits, labels.long()) + F.cross_entropy(
            aux_logits, labels.long()
        )

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
        inputs = batch["spectrogram"].unsqueeze(1)
        labels = batch["label"]

        # Scale spectrogram to match Inception3 input dimensions
        inputs_scaled = F.interpolate(inputs, (299, 299))
        logits = self(inputs_scaled)
        loss = F.cross_entropy(logits, labels.long())
        self.log(
            "val/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        prediction = F.softmax(logits, dim=-1)
        prediction = pl.metrics.utils.to_categorical(prediction)

        true_label = batch["label"]

        accuracy = pl.metrics.functional.accuracy(prediction, true_label)
        self.log(
            "val/accuracy",
            accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        precision = pl.metrics.functional.precision(prediction, true_label)
        self.log(
            "val/precision",
            precision.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        recall = pl.metrics.functional.recall(prediction, true_label)
        self.log(
            "val/recall",
            recall.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }