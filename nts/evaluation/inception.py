import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models.inception import Inception3, BasicConv2d


class InceptionSingleChannel(Inception3):
    def __init__(self, **kwargs):
        super().__init__(init_weights=True, **kwargs)

        initialised_kernel = self.Conv2d_1a_3x3.conv.weight[:, 0:1]
        self.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        with torch.no_grad():
            self.Conv2d_1a_3x3.conv.weight.copy_(initialised_kernel)


class InceptionInstrumentClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 3, aux_logits: bool = True):
        super().__init__()

        self.net = InceptionSingleChannel(
            num_classes=num_classes, aux_logits=aux_logits, transform_input=False
        )

    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=[0, 0.99])
    
    def training_step(self, batch, batch_idx):
        pass