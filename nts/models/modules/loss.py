import torch
import torch.nn as nn


class WaveformStatisticLoss(nn.Module):
    def forward(self, a, b):
        return (
            torch.abs(a.mean(dim=-1) - b.mean(dim=-1)).mean()
            + torch.abs(torch.std(a, dim=-1) - torch.std(b, dim=-1)).mean()
        )
