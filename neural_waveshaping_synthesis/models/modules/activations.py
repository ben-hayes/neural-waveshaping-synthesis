from typing import Callable, Sequence

import torch
import torch.nn as nn


class MultiActivationBank(nn.Module):
    def __init__(self, activations: Sequence[Callable], channels: int):
        super().__init__()
        self.activations = activations
        self.modules = nn.ModuleList(
            [ac for ac in activations if isinstance(ac, nn.Module)]
        )
        self.time_distributed_dense = nn.Conv1d(
            channels * (len(activations) + 1), channels, 1
        )

    def forward(self, x: torch.Tensor):
        activated = [activation(x) for activation in self.activations]
        out = torch.cat((*activated, x), dim=1)
        return self.time_distributed_dense(out)