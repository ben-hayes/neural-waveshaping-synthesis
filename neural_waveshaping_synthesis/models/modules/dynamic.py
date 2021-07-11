import gin
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta


class TimeDistributedLayerNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        return self.layer_norm(x.transpose(1, 2)).transpose(1, 2)


@gin.configurable
class TimeDistributedMLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, depth: int = 3):
        super().__init__()
        assert depth >= 3, "Depth must be at least 3"
        layers = []
        for i in range(depth):
            layers.append(
                nn.Conv1d(
                    in_size if i == 0 else hidden_size,
                    hidden_size if i < depth - 1 else out_size,
                    1,
                )
            )
            if i < depth - 1:
                layers.append(TimeDistributedLayerNorm(hidden_size))
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
