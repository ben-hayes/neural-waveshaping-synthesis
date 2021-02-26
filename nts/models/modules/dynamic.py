import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta


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
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Dynamic1dConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        conditioning_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        filter_size = in_channels * out_channels * kernel_size

        self.filter_net = TimeDistributedMLP(
            conditioning_size, conditioning_size, filter_size
        )
        self.causal_pad = CausalPad(kernel_size - 1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        filters = self.filter_net(conditioning)
        filters = filters.view(
            conditioning.shape[0],
            conditioning.shape[-1],
            self.out_channels,
            self.in_channels * self.kernel_size,
        )

        x_padded = self.causal_pad(x)
        x_unfolded = F.unfold(x_padded, (1, self.kernel_size))
        x_unfolded = x_unfolded.view(
            x.shape[0], x.shape[-1], self.in_channels * self.kernel_size, -1
        )

        out = torch.matmul(filters, x_unfolded)
        return out.squeeze().transpose(1, 2)
