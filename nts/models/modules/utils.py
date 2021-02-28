import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPad(nn.Module):
    def __init__(self, pad_size: int, lookahead: int = 0):
        super().__init__()
        self.pad_size = pad_size
        self.lookahead = lookahead

    def forward(self, x: torch.Tensor):
        return F.pad(x, (self.pad_size - self.lookahead, self.lookahead))


class Identity(nn.Module):
    def forward(self, x):
        return x

class View(nn.Module):
    def __init__(self, *out_shape):
        super().__init__()
        self.out_shape = out_shape
    
    def forward(self, x):
        return x.view(*self.out_shape)