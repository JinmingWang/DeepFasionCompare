import torch
from torch import nn
from typing import *
from torch.nn import functional as func

class ConvBnReLU(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int = 0, d: int = 1, g: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )