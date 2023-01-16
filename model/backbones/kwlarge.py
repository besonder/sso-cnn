"""
From https://github.com/locuslab/orthogonal-convolutions
"""

from torch import nn

from ..cayley import CayleyConv, CayleyLinear
from ..utils import GroupSort

class KWLarge(nn.Module):
    def __init__(self, conv=CayleyConv, linear=CayleyLinear, w=1, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32 * w, 3), GroupSort(),
            conv(32 * w, 32 * w, 3, stride=2), GroupSort(),
            conv(32 * w, 64 * w, 3), GroupSort(),
            conv(64 * w, 64 * w, 3, stride=2), GroupSort(),
            nn.Flatten(),
            linear(4096 * w, 512 * w), GroupSort(),
            linear(512 * w, 512), GroupSort(),
            linear(512, num_classes)
        )
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out
      
