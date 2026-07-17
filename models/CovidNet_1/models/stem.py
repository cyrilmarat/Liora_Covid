import torch
import torch.nn as nn

from .blocks import ConvBNReLU


class Stem(nn.Module):
    """
    CovidNet Stem

    224x224
        ↓
    Conv3x3 s=2
        ↓
    112x112
        ↓
    Conv3x3
        ↓
    112x112
        ↓
    MaxPool
        ↓
    56x56
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=64):

        super().__init__()

        self.stem = nn.Sequential(

            ConvBNReLU(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            ConvBNReLU(
                32,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )

        )

    def forward(self, x):

        return self.stem(x)