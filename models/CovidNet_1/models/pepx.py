import torch
import torch.nn as nn

from .blocks import (
    PointwiseConv,
    DepthwiseConv,
)


class PEPX(nn.Module):
    """
    Projection → Expansion → Depthwise →
    Projection → Extension
    """

    def __init__(
        self,
        in_channels,
        proj_channels,
        expand_channels,
        out_channels
    ):
        super().__init__()

        self.pepx = nn.Sequential(

            # Projection
            PointwiseConv(
                in_channels,
                proj_channels
            ),

            # Expansion
            PointwiseConv(
                proj_channels,
                expand_channels
            ),

            # Depthwise convolution
            DepthwiseConv(
                expand_channels
            ),

            # Projection
            PointwiseConv(
                expand_channels,
                proj_channels
            ),

            # Extension
            PointwiseConv(
                proj_channels,
                out_channels
            )

        )

    def forward(self, x):

        return self.pepx(x)

class ResidualPEPX(nn.Module):
    """
    Residual PEPX block
    """

    def __init__(
        self,
        in_channels,
        proj_channels,
        expand_channels,
        out_channels
    ):
        super().__init__()

        self.pepx = PEPX(
            in_channels,
            proj_channels,
            expand_channels,
            out_channels
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.pepx(x)

        out = out + identity

        out = self.relu(out)

        return out