import torch
import torch.nn as nn


# ==========================================================
# Conv -> BatchNorm -> ReLU
# ==========================================================

class ConvBNReLU(nn.Module):
    """
    Standard convolution block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=False
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU()

        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Depthwise Convolution
# ==========================================================

class DepthwiseConv(nn.Module):

    def __init__(
        self,
        in_channels,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()

        self.block = ConvBNReLU(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Pointwise Convolution
# ==========================================================

class PointwiseConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Squeeze Expansion Block
# ==========================================================

class SqueezeExpansion(nn.Module):
    """
    1x1 reduction followed by 1x1 expansion.
    """

    def __init__(
        self,
        in_channels,
        reduced_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            PointwiseConv(
                in_channels,
                reduced_channels
            ),

            PointwiseConv(
                reduced_channels,
                out_channels
            )

        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Residual Block
# ==========================================================

class ResidualBlock(nn.Module):

    def __init__(self, block):

        super().__init__()

        self.block = block

    def forward(self, x):

        return x + self.block(x)


# ==========================================================
# Projection Residual Block
# Used when channel dimensions differ
# ==========================================================

class ProjectionResidual(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        block,
        stride=1
    ):
        super().__init__()

        self.block = block

        self.projection = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),

            nn.BatchNorm2d(out_channels)

        )

    def forward(self, x):

        shortcut = self.projection(x)

        return shortcut + self.block(x)


# ==========================================================
# DownSampling Block
# ==========================================================

class DownsampleBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            ConvBNReLU(
                in_channels,
                out_channels,
                stride=2
            )

        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# Global Average Pooling
# ==========================================================

class GlobalAvgPool(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x = self.pool(x)

        return torch.flatten(x, 1)