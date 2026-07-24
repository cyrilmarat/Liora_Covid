
import torch
import torch.nn as nn

from .blocks import (
    ConvBNReLU,
    DownsampleBlock,
    GlobalAvgPool
)

from .pepx import (
    ResidualPEPX
)


class CovidNet(nn.Module):

    def __init__(
        self,
        num_classes=4,
        dropout=0.30
    ):
        super().__init__()

        ##################################################
        # STEM
        ##################################################

        self.stem = ConvBNReLU(
            1,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        ##################################################
        # STAGE 1
        ##################################################

        self.stage1 = nn.Sequential(

            ResidualPEPX(
                32,
                16,
                64,
                64
            ),

            ResidualPEPX(
                64,
                16,
                64,
                64
            )

        )

        self.down1 = DownsampleBlock(
            64,
            128
        )

        ##################################################
        # STAGE 2
        ##################################################

        self.stage2 = nn.Sequential(

            ResidualPEPX(
                128,
                32,
                128,
                128
            ),

            ResidualPEPX(
                128,
                32,
                128,
                128
            ),

            ResidualPEPX(
                128,
                32,
                128,
                128
            )

        )

        self.down2 = DownsampleBlock(
            128,
            256
        )

        ##################################################
        # STAGE 3
        ##################################################

        self.stage3 = nn.Sequential(

            ResidualPEPX(
                256,
                64,
                256,
                256
            ),

            ResidualPEPX(
                256,
                64,
                256,
                256
            ),

            ResidualPEPX(
                256,
                64,
                256,
                256
            ),

            ResidualPEPX(
                256,
                64,
                256,
                256
            )

        )

        self.down3 = DownsampleBlock(
            256,
            512
        )

        ##################################################
        # STAGE 4
        ##################################################

        self.stage4 = nn.Sequential(

            ResidualPEPX(
                512,
                128,
                512,
                512
            ),

            ResidualPEPX(
                512,
                128,
                512,
                512
            ),

            ResidualPEPX(
                512,
                128,
                512,
                512
            )

        )

        ##################################################
        # HEAD
        ##################################################

        self.pool = GlobalAvgPool()

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)

        x = self.down1(x)

        x = self.stage2(x)

        x = self.down2(x)

        x = self.stage3(x)

        x = self.down3(x)

        x = self.stage4(x)

        x = self.pool(x)

        x = self.dropout(x)

        x = self.classifier(x)

        return x