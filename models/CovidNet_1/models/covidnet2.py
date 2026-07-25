
import torch
import torch.nn as nn

import models.stem as stem
from models.stem import Stem
from .blocks import DownsampleBlock
from .blocks import GlobalAvgPool
from .pepx import ResidualPEPX
from .slrc import SelectiveLongRangeConnection


class CovidNet(nn.Module):

    def __init__(
        self,
        num_classes=4,
        dropout=0.3
    ):

        super().__init__()
        ##################################################
        # STEM
        ##################################################

        self.stem = stem.Stem(
            in_channels=1,
            out_channels=64
        )

        ##################################################
        # STAGE 1
        ##################################################

        self.stage1 = nn.Sequential(

            ResidualPEPX(
                64,
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

        self.slrc13 = SelectiveLongRangeConnection(
            in_channels=64,
            out_channels=256
            )

        self.slrc24 = SelectiveLongRangeConnection(
            in_channels=128,
            out_channels=512
            )

    def forward(self, x):

        x = self.stem(x)


        # Connexion Stage1
        x = self.stage1(x)
        x1=x

        x = self.down1(x)

        # Connexion Stage2
        x = self.stage2(x)
        x2=x

        x = self.down2(x)


        # Connexion Stage1 → Stage3
        x = self.slrc13(x1, x)


        # Connexion Stage3
        x = self.stage3(x)
        x3=x

        x = self.down3(x)

        # Connexion Stage2 → Stage4
        x = self.slrc24(x2, x)

        x = self.stage4(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x