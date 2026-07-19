import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveLongRangeConnection(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):

        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Projection des canaux
        self.projection = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

        # Porte de sélection (gating)
        self.selection = nn.Sequential(

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=True
            ),

            nn.Sigmoid()

        )

       	self.fusion = nn.Sequential(

 		   nn.Conv2d(
        	out_channels,
        	out_channels,
        	kernel_size=1,
        	bias=False
    	),

    nn.BatchNorm2d(out_channels),

    nn.ReLU(inplace=True)

)

    def forward(self, source, target):

        source = self.projection(source)

        if source.shape[-2:] != target.shape[-2:]:

            source = F.interpolate(
                source,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # sélection des informations
        gate = self.selection(source + target)

        source = source * (1 + gate)

        out= target + self.alpha * source

        out = self.fusion(out)

        return out