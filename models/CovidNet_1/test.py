import torch
from torchvision import models
from models.blocks import *

x = torch.randn(4, 3, 224, 224)

conv = ConvBNReLU(3, 32)

y = conv(x)

print(y.shape)

import torch

from models.pepx import PEPX

x = torch.randn(8, 64, 224, 224)

model = PEPX(
    in_channels=64,
    proj_channels=16,
    expand_channels=64,
    out_channels=64
)

y = model(x)

print(y.shape)

from models.pepx import ResidualPEPX

model = ResidualPEPX(
    64,
    16,
    64,
    64
)

y = model(x)

print(y.shape)

import torch

from models.covidnet import CovidNet

model = CovidNet(num_classes=4)

x = torch.randn(8,3,224,224)

y = model(x)

print(y.shape)

