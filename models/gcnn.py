import torch
import torch.nn as nn

from .groupconv import StereoZ2ConvG, StereoGMaxPool2d, StereoGConvBlock, StereoGBatchNorm2d, StereoGAveragePool
from .stereoconv import StereoConvBlock, StereoMaxPool2d


class GCNNModel(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.lifting_conv = nn.Sequential(
                                # (B, 3, 2, 200, 400)
                                StereoZ2ConvG(group, 3, 32, 3, 1),
                                StereoGBatchNorm2d(group, 32),
                                nn.ReLU(),
                                # (B, n, 32, 2, 200, 400)
                                )

        self.g_conv_layers = nn.Sequential(
                                StereoGConvBlock(group, 32, 32, 3, 1),
                                StereoGMaxPool2d(group, 5, 5),
                                # (B, n, 32, 2, 40, 80)
                                StereoGConvBlock(group, 32, 16, 3, 1),
                                StereoGMaxPool2d(group, 2, 2),
                                # (B, n, 16, 2, 20, 40)
                                StereoGConvBlock(group, 16, 8, 3, 1),
                                StereoGAveragePool(group, reduction="mean"),
                                # G-invariant from here on
                                # (B, 8, 2, 20, 40)
                                StereoConvBlock(8, 8, 3, 1),
                                StereoMaxPool2d(2, 2),
                                # (B, 8, 2, 10, 20)
                                StereoConvBlock(8, 4, 3, 1),
                                StereoMaxPool2d(2, 2),
                                StereoConvBlock(4, 4, 3, 1),
                                # (B, 4, 2, 5, 10)
                                # 4 * 2 * 5 * 10 = 400 features out
                                )
        self.fc = nn.Sequential(
                        nn.Linear(400, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        x = self.lifting_conv(x)
        x = self.g_conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1)
        return x
