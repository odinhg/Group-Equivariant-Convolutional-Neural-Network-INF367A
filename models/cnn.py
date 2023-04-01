import torch
import torch.nn as nn
from .stereoconv import StereoConvBlock, StereoMaxPool2d

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
                                # (B, 3, 2, 200, 400)
                                StereoConvBlock(3, 32, 3, 1),
                                StereoMaxPool2d(5, 5),
                                # (B, 32, 2, 40, 80)
                                StereoConvBlock(32, 32, 3, 1),
                                StereoConvBlock(32, 16, 3, 1),
                                StereoMaxPool2d(2, 2),
                                # (B, 16, 2, 20, 40)
                                StereoConvBlock(16, 16, 3, 1),
                                StereoConvBlock(16, 8, 3, 1),
                                StereoMaxPool2d(2, 2),
                                # (B, 8, 2, 10, 20)
                                StereoConvBlock(8, 8, 3, 1),
                                StereoConvBlock(8, 4, 3, 1),
                                StereoMaxPool2d(2, 2),
                                # (B, 4, 2, 5, 10)
                                # 4 * 2 * 5 * 10 = 400 features out
                            )

        self.fc = nn.Sequential(
                        nn.Linear(400, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1)
        return x
