import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Some wrapper classes for working with stereo images.
"""

class StereoConv2d(nn.Module):
    """
        Convolution for stereo images (tensors) of shape (B, C, 2, H, W). With support for circular padding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1 , 
                 stride: int=1, groups: int=1, bias: bool=False):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(size=(out_channels, in_channels, 2, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,)), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x[:,:,0], x[:,:,1]
        if self.padding:
            left = F.pad(left, pad=(self.padding,) * 4, mode="circular")
            right = F.pad(right, pad=(self.padding,) * 4, mode="circular")
        out_left = F.conv2d(left, self.weight[:,:,0], bias=self.bias, stride=self.stride, padding=0, groups=self.groups)
        out_right = F.conv2d(right, self.weight[:,:,1], bias=self.bias, stride=self.stride, padding=0, groups=self.groups)
        out = torch.stack([out_left, out_right], dim=2)
        return out

class StereoBatchNorm2d(nn.Module):
    """
        Batch normalization for stereo images.
    """
    def __init__(self, num_features: int, affine:bool=False):
        super().__init__()
        self.bn_left = nn.BatchNorm2d(num_features, affine=affine)
        self.bn_right = nn.BatchNorm2d(num_features, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_left = self.bn_left(x[:,:,0])
        out_right = self.bn_right(x[:,:,1])
        out = torch.stack([out_left, out_right], dim=2)
        return out

class StereoMaxPool2d(nn.Module):
    """
        Max pooling for stereo images.
    """
    def __init__(self, kernel_size: int=2, stride:int=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_left = F.max_pool2d(x[:,:,0], self.kernel_size, stride=self.stride)
        out_right = F.max_pool2d(x[:,:,1], self.kernel_size, stride=self.stride)
        out = torch.stack([out_left, out_right], dim=2)
        return out

class StereoConvBlock(nn.Module):
    """
        Simple convolutional block for stereo images: CNN -> BN -> ReLU. 
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1, bias: bool=False):
        super().__init__()
        self.layers = nn.Sequential(
                            StereoConv2d(in_channels, out_channels, kernel_size, padding, bias=bias),
                            StereoBatchNorm2d(out_channels),
                            nn.ReLU(),
                        )

    def forward(self, x):
        out = self.layers(x)
        return out
