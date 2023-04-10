import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Some wrapper classes for working with stereo images. We operate on each view (left and right) separately with two sets of weights (and biases).
"""

class StereoConv2d(nn.Module):
    """
        Convolution for stereo images (tensors) of shape (B, C, 2, H, W). With support for circular padding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1, 
                 stride: int=1, bias: bool=True):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.zeros(size=(out_channels, in_channels, 2, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(2, out_channels,)), requires_grad=True)
        else:
            self.bias = None
    
    @staticmethod
    def split_and_pad(x: torch.Tensor, padding: int=1) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Split into left and right view, add circular padding if needed, and return both views.
            Input shape: (B, C, 2, H, W)
            Output shape: ((B, C, H', W'), (B, C, H', W')) where H' = H + 2 * padding and W' = W + 2 * padding.
        """
        left, right = x[:,:,0], x[:,:,1]
        if padding:
            left = F.pad(left, pad=(padding,) * 4, mode="circular")
            right = F.pad(right, pad=(padding,) * 4, mode="circular")
        return (left, right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.split_and_pad(x, self.padding)
        out_left = F.conv2d(left, self.weight[:,:,0], bias=self.bias[0], stride=self.stride, padding=0)
        out_right = F.conv2d(right, self.weight[:,:,1], bias=self.bias[1], stride=self.stride, padding=0)
        out = torch.stack([out_left, out_right], dim=2)
        return out

class StereoBatchNorm2d(nn.Module):
    """
        Batch normalization for stereo images.
    """
    def __init__(self, num_features: int, affine:bool=True):
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
        Simple convolutional block for stereo images: Conv -> BN -> ReLU. 
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1, bias: bool=True, stride: int=1):
        super().__init__()
        self.layers = nn.Sequential(
                            StereoConv2d(in_channels, out_channels, kernel_size, padding, stride, bias=bias),
                            StereoBatchNorm2d(out_channels),
                            nn.ReLU(),
                        )

    def forward(self, x):
        out = self.layers(x)
        return out
