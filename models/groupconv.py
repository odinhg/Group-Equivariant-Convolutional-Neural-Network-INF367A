import torch
import torch.nn as nn
import torch.nn.functional as F

from .stereoconv import StereoConv2d, StereoMaxPool2d 
from utils import Group 

"""
    Implementation of group equivariant convolutional layers.
    Disclaimer: I did my best to follow the paper by Cohen and Welling in detail, but there might be some mistakes in my implementation.
"""

class StereoZ2ConvG(StereoConv2d):
    """
        The first layer also called the "lifting convolution". 
        Takes a stereo image (signal on Omega) and returns a signal on the (affine) group G = Aff(H).
        Input of shape (B, C, 2, H, W). Output of shape (B, n, C', 2, H', W') where n is order of the group H.
    """
    def __init__(self, group: Group, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1, stride: int=1):
        super().__init__(in_channels, out_channels, kernel_size, padding, stride, bias=False)
        self.group = group
        self.n = group.order
        # One bias parameter per G feature map following the paper by Cohen & Welling
        self.left_bias = nn.Parameter(torch.zeros(size=(self.n,)), requires_grad=True)
        self.right_bias = nn.Parameter(torch.zeros(size=(self.n,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Create one feature map for each group element. We let the group act on the weights instead of the actual input tensor.
        """
        left, right = self.split_and_pad(x)
        feature_maps = []
        for j, g in enumerate(self.group.inverses):
            out_left = F.conv2d(left, g(self.left_weight), stride=self.stride, padding=0) + self.left_bias[j]
            out_right = F.conv2d(right, g(self.right_weight), stride=self.stride, padding=0) + self.right_bias[j]
            # Stack left and right views
            out = torch.stack([out_left, out_right], dim=2)
            feature_maps.append(out)
        # Stack group dimension
        out = torch.stack(feature_maps, dim=1)
        return out

def merge_dims(x):
    """
        Merge channel and group dimensions.
        Input shape: (B, n, C, 2, H, W)
        Output shape: (B, -1, 2, H, W)
    """
    return x.view(x.shape[0], -1, 2, x.shape[4], x.shape[5])

def unmerge_dims(x, n):
    """
        Unmerge channel and group dimensions. Need group order argument n.
        Input shape: (B, K, 2, H, W)
        Output shape: (B, n, -1, 2, H, W)
    """
    return x.view(x.shape[0], n, -1, 2, x.shape[3], x.shape[4])

class StereoGConv(StereoZ2ConvG):
    """
        The later layers in a group equivariant CNN, also called "group convolutions". 
        Takes as input a signal on the (affine) group G = Aff(H) and return a signal on the same domain. 
        Input of shape (B, n, C, 2, H, W). Output of shape (B, n, C', 2, H', W') where n is order of the group H.

        Note that we permute the group dimension view the G-action. See readme for details.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Merge group and channel dimensions
        x = merge_dims(x) 
        left, right = self.split_and_pad(x)
        left, right = unmerge_dims(left, self.n), unmerge_dims(right, self.n)
        feature_maps = []
        for j, (I, g) in enumerate(zip(self.group.cayley_table, self.group.inverses)):
            # Act on by g in the group dimension
            permuted_left = merge_dims(left[:,I])
            permuted_right = merge_dims(right[:,I])
            out_left = F.conv2d(permuted_left, g(self.left_weight), stride=self.stride, padding=0, groups=self.n) + self.left_bias[j]
            out_right = F.conv2d(permuted_right, g(self.right_weight), stride=self.stride, padding=0, groups=self.n) + self.right_bias[j]
            # Stack left and right views
            out = torch.stack([out_left, out_right], dim=2)
            feature_maps.append(out)
        # Stack group dimension
        out = torch.stack(feature_maps, dim=1)
        return out

class StereoGAveragePool(nn.Module):
    """
        Pool over group dimension. Takes a signal on the group G and returns a signal on the domain Omega.
        If the argument 'group' is given, also pool all transformed versions of input making the output G-invariant.
        Input shape: (B, n, C, 2, H, W)
        Output shape: (B, C, 2, H, W)
    """
    def __init__(self, group: Group=None, reduction: str="mean"):
        super().__init__()
        self.reduction = reduction
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group:
            # Make output G-invariant
            x = torch.cat([g(x) for g in self.group.functions], dim=1)
        if self.reduction == "mean":
            return torch.mean(x, dim=1)
        if self.reduction == "sum":
            return torch.sum(x, dim=1)
        if self.reduction == "max":
            return torch.max(x, dim=1).values
        if self.reduction == "mean":
            return torch.min(x, dim=1).values
        raise ValueError(f"Invalid reduction method \"{self.reduction}\" given in StereoGAveragePool layer.")

class StereoGMaxPool2d(StereoMaxPool2d):
    """
        Max pooling signals on G.
    """
    def __init__(self, group: Group, kernel_size: int=2, stride:int=2):
        super().__init__(kernel_size, stride)
        self.n = group.order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
