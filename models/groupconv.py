import torch
import torch.nn as nn
import torch.nn.functional as F

from .stereoconv import StereoConv2d, StereoMaxPool2d, StereoBatchNorm2d
from utils import Group 

"""
    Implementation of group equivariant convolutional layers for stereo images and signals.
    Disclaimer: I did my best to follow the paper by Cohen and Welling in detail, but there might be some mistakes in my implementation.
"""

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
        left = left.view(left.shape[0], self.n, -1, left.shape[2], left.shape[3])
        right = right.view(right.shape[0], self.n, -1, right.shape[2], right.shape[3])
        feature_maps = []
        for j, (I, g) in enumerate(zip(self.group.cayley_table, self.group.inverses)):
            # Act on by g in the group dimension
            permuted_left = left[:,I].view(left.shape[0], -1, left.shape[3], left.shape[4])
            permuted_right = right[:,I].view(right.shape[0], -1, right.shape[3], right.shape[4])
            # Perform group convolution. Note that we share weights over the group dimensions.
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
        Input shape: (B, n, C, 2, H, W)
        Output shape: (B, n, C, 2, H', W')
        For example, if kernel_size = stride = 2, we get H' = H/2 and W' = W/2.
    """
    def __init__(self, group: Group, kernel_size: int=2, stride:int=2):
        super().__init__(kernel_size, stride)
        self.n = group.order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = merge_dims(x)
        out_left = F.max_pool2d(x[:,:,0], self.kernel_size, stride=self.stride) 
        out_right = F.max_pool2d(x[:,:,1], self.kernel_size, stride=self.stride)
        out = torch.stack([out_left, out_right], dim=2)
        out = unmerge_dims(out, self.n)
        return out

class StereoGBatchNorm2d(nn.Module):
    """
        Batch normalization for stereo signals on the group. Note that we use one scale and one bias parameter for each group channel as described in the paper by Cohen and Welling.
    """
    def __init__(self, group: Group, num_features: int):
        super().__init__()
        self.n = group.order
        self.bn_left = nn.GroupNorm(self.n, self.n * num_features, affine=False)
        self.bn_right = nn.GroupNorm(self.n, self.n * num_features, affine=False)
        self.scale_left = nn.Parameter(torch.ones(size=(self.n,)).view(1, -1, 1, 1, 1), requires_grad=True)
        self.scale_right = nn.Parameter(torch.ones(size=(self.n,)).view(1, -1, 1, 1, 1), requires_grad=True)
        self.bias_left = nn.Parameter(torch.zeros(size=(self.n,)).view(1, -1, 1, 1, 1), requires_grad=True)
        self.bias_right = nn.Parameter(torch.zeros(size=(self.n,)).view(1, -1, 1, 1, 1), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = merge_dims(x)
        out_left = self.bn_left(x[:,:,0]).view(x.shape[0], self.n, -1, x.shape[3], x.shape[4])
        out_left = out_left * self.scale_left + self.bias_left
        out_right = self.bn_left(x[:,:,1]).view(x.shape[0], self.n, -1, x.shape[3], x.shape[4])
        out_right = out_right * self.scale_right + self.bias_right
        out = torch.stack([out_left, out_right], dim=3)
        return out 

class StereoGConvBlock(nn.Module):
    """
        Simple G-convolutional block for stereo signals on G: G-Conv -> G-BN -> ReLU. 
    """
    def __init__(self, group: Group, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1, stride: int=1):
        super().__init__()
        self.layers = nn.Sequential(
                            StereoGConv(group, in_channels, out_channels, kernel_size, padding, stride),
                            StereoGBatchNorm2d(group, out_channels),
                            nn.ReLU(),
                        )

    def forward(self, x):
        out = self.layers(x)
        return out
