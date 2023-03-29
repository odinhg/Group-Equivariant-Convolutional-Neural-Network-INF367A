import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNNModel, SmoothCNNModel
from torchinfo import summary
import numpy as np

from utils import create_dataloaders, Trainer, seed_random_generators, visualize_tensor, d2_r, d2_mh, d2_mv, d2_e
from config import config_cnn, config_smoothcnn, val_fraction, test_fraction

"""
    Playground for implementing G convolution. Not to be taken seriously.
"""


class Group():
    """
        Data structure for storing a representation of a group via a list of functions acting on the signals, and a Cayley table encoding the group structure. Note that the order of the list of functions must match the order in the Cayley table. Furthermore, the identity function must be the first function in the list.
    """
    def __init__(self, functions, cayley_table):
        self._functions = np.array(functions)
        self._cayley_table = np.array(cayley_table, dtype=np.int64)
        self._order = len(functions)
        # Find and store an ordered list of inverses of the functions
        self._inverses = self._functions[np.where(self.cayley_table == 0)[1]]

    @property
    def functions(self):
        return self._functions

    @property
    def cayley_table(self):
        return self._cayley_table

    @property
    def order(self):
        return self._order

    @property
    def inverses(self):
        return self._inverses

class Z2ConvG(nn.Module):
    """
        Lifting convolution. Takes an image (signal on Omega) and returns a signal on the (affine) group G = Aff(H).
    """
    def __init__(self, group, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.zeros(size=(out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, pad=(self.padding,) * 4, mode="circular")
        x = torch.stack([F.conv2d(x, weight=g(self.weight), padding=0, stride=self.stride) for g in self.group.inverses], dim=1)
        return x

def merge_dims(x):
    """
        Merge channel and group dimensions.
    """
    return x.view(x.shape[0], -1, x.shape[3], x.shape[4])

def unmerge_dims(x, n):
    """
        Unmerge channel and group dimensions. Need group order argument n.
    """
    return x.view(x.shape[0], n, -1, x.shape[2], x.shape[3])

class GConv(nn.Module):
    """
        G-convolution layer.
    """
    def __init__(self, group, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.zeros(size=(out_channels, self.group.order * in_channels, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # Merge dimensions needed because of padding
        x = merge_dims(x) 
        if self.padding:
            x = F.pad(x, pad=(self.padding,) * 4, mode="circular")
        x = unmerge_dims(x, self.group.order)
        # Convolve over all group elements (the channels are "multiplied" with g as well, i.e., we are permuting the group dimension)
        feature_maps = [F.conv2d(merge_dims(x[:,I,:,:]), g(self.weight), padding=0, stride=self.stride) for I, g in zip(self.group.cayley_table, self.group.inverses)]
        x = torch.stack(feature_maps, dim=1)
        return x

class GPool(nn.Module):
    """
        Pool over group dimension. Takes a signal on the group G and returns a signal on Omega, i.e., an image.
        If the argument 'group' is given, pool in addition over all transformed versions of input making the output G-invariant.
    """
    def __init__(self, group=None, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "max", "min"], f"Invalid reduction type \"{reduction}\" specified."
        self.reduction = reduction
        self.group = group

    def forward(self, x):
        if self.group:
            x = torch.cat([g(x) for g in self.group.functions], dim=1)
        match self.reduction:
            case "mean":
                return torch.mean(x, dim=1)
            case "sum":
                return torch.sum(x, dim=1)
            case "max":
                return torch.max(x, dim=1).values
            case "min":
                return torch.min(x, dim=1).values

def G_max_pool2d(x, group, kernel_size=2, stride=2):
    """
        Wrapper for max pooling layer to support an extra group dimension.
    """
    y = merge_dims(x)
    y = F.max_pool2d(y, kernel_size, stride)
    y = unmerge_dims(y, group.order)
    return y

class GNNModel(nn.Module):
    def __init__(self, group):
        self.group = group
        self.lifting_conv = Z2ConvG(group, in_channels=3, out_channels=32, kernel_size=3, padding=1)
        second_layer = GConv(group, in_channels=1, out_channels=32, kernel_size=3, padding=1)
        third_layer = GConv(group, in_channels=32, out_channels=3, kernel_size=3, padding=1)
        pooling_layer = GPool(reduction="mean")

device = "cpu"
train_dl, val_dl, test_dl = create_dataloaders(batch_size=2, val=val_fraction, test=test_fraction, image_size=(200,800))
for x in train_dl:
    images, labels = x[0], x[1]
    break

# Functions and Cayley table representing the symmetry group of a rectangle
functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)

first_layer = Z2ConvG(group, in_channels=3, out_channels=1, kernel_size=5, padding=2)
second_layer = GConv(group, in_channels=1, out_channels=32, kernel_size=3, padding=1)
third_layer = GConv(group, in_channels=32, out_channels=3, kernel_size=3, padding=1)
pooling_layer = GPool(reduction="mean")

def ff(images, r=False):
    print(f"input size: {images.shape}")
    out0 = first_layer(images)
    print(f"lifted size: {out0.shape}")
    out1 = second_layer(out0)
    print(f"after G conv size: {out1.shape}")
    out2 = third_layer(out1)
    print(f"after second G conv size: {out2.shape}")
    out3 = pooling_layer(out2)
    print(f"Pooled size: {out3.shape}")
    k = 1
    return torch.cat([images[k], out3[k]], dim=-2)
    #return torch.cat([images[k], out3[k]], dim=-2)

normal = ff(images)
rotated = ff(d2_r(images), r=True)
visualize_tensor(torch.cat([normal, rotated], dim=-1))
