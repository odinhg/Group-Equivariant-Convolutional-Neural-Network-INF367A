import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNNModel, SmoothCNNModel
from torchinfo import summary

from utils import create_dataloaders, Trainer, seed_random_generators, visualize_tensor, d2_r, d2_mh, d2_mv, d2_e
from config import config_cnn, config_smoothcnn, val_fraction, test_fraction

class D2():
    def __init__(self):
        pass

    def permute_channels(self, i):
        pass

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
        print(self.weight.shape)

    def forward(self, x):
        if self.padding:
            # Padding with cyclic boundary
            x = F.pad(x, pad=(self.padding,)*4, mode="circular")
        x = torch.stack([F.conv2d(x, weight=g(self.weight), padding=0, stride=self.stride) for g in self.group], dim=1)
        return x

class GConv(nn.Module):
    """
        G-convultion.
    """
    def __init__(self, group, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.n = len(self.group)

        self.weight = nn.Parameter(torch.zeros(size=(out_channels, self.n * in_channels, kernel_size, kernel_size)), requires_grad=True)
        print(self.weight.shape)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        if self.padding:
            # Padding with cyclic boundary
            x = F.pad(x, pad=(self.padding,)*4, mode="circular")
        x = x.view(x.shape[0], 4, -1, x.shape[2], x.shape[3])
        #x = torch.stack([F.conv2d(x, weight=g(self.weight), padding=0, stride=self.stride) for g in self.group], dim=1)

        # Ugly way of doing permutation / channel cycling...
        x = torch.stack([
        F.conv2d(x.view(x.shape[0], -1, x.shape[3], x.shape[4]) , weight=self.weight, padding=0, stride=self.stride),
        F.conv2d(x[:,[1,0,3,2],:,:].view(x.shape[0], -1, x.shape[3], x.shape[4]), weight=d2_r(self.weight), padding=0, stride=self.stride),
        F.conv2d(x[:,[2,3,0,1],:,:].view(x.shape[0], -1, x.shape[3], x.shape[4]), weight=d2_mh(self.weight), padding=0, stride=self.stride),
        F.conv2d(x[:,[3,2,1,0],:,:].view(x.shape[0], -1, x.shape[3], x.shape[4]), weight=d2_mv(self.weight), padding=0, stride=self.stride)], dim=1)
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
            x = torch.cat([g(x) for g in self.group], dim=1)
        match self.reduction:
            case "mean":
                return torch.mean(x, dim=1)
            case "sum":
                return torch.sum(x, dim=1)
            case "max":
                return torch.max(x, dim=1).values
            case "min":
                return torch.min(x, dim=1).values

device = "cpu"

train_dl, val_dl, test_dl = create_dataloaders(batch_size=8, val=val_fraction, test=test_fraction)

for x in train_dl:
    images, labels = x[0], x[1]
    break

group = [d2_e, d2_r, d2_mh, d2_mv]
first_layer = Z2ConvG(group, in_channels=3, out_channels=32, kernel_size=3, padding=1)
second_layer = GConv(group, in_channels=32, out_channels=16, kernel_size=3, padding=1)
third_layer = GConv(group, in_channels=16, out_channels=3, kernel_size=3, padding=1)
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
    if r:
        out3 = d2_r(out3)
    return torch.cat([images[k], out3[k]], dim=-2)

normal = ff(images)
rotated = ff(d2_r(images), r=True)
visualize_tensor(torch.cat([normal, rotated], dim=-1))

# Lifting layer works: it is equivariant
# Pooling layer also
# GConv is NOT equivariant
# Need Channel Shifting? Permute dim 1 with inverses? Maybe have a group class with cayley table?
# Update: GConv is now equivariant 


"""
# Test lifting convolution and pooling layer
print(images.shape)
group = [d2_e, d2_r, d2_mv, d2_mh]
test_symmetry = d2_mv
first_layer = Z2ConvG(group, in_channels=3, out_channels=3, kernel_size=3, padding=1)
feature_map = first_layer(images)
feature_map_rotated = first_layer(test_symmetry(images))
print(feature_map.shape)
print(feature_map_rotated.shape)
pooling_layer = GPool(reduction="mean")
pooled = pooling_layer(feature_map)
pooled_rotated = pooling_layer(feature_map_rotated)
print(pooled.shape)
print(pooled_rotated.shape)
k = 2
visualize_tensor(torch.cat([images[k], pooled[k], pooled_rotated[k], test_symmetry(pooled_rotated[k])], dim=-2))
"""

"""
#visualize_tensor(images[1])
d2features  = [images[1]]

transforms = [d2_e, d2_r, d2_mv, d2_mh]

# Pad circular
images = nn.functional.pad(images, pad=(1,1,1,1), mode="circular")

for g in transforms:
    weight = torch.Tensor([
                [[1,2,1], [0,0,0], [-1,-2,1]],
                [[1,0,-1], [2,0,-2], [1,0,-1]],
                [[0,0,0], [0,0,0], [0,0,0]],
                ]) * (1/64)
    print(weight)
    weight = weight.view(3,1,3,3)
    weight = g(weight)
    print(weight)
    #weight = weight.repeat(3, 1, 1, 1)
    feature_maps = nn.functional.conv2d(images, weight, padding=0, groups=3, stride=1)
    d2features.append(feature_maps[1])

d2features = torch.cat(d2features, dim=1) 
visualize_tensor(d2features)
"""
