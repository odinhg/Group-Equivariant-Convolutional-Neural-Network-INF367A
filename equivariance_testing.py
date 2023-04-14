import torch
import torch.nn as nn
from torchinfo import summary
from os.path import isfile

from models import StereoZ2ConvG, StereoGConv, StereoGMaxPool2d, StereoGBatchNorm2d, StereoGAveragePool, StereoGConvBlock
from utils import create_dataloaders, Trainer, seed_random_generators, choose_model, Group, d2_r, d2_mh, d2_mv, d2_e
from config import configs, val_fraction, test_fraction, device
from utils import visualize_tensor, normalize_tensor

functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)

class Model(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.conv1 = nn.Sequential(StereoZ2ConvG(group, 3, 8, 3, 1), StereoGBatchNorm2d(group, 8), nn.ReLU())
        self.conv2 = StereoGConvBlock(group, 8, 8, 3, 1)
        self.conv3 = StereoGConvBlock(group, 8, 3, 5, 2)
        self.g_pool = StereoGAveragePool(group, reduction="sum")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.g_pool(x)
        return x

dl, _, _ = create_dataloaders(batch_size=3, val=val_fraction, test=test_fraction)

model = Model(group)
model.eval()

for batch in dl:
    images, _ = batch
    break

k = 1 # Index for image to use from batch
g = d2_r

x = images[k]
gx = g(x) 
out = model(images)[k]
gout = model(g(images))[k]

x, gx, out, gout = normalize_tensor(x), normalize_tensor(gx), normalize_tensor(out), normalize_tensor(gout)

visualize_tensor(torch.cat([torch.cat([x,out], dim=-2), torch.cat([gx, gout, g(gout)], dim=-2)], dim=-2))

# TODO: Save figures illustrating both equivariance and invariance (use group argument to pooling layer) of the network 
