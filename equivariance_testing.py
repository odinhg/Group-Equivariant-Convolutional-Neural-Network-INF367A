import torch.nn as nn
from models import StereoZ2ConvG, StereoGBatchNorm2d, StereoGAveragePool, StereoGConvBlock
from utils import create_dataloaders, Group, d2_r, d2_mh, d2_mv, d2_e
from config import val_fraction, test_fraction
from utils import visualize_tensor, normalize_tensor

"""
    This script is only for generating examples illustrating group equivariance (and invariance)
"""

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
        self.g_pool = StereoGAveragePool(reduction="sum")
        #self.g_pool = StereoGAveragePool(group, reduction="sum") # Force invariance

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
x = images[k]
out = model(images)[k]
x, out= normalize_tensor(x), normalize_tensor(out)
visualize_tensor(x, filename="docs/input_image_original.png")
visualize_tensor(out, filename="docs/output_original.png")

for g in functions:
    gx = g(x)
    gout = model(g(images))[k]
    gx, gout = normalize_tensor(gx), normalize_tensor(gout)
    visualize_tensor(gx, filename=f"docs/input_image_{g.__name__}.png")
    visualize_tensor(gout, filename=f"docs/output_{g.__name__}.png")

# TODO: Save figures illustrating both equivariance and invariance (use group argument to pooling layer) of the network 
