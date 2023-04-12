import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import GCNNModel
from utils import create_dataloaders, Trainer, seed_random_generators, choose_model, Group, d2_r, d2_mh, d2_mv, d2_e
from config import config_gcnn, val_fraction, test_fraction, device

# Use the GCNN model
config = config_gcnn 

train_dl, val_dl, test_dl = create_dataloaders(batch_size=config["batch_size"], val=val_fraction, test=test_fraction)

# Functions and Cayley table representing the symmetry group of a rectangle
functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)
model = GCNNModel(group)
model.eval()

batch = None
for data in train_dl:
    batch, _ = data[0], data[1]
    break

weights = model.g_conv_1[0].layers[0].weight[:,0].detach().numpy()
print(weights.shape)
#[4, 16, 2, 3, 3]

channel = 0
view = 0

single_channel_weights = np.hstack([weights[g,channel,view] for g in range(weights.shape[0])])
print(single_channel_weights)
plt.imshow(single_channel_weights)
plt.savefig("weights.png")
