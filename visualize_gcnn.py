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

x1, x2, x3, x4 = model.get_activations(batch)

x = batch[0].detach().numpy()
a1, a2, a3, a4 = x1[0].detach().numpy(), x2[0].detach().numpy(), x3[0].detach().numpy(), x4[0].detach().numpy()

a1 = np.hstack([a1[i,0,0] for i in range(a1.shape[0])])
plt.imshow(a1)
plt.savefig("activations_layer_1.png")

a2 = np.hstack([a2[i,0,0] for i in range(a2.shape[0])])
plt.imshow(a2)
plt.savefig("activations_layer_2.png")

a3 = np.hstack([a3[i,0,0] for i in range(a3.shape[0])])
plt.imshow(a3)
plt.savefig("activations_layer_3.png")
