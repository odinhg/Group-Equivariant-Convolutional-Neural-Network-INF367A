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
x1_r, x2_r, x3_r, x4_r = model.get_activations(d2_r(batch))

x = batch[0].detach().numpy()
x_r =d2_r(batch[0]).detach().numpy()
a1, a2, a3, a4 = x1[0].detach().numpy(), x2[0].detach().numpy(), x3[0].detach().numpy(), x4[0].detach().numpy()
a1_r, a2_r, a3_r, a4_r = x1_r[0].detach().numpy(), x2_r[0].detach().numpy(), x3_r[0].detach().numpy(), x4_r[0].detach().numpy()

a1 = np.hstack([np.mean(a1, axis=1)[i,0] for i in range(a1.shape[0])])
a1_r = np.hstack([np.mean(a1_r, axis=1)[i,0] for i in range(a1_r.shape[0])])
plt.imshow(np.vstack([a1, a1_r]))
plt.savefig("activations_layer_1.png")

"""
a2 = np.hstack([a2[i,0,0] for i in range(a2.shape[0])])
a2_r = np.hstack([a2_r[i,0,0] for i in range(a2_r.shape[0])])
plt.imshow(np.vstack([a2, a2_r]))
plt.savefig("activations_layer_2.png")

a3 = np.hstack([a3[i,0,0] for i in range(a3.shape[0])])
a3_r = np.hstack([a3_r[i,0,0] for i in range(a3_r.shape[0])])
plt.imshow(np.vstack([a3, a3_r]))
plt.savefig("activations_layer_3.png")

plt.imshow(np.vstack([a4[0,0], a4_r[0,0]]))
plt.savefig("activations_layer_4_invariant.png")
"""
