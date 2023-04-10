import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

from utils import create_dataloaders, Trainer, seed_random_generators, visualize_tensor, Group, d2_r, d2_mh, d2_mv, d2_e
from config import config_cnn, config_smoothcnn, val_fraction, test_fraction, device
from models import GCNNModel

from os.path import join
figs_path = "figs"
checkpoints_path = "checkpoints"

config = {
                "name" : "G-CNN", 
                "batch_size" : 16,
                "lr" : 1e-4,
                "epochs" : 50,
                "val_per_epoch" : 4,
                "checkpoint_file" : join(checkpoints_path, "gcnn.pth"),
                "loss_plot_file" : join(figs_path, "gcnn_loss_plot.png"),
                "earlystop_limit" : 20

            }
# Functions and Cayley table representing the symmetry group of a rectangle
functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)
model = GCNNModel(group)

train_dl, val_dl, test_dl = create_dataloaders(batch_size=config["batch_size"], val=val_fraction, test=test_fraction)

"""
# Compute mean and std of training dataset for both views
from tqdm import tqdm

left_sum = torch.zeros(3).to(device)
right_sum = torch.zeros(3).to(device) 
left_sum_sq = torch.zeros(3).to(device)
right_sum_sq = torch.zeros(3).to(device)

for data in tqdm(train_dl):
    images, labels = data[0].to(device), data[1].to(device)
    left_sum += images[:,:,0].sum(-1).sum(-1).sum(0)
    right_sum += images[:,:,1].sum(-1).sum(-1).sum(0)
    left_sum_sq += (images[:,:,0]**2).sum(-1).sum(-1).sum(0)
    right_sum_sq += (images[:,:,1]**2).sum(-1).sum(-1).sum(0)

count = len(train_dl) * config["batch_size"] * 200 * 400
left_mean = left_sum / count
right_mean = right_sum / count

left_var = left_sum_sq / count - left_mean**2
right_var = right_sum_sq / count - right_mean**2

left_std = torch.sqrt(left_var)
right_std = torch.sqrt(right_var)

print(left_mean)
print(left_std)

print(right_mean)
print(right_std)

"""

"""
#TEST G INVARIANCE
model.eval()
with torch.no_grad():
    for data in train_dl:
        images, labels = data[0], data[1]
        print(model(images))
        print(model(d2_r(images)))
        print(model(d2_mh(images)))
        print(model(d2_mv(images)))
        break
exit()
"""

summary(model)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-2) 

trainer = Trainer(model, train_dl, val_dl, test_dl, config, loss_function, device)

trainer.train(optimizer)
trainer.summarize_training()
trainer.evaluate()
