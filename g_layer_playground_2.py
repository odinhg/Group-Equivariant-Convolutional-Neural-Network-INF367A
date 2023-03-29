import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

from utils import create_dataloaders, Trainer, seed_random_generators, visualize_tensor, Group, d2_r, d2_mh, d2_mv, d2_e
from config import config_cnn, config_smoothcnn, val_fraction, test_fraction, device
from models import GNNModel

from os.path import join
figs_path = "figs"
checkpoints_path = "checkpoints"

#device = "cpu"

config = {
                "name" : "G-CNN", 
                "batch_size" : 16,
                "lr" : 5e-5,
                "epochs" : 25,
                "val_per_epoch" : 4,
                "checkpoint_file" : join(checkpoints_path, "gcnn.pth"),
                "loss_plot_file" : join(figs_path, "gcnn_loss_plot.png"),
                "earlystop_limit" : 10

            }
# Functions and Cayley table representing the symmetry group of a rectangle
functions = [d2_e, d2_r, d2_mh, d2_mv]
cayley_table = [[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]
group = Group(functions, cayley_table)
model = GNNModel(group)

train_dl, val_dl, test_dl = create_dataloaders(batch_size=config["batch_size"], val=val_fraction, test=test_fraction)

summary(model)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5) 

trainer = Trainer(model, train_dl, val_dl, test_dl, config, loss_function, device)

trainer.train(optimizer)
trainer.summarize_training()
trainer.evaluate()
