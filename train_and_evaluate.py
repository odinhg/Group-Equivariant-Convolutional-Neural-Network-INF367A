import torch
import torch.nn as nn
from torchinfo import summary
from os.path import isfile

from models import CNNModel, SmoothCNNModel, GCNNModel
from utils import create_dataloaders, Trainer, seed_random_generators, choose_model, Group, d2_r, d2_mh, d2_mv, d2_e
from config import configs, val_fraction, test_fraction, device

# Let user choose model to use
config = choose_model(configs)

# Data loaders
train_dl, val_dl, test_dl = create_dataloaders(batch_size=config["batch_size"], val=val_fraction, test=test_fraction)

# Initialize chosen model
match config["name"]:
    case "CNN":
        model = CNNModel()
    case "SmoothCNN":
        model = SmoothCNNModel()
    case "GCNN":
        # Functions and Cayley table representing the symmetry group of a rectangle
        functions = [d2_e, d2_r, d2_mh, d2_mv]
        cayley_table = [[0,1,2,3],
                        [1,0,3,2],
                        [2,3,0,1],
                        [3,2,1,0]]
        group = Group(functions, cayley_table)
        model = GCNNModel(group)
    case _:
        raise ValueError("Unexpected model name.")

# Print summary of layers and number of parameters
summary(model)

# Initialize trainer with loss function and optimizer
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-2) 
trainer = Trainer(model, train_dl, val_dl, test_dl, config, loss_function, device)

# Train model if checkpoint is not found
if not isfile(config["checkpoint_file"]):
    trainer.train(optimizer)
    trainer.summarize_training()

# Evaluate model on test data
trainer.evaluate()
