import torch
import torch.nn as nn
from models import CNNModel, SmoothCNNModel
from torchinfo import summary

from utils import create_dataloaders, Trainer, seed_random_generators
from config import config_cnn, config_smoothcnn, device, val_fraction, test_fraction, device

config = config_smoothcnn 

train_dl, val_dl, test_dl = create_dataloaders(batch_size=config["batch_size"], val=val_fraction, test=test_fraction)
model = CNNModel()
#model = SmoothCNNModel()

summary(model)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5) 

trainer = Trainer(model, train_dl, val_dl, test_dl, config, loss_function, device)

trainer.train(optimizer)
trainer.summarize_training()
trainer.evaluate()
