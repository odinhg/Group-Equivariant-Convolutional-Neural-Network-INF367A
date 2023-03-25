import torch
import torch.nn as nn
from models import CNNModel, SmoothCNNModel
from torchinfo import summary

from utils import create_dataloaders, Trainer, seed_random_generators
from config import config_cnn, device, val_fraction, test_fraction

train_dl, val_dl, test_dl = create_dataloaders(batch_size=config_cnn["batch_size"], val=val_fraction, test=test_fraction)
model = CNNModel()
#model = SmoothCNNModel()

summary(model)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config_cnn["lr"]) 

trainer = Trainer(model, train_dl, val_dl, test_dl, config_cnn, loss_function, device)

trainer.train(optimizer)
trainer.summarize_training()
trainer.evaluate()

"""
for x in dl:
    images, labels = x[0], x[1]
    out = model(images)
    preds = (out > 0.5).to(torch.uint8).view(-1)
    accuracy = (preds == labels).sum() / preds.shape[0]
    print(accuracy.item())
    print(out)
    break
    # Use stack to add a new dimension, cat to concat in batch dimension
    images = torch.cat([images, d2_r(images), d2_mh(images), d2_mv(images)], dim=0)
    out = model(images)
    print(out)
"""
