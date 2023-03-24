from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from .earlystopper import EarlyStopper

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.loss_plot_file = config["loss_plot_file"]
        self.checkpoint_file = config["checkpoint_file"]
        self.loss_function = loss_function
        self.device = device
        self.val_steps = len(train_dataloader) // config["val_per_epoch"]
        self.earlystopper = EarlyStopper(limit=config["earlystop_limit"])
        self.train_history = {"train_loss" : [], "val_loss" : [], "train_acc" : [], "val_acc" : []}
        self.epoch_times = []
        self.checkpoint_saved_epoch = 0

    def get_data_and_targets(self, data):
        return data[0].to(self.device), data[1].to(self.device) 

    def prob_to_pred(self, probs, threshold=0.5):
        preds = (probs > threshold).to(torch.uint8).view(-1)
        return preds

    def compute_accuracy(self, preds, labels):
        accuracy = (preds == labels).sum() / preds.shape[0]
        return accuracy.item()

    def train_step(self, data):
        images, labels = self.get_data_and_targets(data)
        self.optimizer.zero_grad()
        probs = self.model(images)
        preds = self.prob_to_pred(probs)
        loss = self.loss_function(probs, labels)
        accuracy = self.compute_accuracy(preds, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy

    def train(self, optimizer, scheduler=None):
        print(f"Training model...")
        self.optimizer = optimizer
        self.scheduler = scheduler
        for epoch in range(self.epochs):
            time_start = time.time()
            train_losses = []
            train_accuracies = []
            for i, data in enumerate((pbar := tqdm(self.train_dataloader))):
                loss, accuracy = self.train_step(data)
                train_losses.append(loss)
                train_accuracies.append(accuracy)

                if i % self.val_steps == self.val_steps - 1:
                    mean_train_loss = np.mean(train_losses)
                    mean_train_acc = np.mean(train_accuracies)
                    train_losses = []
                    train_accuracies = []
                    mean_val_loss, mean_val_acc = self.validate()

                    if mean_val_acc > np.max(self.train_history["val_acc"], initial=0.0):
                        torch.save(self.model.state_dict(), self.checkpoint_file)
                        self.checkpoint_saved_epoch = epoch

                    self.train_history["train_loss"].append(mean_train_loss)
                    self.train_history["train_acc"].append(mean_train_acc)
                    self.train_history["val_loss"].append(mean_val_loss)
                    self.train_history["val_acc"].append(mean_val_acc)

                    if self.earlystopper(mean_val_acc):
                        print(f"Early stopped at epoch {epoch}!")
                        return

                    pbar_str = f"Epoch {epoch:02}/{self.epochs:02} | "
                    pbar_str += f"Train [{mean_train_loss:.3f}|{mean_train_acc:.2f}] | "
                    pbar_str += f"Val [{mean_val_loss:.3f}|{mean_val_acc:.2f}] | "
                    pbar_str += f"ES: {self.earlystopper.counter:02}/{self.earlystopper.limit:02} | "
                    if self.scheduler:
                        pbar_str += f"LR: {self.scheduler.get_last_lr()[0]} |"
                    pbar.set_description(pbar_str)
            if self.scheduler:
                self.scheduler.step()
            self.epoch_times.append(time.time() - time_start)

    def val_step(self, data):
        images, labels = self.get_data_and_targets(data)
        probs = self.model(images)
        preds = self.prob_to_pred(probs)
        loss = self.loss_function(probs, labels)
        accuracy = self.compute_accuracy(preds, labels)
        return loss.item(), accuracy

    def validate(self):
        self.model.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for data in self.val_dataloader:
                val_loss, val_acc = self.val_step(data)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
        mean_val_loss = np.mean(val_losses)
        mean_val_acc = np.mean(val_accuracies)
        self.model.train()
        return mean_val_loss, mean_val_acc

    def save_loss_plot(self):
        print(f"Saving loss plot to {self.loss_plot_file}...")
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        ax[0].plot(self.train_history["train_loss"], label="Training")
        ax[0].plot(self.train_history["val_loss"], label="Validation")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("Mean loss")
        ax[0].legend(loc="upper right")

        ax[1].plot(self.train_history["train_acc"], label="Training")
        ax[1].plot(self.train_history["val_acc"], label="Validation")
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("Mean accuracy")
        ax[1].legend(loc="upper right")

        fig.suptitle("Loss and accuracy")
        fig.tight_layout()
        plt.savefig(self.loss_plot_file, dpi=100)

    def summarize_training(self):
        self.save_loss_plot()
        print(f"Last checkpoint saved at epoch {self.checkpoint_saved_epoch}.")
        print(f"Total training time: {np.sum(self.epoch_times):.2f}s.")
        print(f"Mean epoch time: {np.mean(self.epoch_times):.2f}s.")

    def evaluate(self):
        print("Loading checkpoint...")
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        self.model.eval()
        print("Evaluating model on test data...")
        with torch.no_grad():
            test_accuracies = []
            for data in tqdm(self.test_dataloader): 
                images, labels = self.get_data_and_targets(data) 
                probs = self.model(images)
                preds = self.prob_to_pred(probs)
                accuracy = self.compute_accuracy(preds, labels)
                test_accuracies.append(accuracy)

        mean_test_accuracy = np.mean(test_accuracies)
        print(f"Test Accuracy: {mean_test_accuracy:.4f}")
