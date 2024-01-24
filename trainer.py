"""
Word2Vec from Scratch: Trainer
"""
import os
import time
import json
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from dataloader import W2VDataLoader


@dataclass
# pylint: disable=too-many-instance-attributes
class Trainer:
    """Main class for model training"""

    model: nn.Module
    epochs: int
    learning_rate: float
    dataloader: W2VDataLoader
    model_dir: str
    train_steps: int = None  # limit the number of steps
    val_steps: int = None  # limit the number of steps
    checkpoint_frequency: int = None  # number of steps between saves

    def __post_init__(self):
        self.loss = {"train": [], "val": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        self.criterion = nn.CrossEntropyLoss()  # loss function

        # set up optimizer (e.g. gradient descent calculation) and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = LambdaLR(
            self.optimizer,
            # variable learning rate that ends at 0 in the last epoch
            lr_lambda=lambda epoch: (self.epochs - epoch) / self.epochs,
            verbose=True,
        )
        self.model.to(self.device)

    def train(self):
        """Train model for `self.epochs` epochs"""
        start_time = time.time()
        print(f"running on {self.device}")
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            current_time = time.time()
            print(
                f"""Epoch: {epoch+1}/{self.epochs} 
                Train Loss={self.loss['train'][-1]:.5f},
                Val Loss={self.loss['val'][-1]:.5f}
                Seconds Elapsed={current_time-start_time}"""
            )
            self.lr_scheduler.step()  # adjust learning rate

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()  # train mode for model
        running_loss = []
        batches = len(self.dataloader.train)

        for i, batch_data in enumerate(self.dataloader.train, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            self.optimizer.zero_grad()  # clear gradient
            outputs = self.model(inputs)  # generate predictions from current model
            loss = self.criterion(outputs, labels)  # calculate loss
            loss.backward()  # calculate derivative
            self.optimizer.step()  # apply gradient descent to model
            running_loss.append(loss.item())

            if i % int(batches / 25) == 0:
                print(f"Batch: {i} / {batches}")

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.dataloader.val, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())
                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = f"checkpoint_{str(epoch_num).zfill(3)}.pt"
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w", encoding="utf-8") as file:
            json.dump(self.loss, file)
