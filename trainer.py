"""
Word2Vec from Scratch: Trainer
Based on https://github.com/OlgaChernytska/word2vec-pytorch
Updated to avoid deprecated libraries like Torchtext
"""
import os
import json
from typing import Any
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
# pylint: disable=too-many-instance-attributes
class Trainer:
    """Main class for model training"""

    model: nn.Module
    epochs: int
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    criterion: Any
    optimizer: Any
    lr_scheduler: Any
    train_steps: int = None # limit the number of steps
    val_steps: int = None # limit the number of steps
    checkpoint_frequency: int = None # number of steps between saves


    def __post_init__(self):
        self.loss = {"train": [], "val": []}
        self.model_dir = (
            self.model.__class__.__name__ + "_data"
        )  # e.g. CBOWModel or SkipGramModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        """Train model for `self.epochs` epochs"""
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                f"""Epoch: {epoch+1}/{self.epochs} 
                Train Loss={self.loss['train'][-1]:.5f},
                Val Loss{self.loss['val'][-1]:.5f}"""
            )
            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
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
