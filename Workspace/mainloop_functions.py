import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, ROC
import torch_directml
import os
import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        self.should_stop = False

    def step(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"Validation loss improved to {current_loss:.4f}")
        else:
            self.epochs_no_improve += 1
            if self.verbose:
                print(f"No improvement for {self.epochs_no_improve}/{self.patience} epochs")
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def restore_best_weights(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)

# 0.9653515277365767, 5
class GraphFocalLoss(nn.Module):
    def __init__(self, alpha=.75, gamma=2, reduction='mean'):
        """
        Binary Focal Loss for models outputting 2-class logits
        
        Args:
            alpha (float): Weighting factor for positive class (default: 0.25)
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, 2) tensor of class logits
            targets: (N) tensor of class indices (0 or 1)
        """
        # Convert targets to one-hot encoding
        targets_onehot = F.one_hot(targets, num_classes=2).float()
        
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Extract probabilities of true classes
        pt = torch.sum(probs * targets_onehot, dim=1)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute alpha weights: alpha for class 1, (1-alpha) for class 0
        alpha_t = torch.where(targets == 1, 
                             self.alpha * torch.ones_like(pt),
                             (1 - self.alpha) * torch.ones_like(pt))
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train(model, optimizer, train_dataloader, max_epochs=20, patience=2):
    device = torch_directml.device()
    device = device if device else "cpu"
    device = "cpu"
    model.to(device)

    loss_fn = GraphFocalLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    early_stopper = EarlyStopping(patience=patience, verbose=True)

    epoch_losses = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = batch.to(device)
            if batch.x is None:
                print("Skipping batch due to missing node features")
                continue

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # loss = loss_fn(out, batch.y)
            loss = F.cross_entropy(out,batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        

        scheduler.step(avg_loss)
        early_stopper.step(avg_loss, model)

        if early_stopper.should_stop:
            print(f"Stopping early at epoch {epoch+1}")
            break

    early_stopper.restore_best_weights(model)
    return model, optimizer, early_stopper.best_loss, epoch_losses





def test(model, test_dataloader):
    device = torch_directml.device()
    device = device if device else "cpu"
    device = "cpu"
    model.to(device)

    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1score = F1Score(task="binary")
    roc_metric = ROC(task="binary")

    metrics = {
        "acc": [0, accuracy],
        "pre": [0, precision],
        "recall": [0, recall],
        "f1": [0, f1score],
    }

    model.eval()
    roc = 0
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            if batch.x is None:
                print("Skipping batch due to missing node features")
                continue

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            preds_prob = out.softmax(dim=1)
            fpr, tpr, _ = roc_metric(preds_prob, F.one_hot(batch.y, num_classes=2))
            roc += torch.trapz(tpr, fpr)
            for key in metrics:
                metrics[key][0] += metrics[key][1](preds, batch.y)

    for key in metrics:
        metrics[key][0] /= len(test_dataloader)
    roc /= len(test_dataloader)
    avg_loss = total_loss / len(test_dataloader)

    results = {
        "loss": avg_loss,
        "ROC": roc.item(),
        "accuracy": metrics["acc"][0].item(),
        "precision": metrics["pre"][0].item(),
        "recall": metrics["recall"][0].item(),
        "f1": metrics["f1"][0].item()
    }

    print("Test Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results