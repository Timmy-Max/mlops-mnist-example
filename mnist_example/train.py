import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    loss_function: nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    device: torch.device,
):
    """Function trains the model"""
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            out = model(data)

            loss = loss_function(out, labels)
            epoch_loss += loss.item()
            _, out = torch.max(out.data, 1)
            epoch_acc += (out == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader.dataset)
        print(
            f"Epoch [{epoch + 1}/{n_epochs}]: loss = {epoch_loss:.5f}, accuracy = {epoch_acc:.5f}"
        )
