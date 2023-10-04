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
    """Function trains the model

    Args:
        model: model to train
        optimizer: optimizer
        loss_function: loss function
        train_loader: dataloader with training data
        n_epochs: number of training epochs
        device: device on which model will be trained
    """
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
            f"Epoch [{epoch + 1}/{n_epochs}]: loss = {epoch_loss:.3f}, accuracy = {epoch_acc:.3f}"
        )


def eval_model(
    model: nn.Module,
    loss_function: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Function checks the quality of the model and returns the loss and accuracy

    Args:
        model: model to evaluate
        loss_function: loss function
        eval_loader: dataloader with evaluation data
        device: the device on which model will be tested

    Returns:
        loss value and accuracy on whole evaluation dataset
    """
    model.eval()
    accuracy = 0
    loss_value = 0
    with torch.no_grad():
        for data, labels in eval_loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)
            loss_value += loss_function(out, labels).item()
            _, out = torch.max(out.data, 1)
            accuracy += (out == labels).sum().item()

    loss_value /= len(eval_loader)
    accuracy /= len(eval_loader.dataset)
    return loss_value, accuracy
