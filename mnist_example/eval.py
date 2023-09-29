import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
