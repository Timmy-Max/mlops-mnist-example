import os
import time

import torch
import torch.nn as nn
from config import (
    CNN_CONFIG,
    CNN_OPTIMIZER_CONFIG,
    CNN_TRAIN_CONFIG,
    FCN_CONFIG,
    FCN_OPTIMIZER_CONFIG,
    FCN_TRAIN_CONFIG,
)

from mnist_example.datasets import mnist_dataloader
from mnist_example.models import CNN, FCN
from mnist_example.train_eval import train_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size_fcn = FCN_TRAIN_CONFIG["batch_size"]
    n_epochs_fcn = FCN_TRAIN_CONFIG["n_epochs"]

    batch_size_cnn = CNN_TRAIN_CONFIG["batch_size"]
    n_epochs_cnn = CNN_TRAIN_CONFIG["n_epochs"]

    train_loader_fcn = mnist_dataloader(
        batch_size=batch_size_fcn, train=True, shuffle=True
    )

    train_loader_cnn = mnist_dataloader(
        batch_size=batch_size_cnn, train=True, shuffle=True
    )

    fcn = FCN(**FCN_CONFIG).to(device)
    cnn = CNN(**CNN_CONFIG).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer_fcn = torch.optim.Adam(fcn.parameters(), **FCN_OPTIMIZER_CONFIG)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), **CNN_OPTIMIZER_CONFIG)

    start_time = time.time()
    print("FCN training:")
    train_model(
        fcn,
        optimizer_fcn,
        loss_function,
        train_loader_fcn,
        n_epochs_fcn,
        device,
    )
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")
    if os.path.exists("models"):
        torch.save(fcn.state_dict(), "models/fcn.pt")
    else:
        os.makedirs("models")
        torch.save(fcn.state_dict(), "models/fcn.pt")
    print("FCN was successfully saved: models/fcn.pt")

    print()

    start_time = time.time()
    print("CNN training:")
    train_model(
        cnn,
        optimizer_cnn,
        loss_function,
        train_loader_cnn,
        n_epochs_cnn,
        device,
    )
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")

    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(cnn.state_dict(), "models/cnn.pt")
    print("CNN was successfully saved: models/cnn.pt")
