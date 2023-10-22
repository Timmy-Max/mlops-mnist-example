import os
import time

import hydra
import torch
import torch.nn as nn
from config import Params

from mnist_example.datasets import mnist_dataloader
from mnist_example.models import CNN, FCN
from mnist_example.train_eval import train_model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: Params) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader_fcn = mnist_dataloader(
        batch_size=cfg.fcn_training.batch_size, train=True, shuffle=True
    )

    train_loader_cnn = mnist_dataloader(
        batch_size=cfg.cnn_training.batch_size, train=True, shuffle=True
    )

    fcn = FCN(**dict(cfg.fcn)).to(device)
    cnn = CNN(**dict(cfg.cnn)).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer_fcn = torch.optim.Adam(
        fcn.parameters(), lr=cfg.fcn_training.learning_rate
    )
    optimizer_cnn = torch.optim.Adam(
        cnn.parameters(), lr=cfg.cnn_training.learning_rate
    )
    start_time = time.time()
    print("FCN training:")
    train_model(
        fcn,
        optimizer_fcn,
        loss_function,
        train_loader_fcn,
        cfg.fcn_training.n_epochs,
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
        cfg.cnn_training.n_epochs,
        device,
    )
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")

    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(cnn.state_dict(), "models/cnn.pt")
    print("CNN was successfully saved: models/cnn.pt")


if __name__ == "__main__":
    train()
