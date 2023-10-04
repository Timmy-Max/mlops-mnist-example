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
from mnist_example.train_eval import eval_model, train_model


def test_fcn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size_fcn = FCN_TRAIN_CONFIG["batch_size"]
    n_epochs_fcn = 1

    train_loader_fcn = mnist_dataloader(
        batch_size=batch_size_fcn, train=True, shuffle=True
    )
    eval_loader_fcn = mnist_dataloader(
        batch_size=batch_size_fcn, train=False, shuffle=True
    )

    fcn = FCN(**FCN_CONFIG).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer_fcn = torch.optim.Adam(fcn.parameters(), **FCN_OPTIMIZER_CONFIG)

    init_eval_loss, init_eval_accuracy = eval_model(
        fcn, loss_function, eval_loader_fcn, device
    )
    train_model(
        fcn,
        optimizer_fcn,
        loss_function,
        train_loader_fcn,
        n_epochs_fcn,
        device,
    )
    epoch_1_eval_loss, epoch_1_eval_accuracy = eval_model(
        fcn, loss_function, eval_loader_fcn, device
    )
    assert (
        init_eval_loss >= epoch_1_eval_loss
        and init_eval_accuracy <= epoch_1_eval_accuracy
    ), "FCN is not learning"


def test_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size_cnn = CNN_TRAIN_CONFIG["batch_size"]
    n_epochs_cnn = 1

    train_loader_cnn = mnist_dataloader(
        batch_size=batch_size_cnn, train=True, shuffle=True
    )
    eval_loader_cnn = mnist_dataloader(
        batch_size=batch_size_cnn, train=False, shuffle=True
    )

    cnn = CNN(**CNN_CONFIG).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), **CNN_OPTIMIZER_CONFIG)

    init_eval_loss, init_eval_accuracy = eval_model(
        cnn, loss_function, eval_loader_cnn, device
    )
    train_model(
        cnn,
        optimizer_cnn,
        loss_function,
        train_loader_cnn,
        n_epochs_cnn,
        device,
    )
    epoch_1_eval_loss, epoch_1_eval_accuracy = eval_model(
        cnn, loss_function, eval_loader_cnn, device
    )
    assert (
        init_eval_loss >= epoch_1_eval_loss
        and init_eval_accuracy <= epoch_1_eval_accuracy
    ), "CNN is not learning"
