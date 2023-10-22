import torch
import torch.nn as nn

from mnist_example.datasets import mnist_dataloader
from mnist_example.models import CNN, FCN
from mnist_example.train_eval import eval_model, train_model


BATCH_SIZE = 50
LEARNING_RATE = 1e-3


def test_fcn() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs_fcn = 1

    train_loader_fcn = mnist_dataloader(batch_size=BATCH_SIZE, train=True, shuffle=True)
    eval_loader_fcn = mnist_dataloader(batch_size=BATCH_SIZE, train=False, shuffle=True)

    fcn = FCN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer_fcn = torch.optim.Adam(fcn.parameters(), lr=LEARNING_RATE)

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


def test_cnn() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs_cnn = 1

    train_loader_cnn = mnist_dataloader(batch_size=BATCH_SIZE, train=True, shuffle=True)
    eval_loader_cnn = mnist_dataloader(batch_size=BATCH_SIZE, train=False, shuffle=True)

    cnn = CNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

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
