import torch
import torch.nn as nn
from mnist_example.datasets import mnist_dataloader
from mnist_example.train import train_model
from mnist_example.eval import eval_model
from mnist_example.models import FCN, CNN
from config import (
    FCN_CONFIG,
    CNN_CONFIG,
    FCN_TRAIN_CONFIG,
    CNN_TRAIN_CONFIG,
    FCN_OPTIMIZER_CONFIG,
    CNN_OPTIMIZER_CONFIG,
)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size_fcn = FCN_TRAIN_CONFIG["batch_size"]
    n_epochs_fcn = FCN_TRAIN_CONFIG["n_epochs"]
    batch_size_cnn = CNN_TRAIN_CONFIG["batch_size"]
    n_epochs_cnn = CNN_TRAIN_CONFIG["n_epochs"]

    train_loader_fcn = mnist_dataloader(batch_size=batch_size_fcn, train=True, shuffle=True)
    eval_loader_fcn = mnist_dataloader(batch_size=batch_size_fcn, train=False, shuffle=True)
    train_loader_cnn = mnist_dataloader(batch_size=batch_size_cnn, train=True, shuffle=True)
    eval_loader_cnn = mnist_dataloader(batch_size=batch_size_cnn, train=False, shuffle=True)

    fcn = FCN(**FCN_CONFIG).to(device)
    cnn = CNN(**CNN_CONFIG).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer_fcn = torch.optim.Adam(fcn.parameters(), **FCN_OPTIMIZER_CONFIG)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), **CNN_OPTIMIZER_CONFIG)

    print("FCN training:")
    train_model(fcn, optimizer_fcn, loss_function, train_loader_fcn, n_epochs_fcn, device)
    fcn_loss, fcn_accuracy = eval_model(fcn, loss_function, eval_loader_fcn, device)

    print("CNN training:")
    train_model(cnn, optimizer_cnn, loss_function, train_loader_cnn, n_epochs_cnn, device)
    cnn_loss, cnn_accuracy = eval_model(cnn, loss_function, eval_loader_cnn, device)

    print("Evaluation:")
    print(f"FCN eval loss = {fcn_loss}")
    print(f"FCN eval accuracy = {fcn_accuracy}")
    print(f"CNN eval loss = {cnn_loss}")
    print(f"CNN eval accuracy = {cnn_accuracy}")
