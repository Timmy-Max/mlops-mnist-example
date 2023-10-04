import os
import time

import torch
import torch.nn as nn
from config import CNN_CONFIG, CNN_TRAIN_CONFIG, FCN_CONFIG, FCN_TRAIN_CONFIG

from mnist_example.datasets import mnist_dataloader
from mnist_example.models import CNN, FCN
from mnist_example.train_eval import eval_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size_fcn = FCN_TRAIN_CONFIG["batch_size"]
    batch_size_cnn = CNN_TRAIN_CONFIG["batch_size"]
    batch_size = (batch_size_fcn + batch_size_cnn) // 2

    eval_loader = mnist_dataloader(batch_size=batch_size, train=False, shuffle=True)

    fcn = FCN(**FCN_CONFIG).to(device)
    cnn = CNN(**CNN_CONFIG).to(device)

    loss_function = nn.CrossEntropyLoss()

    fcn.load_state_dict(torch.load("models/fcn.pt"))
    cnn.load_state_dict(torch.load("models/cnn.pt"))

    start_time = time.time()
    fcn_loss, fcn_accuracy = eval_model(fcn, loss_function, eval_loader, device)
    print("FCN inference:")
    print(f"FCN eval loss = {fcn_loss:.3f}")
    print(f"FCN eval accuracy = {fcn_accuracy:.3f}")
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")

    print()

    start_time = time.time()
    cnn_loss, cnn_accuracy = eval_model(cnn, loss_function, eval_loader, device)
    print("CNN inference:")
    print(f"CNN eval loss = {cnn_loss:.3f}")
    print(f"CNN eval accuracy = {cnn_accuracy:.3f}")
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")

    if not os.path.exists("reports"):
        os.makedirs("reports")

    with open("reports/inference_report.txt", "w") as report:
        report.write(f"FCN eval loss = {fcn_loss}")
        report.write("\n")
        report.write(f"FCN eval accuracy = {fcn_accuracy}")
        report.write("\n")
        report.write(f"CNN eval loss = {cnn_loss}")
        report.write("\n")
        report.write(f"CNN eval accuracy = {cnn_accuracy}")
