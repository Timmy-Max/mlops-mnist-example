import os
import time

import hydra
import torch
from config import Params
from pytorch_lightning import Trainer

from mnist_example.datasets import mnist_dataloader
from mnist_example.models import CNN, FCN, MNISTClassifier


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: Params) -> None:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = (cfg.fcn_training.batch_size + cfg.cnn_training.batch_size) // 2
    eval_loader = mnist_dataloader(batch_size=batch_size, train=False, shuffle=True)

    fcn = MNISTClassifier(FCN, cfg.fcn, cfg.fcn_training)
    fcn.model.load_state_dict(torch.load("models/fcn.pt"))
    cnn = MNISTClassifier(CNN, cfg.cnn, cfg.cnn_training)
    cnn.model.load_state_dict(torch.load("models/cnn.pt"))

    assert os.path.isfile("models/fcn.pt") and os.path.isfile(
        "models/cnn.pt"
    ), "You have to train the models first. To train the models, run train.py."

    fcn.model.load_state_dict(torch.load("models/fcn.pt"))
    cnn.model.load_state_dict(torch.load("models/cnn.pt"))

    start_time = time.time()
    trainer = Trainer()
    metrics_fcn = trainer.test(model=fcn, dataloaders=eval_loader)
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")

    start_time = time.time()
    trainer = Trainer()
    metrics_cnn = trainer.test(model=cnn, dataloaders=eval_loader)
    print(f"Elapsed time: {(time.time() - start_time):.3f} sec")

    if not os.path.exists("reports"):
        os.makedirs("reports")

    with open("reports/inference_report.txt", "w") as report:
        for key, value in metrics_fcn[0].items():
            report.write(f"FCN {key} = {value}")

        for key, value in metrics_cnn[0].items():
            report.write(f"CNN {key} = {value}")

    print("Report was successfully saved: reports/inference_report.txt")


if __name__ == "__main__":
    infer()
