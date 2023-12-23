import os
import shutil

import hydra
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import Params
from mlflow import MlflowClient

from mnist_example.datasets import mnist_dataloader
from mnist_example.models import CNN, FCN, MNISTClassifier


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def convert_onnx(model: nn.Module, save_path: str, input_size: tuple[int]) -> None:
    model.eval()
    dummy_input = torch.randn((1,) + input_size, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["LOGITS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "LOGITS": {0: "BATCH_SIZE"}},
    )
    print(f"Model has been converted to ONNX and saved in {save_path}")


@hydra.main(config_path="configs", config_name="models", version_base="1.3")
def train(cfg: Params) -> None:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader_fcn = mnist_dataloader(
        batch_size=cfg.fcn_training.batch_size, train=True, shuffle=True
    )

    train_loader_cnn = mnist_dataloader(
        batch_size=cfg.cnn_training.batch_size, train=True, shuffle=True
    )

    fcn = MNISTClassifier(FCN, cfg.fcn, cfg.fcn_training)
    cnn = MNISTClassifier(CNN, cfg.cnn, cfg.fcn_training)

    fcn_trainer = pl.Trainer(max_epochs=cfg.fcn_training.n_epochs)
    cnn_trainer = pl.Trainer(max_epochs=cfg.cnn_training.n_epochs)

    remote_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.pytorch.autolog()

    mlflow.set_experiment("MNIST FCN and CNN training")

    with mlflow.start_run(run_name="fcn_training") as run:
        mlflow.log_params(cfg.fcn)
        fcn_trainer.fit(fcn, train_loader_fcn)
        input_size = next(iter(train_loader_fcn))[0].shape[1:]

    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(fcn.model.state_dict(), "models/fcn.pt")
    print("FCN was successfully saved: models/fcn.pt")

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    convert_onnx(fcn.model, "models/fcn.onnx", input_size)

    with mlflow.start_run(run_name="cnn_training") as run:
        mlflow.log_params(cfg.cnn)
        cnn_trainer.fit(cnn, train_loader_cnn)

    torch.save(cnn.model.state_dict(), "models/cnn.pt")
    print("CNN was successfully saved: models/cnn.pt")

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    input_size = next(iter(train_loader_cnn))[0].shape[1:]
    convert_onnx(cnn.model, "models/cnn.onnx", input_size)
    shutil.copyfile(
        "models/cnn.onnx", "triton_server/model_repository/cnn_onnx/1/model.onnx"
    )


if __name__ == "__main__":
    train()
