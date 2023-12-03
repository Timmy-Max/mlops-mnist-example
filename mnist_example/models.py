import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall


class FCN(nn.Module):
    """Simple fully connected network"""

    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: size of the flattened image
            output_dim: number of classes
            hidden_dim_1: first hidden dimension size
            hidden_dim_2: second hidden dimension size
        """
        super(FCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.input_dim)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    """Simple convolutional network"""

    def __init__(self, output_dim: int = 10, dropout: float = 0.5):
        """
        Args:
            output_dim: number of classes
            dropout: dropout probability
        """
        super(CNN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(7 * 7 * 64, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MNISTClassifier(pl.LightningModule):
    def __init__(self, model, model_params, training_params):
        super().__init__()
        self.model = model(**model_params)
        num_classes = model_params["output_dim"]
        self.accuracy = Accuracy("multiclass", num_classes=num_classes)
        self.top_3_accuracy = Accuracy("multiclass", num_classes=num_classes, top_k=3)
        self.precision = Precision("multiclass", num_classes=num_classes)
        self.recall = Recall("multiclass", num_classes=num_classes)
        self.f1 = F1Score("multiclass", num_classes=num_classes)
        self.lr = training_params["learning_rate"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)
        precision = self.precision(pred, y)
        recall = self.recall(pred, y)
        f1 = self.f1(pred, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        self.log("train_precision", precision, on_epoch=True)
        self.log("train_recall", recall, on_epoch=True)
        self.log("train_f1", f1, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)
        precision = self.precision(pred, y)
        recall = self.recall(pred, y)
        f1 = self.f1(pred, y)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)
        return loss, acc, precision, recall, f1
