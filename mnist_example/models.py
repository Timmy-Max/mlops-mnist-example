import torch.nn as nn
from torch import Tensor


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
