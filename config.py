from dataclasses import dataclass


@dataclass
class FCN:
    input_dim: int
    output_dim: int
    hidden_dim_1: int
    hidden_dim_2: int
    dropout: float


@dataclass
class Training:
    batch_size: int
    n_epochs: int
    learning_rate: float


@dataclass
class CNN:
    output_dim: int
    dropout: float


@dataclass
class Params:
    fcn: FCN
    fcn_training: Training

    cnn: CNN
    cnn_training: Training


@dataclass
class Server:
    tracking_uri: str
    model_path: str
    batch_size: int
