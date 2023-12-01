import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dvc.fs import DVCFileSystem
from torch.utils.data import DataLoader


def check_files() -> bool:
    files_paths = [
        "data/MNIST/raw/t10k-images-idx3-ubyte",
        "data/MNIST/raw/t10k-labels-idx1-ubyte",
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte",
    ]
    result = True
    for path in files_paths:
        result &= os.path.exists(path)

    return result


def load_data_from_dvc():
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)


def mnist_dataloader(
    batch_size: int, train: bool, shuffle: bool = True, load_from_source: bool = False
) -> DataLoader:
    """The function creates a dataloader with preprocessed MNIST images

    Args:
        batch_size: batch size
        train: training or test part of the dataset
        shuffle: to shuffle or not to shuffle the data
        load_from_source: enable loading from source (not from dvc)

    Returns:
        train or test dataloader
    """
    if not check_files() and not load_from_source:
        load_data_from_dvc()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root="./data", train=train, download=load_from_source, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
