import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dvc.fs import DVCFileSystem
from torch.utils.data import DataLoader


def check_files():
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


def mnist_dataloader(batch_size: int, train: bool, shuffle: bool = True) -> DataLoader:
    """The function creates a dataloader with preprocessed MNIST images

    Args:
        batch_size: batch size
        train: training or test part of the dataset
        shuffle: to shuffle or not to shuffle the data

    Returns:
        train or test dataloader
    """
    if not check_files():
        fs = DVCFileSystem()
        fs.get("data", "data", recursive=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root="./data", train=train, download=False, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
