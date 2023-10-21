import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def mnist_dataloader(batch_size: int, train: bool, shuffle: bool = True) -> DataLoader:
    """The function creates a dataloader with preprocessed MNIST images

    Args:
        batch_size: batch size
        train: training or test part of the dataset
        shuffle: to shuffle or not to shuffle the data

    Returns:
        train or test dataloader
    """
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
