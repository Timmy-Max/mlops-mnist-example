from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def mnist_dataloader(batch_size: int, train: bool, shuffle: bool = True) -> DataLoader:
    """The function creates a dataloader with preprocessed MNIST images"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root="./data", train=train, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
