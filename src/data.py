
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from medmnist import PathMNIST


def load_pathmnist():
    """Load PathMNIST training and testing datasets."""

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = PathMNIST(
        split="train",
        transform=transform,
        download=True
    )

    test_dataset = PathMNIST(
        split="test",
        transform=transform,
        download=True
    )

    return train_dataset, test_dataset


class ClientDataset(Dataset):
    """Dataset containing only one client's assigned samples."""

    def __init__(self, base_dataset, indices, labels=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]
        image, original_label = self.base_dataset[original_index]

        if self.labels is not None:
            label = int(self.labels[position])
        else:
            label = int(original_label[0])

        return image, torch.tensor(label, dtype=torch.long)


def create_client_loader(
    dataset,
    indices,
    labels=None,
    batch_size=64,
    shuffle=True,
    generator=None
):
    """Create a reproducible DataLoader for one client."""

    client_data = ClientDataset(dataset, indices, labels)

    return DataLoader(
        client_data,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )
