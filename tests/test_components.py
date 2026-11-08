
import numpy as np
import pytest
import torch

from torch.utils.data import Dataset, DataLoader

from src.partition import dirichlet_partition
from src.noise import add_label_noise
from src.models import PathMNISTCNN
from src.data import create_client_loader
from src.aggregation import fedavg_aggregate
from src.evaluate import evaluate_model


# ============================================================
# SYNTHETIC DATASET FOR TESTING
# ============================================================

class DummyPathDataset(Dataset):
    """
    Small synthetic dataset for testing without downloading
    the actual PathMNIST dataset.
    """

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(3, 28, 28)
        label = np.array([index % 9])

        return image, label


# ============================================================
# TEST 1: NON-IID PARTITIONING
# ============================================================

def test_partition_covers_all_samples():

    labels = np.tile(np.arange(9), 100)

    parts = dirichlet_partition(
        labels,
        num_clients=5,
        alpha=0.3,
        seed=42
    )

    all_indices = np.concatenate(parts)

    assert len(parts) == 5
    assert len(all_indices) == len(labels)
    assert len(np.unique(all_indices)) == len(labels)

    assert np.array_equal(
        np.sort(all_indices),
        np.arange(len(labels))
    )


# ============================================================
# TEST 2: PARTITION REPRODUCIBILITY
# ============================================================

def test_partition_is_reproducible():

    labels = np.tile(np.arange(9), 100)

    first = dirichlet_partition(
        labels, 5, 0.3, 42
    )

    second = dirichlet_partition(
        labels, 5, 0.3, 42
    )

    assert first == second


# ============================================================
# TEST 3: LABEL NOISE
# ============================================================

def test_label_noise_changes_exact_percentage():

    labels = np.tile(np.arange(9), 100)

    noisy, changed = add_label_noise(
        labels,
        noise_rate=0.4,
        num_classes=9,
        seed=42
    )

    assert len(changed) == 360

    assert np.all(
        noisy[changed] != labels[changed]
    )

    unchanged_indices = np.setdiff1d(
        np.arange(len(labels)),
        changed
    )

    assert np.array_equal(
        noisy[unchanged_indices],
        labels[unchanged_indices]
    )


# ============================================================
# TEST 4: CNN OUTPUT SHAPE
# ============================================================

def test_cnn_output_shape():

    model = PathMNISTCNN(num_classes=9)

    images = torch.randn(8, 3, 28, 28)

    outputs = model(images)

    assert outputs.shape == (8, 9)


# ============================================================
# TEST 5: CLIENT DATA LOADER
# ============================================================

def test_client_data_loader():

    dataset = DummyPathDataset(size=100)

    indices = list(range(50))

    loader = create_client_loader(
        dataset,
        indices,
        batch_size=16,
        shuffle=False
    )

    images, labels = next(iter(loader))

    assert len(loader.dataset) == 50

    assert images.shape == (16, 3, 28, 28)

    assert labels.shape == (16,)

    assert labels.dtype == torch.long

    assert labels.min() >= 0

    assert labels.max() <= 8


# ============================================================
# TEST 6: FEDAVG WEIGHTED AGGREGATION
# ============================================================

def test_fedavg_weighted_aggregation():

    client_weights = [
        {
            "weight": torch.tensor([1.0, 2.0])
        },
        {
            "weight": torch.tensor([3.0, 4.0])
        }
    ]

    client_sizes = [1, 3]

    result = fedavg_aggregate(
        client_weights,
        client_sizes
    )

    expected = torch.tensor([2.5, 3.5])

    assert torch.allclose(
        result["weight"],
        expected
    )


# ============================================================
# TEST 7: FEDAVG EMPTY CLIENT VALIDATION
# ============================================================

def test_fedavg_rejects_empty_clients():

    with pytest.raises(ValueError):

        fedavg_aggregate([], [])


# ============================================================
# TEST 8: FEDAVG INVALID CLIENT SIZE VALIDATION
# ============================================================

def test_fedavg_rejects_invalid_client_sizes():

    client_weights = [
        {
            "weight": torch.tensor([1.0])
        }
    ]

    with pytest.raises(ValueError):

        fedavg_aggregate(
            client_weights,
            [0]
        )


# ============================================================
# TEST 9: EMPTY TEST DATASET VALIDATION
# ============================================================

def test_evaluation_rejects_empty_dataset():

    dataset = DummyPathDataset(size=0)

    test_loader = DataLoader(
        dataset,
        batch_size=16
    )

    model = PathMNISTCNN(num_classes=9)

    with pytest.raises(ValueError):

        evaluate_model(
            model,
            test_loader
        )
