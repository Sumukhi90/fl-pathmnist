
import numpy as np


def add_label_noise(labels, noise_rate=0.4, num_classes=9, seed=42):
    """
    Randomly replace a fraction of labels with incorrect classes.

    Returns:
        noisy_labels: Labels after corruption.
        noisy_indices: Positions whose labels were changed.
    """
    rng = np.random.default_rng(seed)

    noisy_labels = np.asarray(labels).reshape(-1).copy()

    num_noisy = int(len(noisy_labels) * noise_rate)

    noisy_indices = rng.choice(
        len(noisy_labels),
        size=num_noisy,
        replace=False
    )

    original_labels = noisy_labels[noisy_indices].copy()

    random_offsets = rng.integers(
        1, num_classes, size=num_noisy
    )

    noisy_labels[noisy_indices] = (
        original_labels + random_offsets
    ) % num_classes

    return noisy_labels, noisy_indices
