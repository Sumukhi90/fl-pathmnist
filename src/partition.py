
import numpy as np


def dirichlet_partition(labels, num_clients=5, alpha=0.3, seed=42):
    """
    Divide dataset indices among clients using a
    Non-IID Dirichlet distribution.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).reshape(-1)

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.repeat(alpha, num_clients)
        )

        split_points = (
            np.cumsum(proportions)[:-1] * len(class_indices)
        ).astype(int)

        splits = np.split(class_indices, split_points)

        for client_id, indices in enumerate(splits):
            client_indices[client_id].extend(indices.tolist())

    for indices in client_indices:
        rng.shuffle(indices)

    return client_indices
