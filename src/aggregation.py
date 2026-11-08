
import torch


def fedavg_aggregate(client_weights, client_sizes):
    """
    Aggregate client model parameters using sample-size-weighted
    Federated Averaging (FedAvg).

    Each client's contribution is proportional to the number
    of training samples it holds.

    Formula:
        W_global = sum(n_k / N * W_k)

    Where:
        W_global = aggregated global model weights
        W_k      = local model weights of client k
        n_k      = number of training samples at client k
        N        = total number of training samples across clients

    Args:
        client_weights (list):
            List of local model state dictionaries returned
            by the participating clients.

        client_sizes (list):
            Number of training samples belonging to each client.

    Returns:
        dict:
            Aggregated global model state dictionary.

    Raises:
        ValueError:
            If client weights or sizes are missing,
            lengths do not match, sample sizes are invalid,
            or model state dictionaries have different keys.
    """

    # --------------------------------------------------------
    # 1. VALIDATE CLIENT INPUTS
    # --------------------------------------------------------

    if not client_weights:
        raise ValueError(
            "No client weights provided for aggregation."
        )

    if not client_sizes:
        raise ValueError(
            "No client sample sizes provided."
        )

    if len(client_weights) != len(client_sizes):
        raise ValueError(
            "Number of client weight dictionaries "
            "must match number of client sizes."
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError(
            "All client sample sizes must be positive."
        )

    # --------------------------------------------------------
    # 2. VALIDATE MODEL STATE DICTIONARIES
    # --------------------------------------------------------

    reference_keys = set(client_weights[0].keys())

    for client_id, weights in enumerate(client_weights):

        if set(weights.keys()) != reference_keys:
            raise ValueError(
                f"Model state dictionary mismatch "
                f"for Client {client_id + 1}."
            )

    # --------------------------------------------------------
    # 3. CALCULATE TOTAL NUMBER OF CLIENT SAMPLES
    # --------------------------------------------------------

    total_samples = sum(client_sizes)

    # --------------------------------------------------------
    # 4. INITIALIZE GLOBAL MODEL WEIGHTS
    # --------------------------------------------------------

    global_weights = {}

    for key in client_weights[0].keys():

        global_weights[key] = torch.zeros_like(
            client_weights[0][key]
        )

        # ----------------------------------------------------
        # 5. COMPUTE SAMPLE-SIZE-WEIGHTED AVERAGE
        # ----------------------------------------------------

        for weights, size in zip(
            client_weights,
            client_sizes
        ):

            client_weight = size / total_samples

            global_weights[key] += (
                weights[key] * client_weight
            )

    # --------------------------------------------------------
    # 6. RETURN AGGREGATED GLOBAL MODEL
    # --------------------------------------------------------

    return global_weights
