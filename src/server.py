
import copy

from src.client import train_local_model
from src.aggregation import fedavg_aggregate


def run_federated_training(
    global_model,
    client_loaders,
    rounds=5,
    local_epochs=1,
    learning_rate=0.01,
    mu=0.0,
    device="cpu"
):
    """
    Run federated training across multiple clients.

    mu = 0.0: FedAvg
    mu > 0.0: FedProx
    """

    global_model = global_model.to(device)
    history = []

    for round_num in range(1, rounds + 1):

        client_weights = []
        client_sizes = []
        client_losses = []
        client_accuracies = []

        print(f"\nFederated Round {round_num}/{rounds}")

        for client_id, loader in enumerate(client_loaders):

            local_model = copy.deepcopy(global_model)

            weights, loss, accuracy = train_local_model(
                model=local_model,
                train_loader=loader,
                device=device,
                epochs=local_epochs,
                learning_rate=learning_rate,
                mu=mu
            )

            client_weights.append(weights)
            client_sizes.append(len(loader.dataset))
            client_losses.append(loss)
            client_accuracies.append(accuracy)

            print(
                f"Client {client_id + 1}: "
                f"Loss={loss:.4f}, "
                f"Accuracy={accuracy:.4f}"
            )

        global_weights = fedavg_aggregate(
            client_weights,
            client_sizes
        )

        global_model.load_state_dict(global_weights)

        round_loss = sum(
            loss * size
            for loss, size in zip(client_losses, client_sizes)
        ) / sum(client_sizes)

        round_accuracy = sum(
            accuracy * size
            for accuracy, size in zip(client_accuracies, client_sizes)
        ) / sum(client_sizes)

        history.append({
            "round": round_num,
            "loss": round_loss,
            "accuracy": round_accuracy
        })

        print(
            f"Global Training Loss: {round_loss:.4f} | "
            f"Weighted Client Accuracy: {round_accuracy:.4f}"
        )

    return global_model, history
