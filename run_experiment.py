
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.tracking import start_run, log_metrics, finish_run
from src.data import load_pathmnist, create_client_loader
from src.partition import dirichlet_partition
from src.noise import add_label_noise
from src.models import PathMNISTCNN
from src.client import train_local_model
from src.aggregation import fedavg_aggregate
from src.evaluate import evaluate_model


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 5
ALPHA = 0.3

NOISE_RATE = 0.4
NOISY_CLIENT = 4  # Client 5 (zero-based indexing)

CLIENT_VALIDATION_FRACTION = 0.20

ROUNDS = 2  # Preliminary smoke test; increase after validation
LOCAL_EPOCHS = 1
BATCH_SIZE = 128

LEARNING_RATE = 0.01
FEDPROX_MU = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WANDB_PROJECT = "indiaai-federated-pathmnist"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):
    """Set random seeds across Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

print("Using device:", DEVICE)
print("Random seed:", SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================================================
# LOAD PATHMNIST
# ============================================================

print("\nLoading PathMNIST...")

train_dataset, test_dataset = load_pathmnist()

train_labels = np.asarray(
    train_dataset.labels
).reshape(-1)


# ============================================================
# NON-IID CLIENT PARTITIONING
# ============================================================

client_indices = dirichlet_partition(
    train_labels,
    num_clients=NUM_CLIENTS,
    alpha=ALPHA,
    seed=SEED
)


# ============================================================
# CREATE CLIENT TRAINING AND VALIDATION LOADERS
# ============================================================

client_loaders = []
client_generators = []
client_validation_loaders = []

for client_id, indices in enumerate(client_indices):

    # Reproducible client-specific split.
    split_rng = np.random.default_rng(SEED + client_id)

    shuffled_indices = np.asarray(indices).copy()
    split_rng.shuffle(shuffled_indices)

    split_point = int(
        len(shuffled_indices) *
        (1 - CLIENT_VALIDATION_FRACTION)
    )

    train_indices = shuffled_indices[:split_point]
    validation_indices = shuffled_indices[split_point:]

    # Original clean training labels.
    client_labels = train_labels[train_indices].copy()

    # Apply label noise only to Client 5's training labels.
    if client_id == NOISY_CLIENT:

        client_labels, noisy_indices = add_label_noise(
            client_labels,
            noise_rate=NOISE_RATE,
            num_classes=9,
            seed=SEED
        )

        print(
            f"Client {client_id + 1}: "
            f"Train={len(train_indices)}, "
            f"Validation={len(validation_indices)}, "
            f"Noisy training labels={len(noisy_indices)}"
        )

    else:

        print(
            f"Client {client_id + 1}: "
            f"Train={len(train_indices)}, "
            f"Validation={len(validation_indices)}, "
            f"Noisy training labels=0"
        )

    # Reproducible client training generator.
    client_generator = torch.Generator()
    client_generator.manual_seed(SEED + client_id)

    train_loader = create_client_loader(
        train_dataset,
        train_indices,
        labels=client_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=client_generator
    )

    # Validation uses original clean labels.
    validation_loader = create_client_loader(
        train_dataset,
        validation_indices,
        labels=None,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    client_loaders.append(train_loader)
    client_generators.append(client_generator)
    client_validation_loaders.append(validation_loader)


# ============================================================
# GLOBAL TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# METRIC HELPERS
# ============================================================

def get_update_vector(new_state, old_state):
    """Calculate flattened model parameter differences."""

    return torch.cat([
        (
            new_state[key].float()
            - old_state[key].float()
        ).reshape(-1)

        for key in old_state
    ])


def calculate_update_metrics(client_updates, global_update):
    """Calculate client update norms and cosine similarities."""

    global_norm = torch.norm(global_update).item()

    metrics = []

    for update in client_updates:

        client_norm = torch.norm(update).item()

        if client_norm == 0 or global_norm == 0:
            cosine = 0.0
        else:
            cosine = torch.nn.functional.cosine_similarity(
                update.unsqueeze(0),
                global_update.unsqueeze(0),
                dim=1
            ).item()

        metrics.append({
            "update_norm": float(client_norm),
            "cosine_similarity": float(cosine)
        })

    return metrics, float(global_norm)


# ============================================================
# FEDERATED EXPERIMENT
# ============================================================

def run_experiment(method, mu):

    print("\n" + "=" * 55)
    print(f"STARTING {method}")
    print("=" * 55)

    # Reset random states before each algorithm.
    seed_everything(SEED)

    # Give both algorithms identical client batch ordering.
    for client_id, generator in enumerate(client_generators):
        generator.manual_seed(SEED + client_id)

    # Initialize global model.
    global_model = PathMNISTCNN().to(DEVICE)

    history = []

    # --------------------------------------------------------
    # WEIGHTS & BIASES INITIALIZATION
    # --------------------------------------------------------

    wandb_run = start_run(
        project=WANDB_PROJECT,
        run_name=method.lower(),
        config={
            "method": method,
            "num_clients": NUM_CLIENTS,
            "alpha": ALPHA,
            "noise_rate": NOISE_RATE,
            "noisy_client": NOISY_CLIENT + 1,
            "client_validation_fraction": CLIENT_VALIDATION_FRACTION,
            "fedprox_mu": mu,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "device": str(DEVICE)
        }
    )

    try:

        # ----------------------------------------------------
        # FEDERATED TRAINING ROUNDS
        # ----------------------------------------------------

        for round_num in range(1, ROUNDS + 1):

            print(
                f"\n{method} | "
                f"Round {round_num}/{ROUNDS}"
            )

            # Save previous global model state.
            old_state = {
                key: value.detach().clone()
                for key, value in global_model.state_dict().items()
            }

            client_weights = []
            client_sizes = []
            client_updates = []

            client_accuracies = []
            client_losses = []

            # ------------------------------------------------
            # LOCAL CLIENT TRAINING
            # ------------------------------------------------

            for client_id, loader in enumerate(client_loaders):

                local_model = copy.deepcopy(global_model)

                weights, loss, accuracy = train_local_model(
                    model=local_model,
                    train_loader=loader,
                    device=DEVICE,
                    epochs=LOCAL_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    mu=mu
                )

                client_weights.append(weights)

                # Use training sample count for aggregation.
                client_sizes.append(len(loader.dataset))

                client_accuracies.append(float(accuracy))
                client_losses.append(float(loss))

                client_updates.append(
                    get_update_vector(
                        weights,
                        old_state
                    )
                )

                print(
                    f"Client {client_id + 1} Training: "
                    f"Loss={loss:.4f}, "
                    f"Accuracy={accuracy:.4f}"
                )

            # ------------------------------------------------
            # FEDERATED AGGREGATION
            # ------------------------------------------------

            aggregated_weights = fedavg_aggregate(
                client_weights,
                client_sizes
            )

            global_model.load_state_dict(aggregated_weights)

            # Calculate global model update.
            global_update = get_update_vector(
                global_model.state_dict(),
                old_state
            )

            update_metrics, global_update_norm = (
                calculate_update_metrics(
                    client_updates,
                    global_update
                )
            )

            # ------------------------------------------------
            # GLOBAL MODEL EVALUATION
            # ------------------------------------------------

            test_metrics = evaluate_model(
                global_model,
                test_loader,
                device=DEVICE
            )

            # ------------------------------------------------
            # PER-CLIENT CLEAN VALIDATION
            # ------------------------------------------------

            client_validation_metrics = []

            for client_id, validation_loader in enumerate(
                client_validation_loaders
            ):

                validation_metrics = evaluate_model(
                    global_model,
                    validation_loader,
                    device=DEVICE
                )

                client_validation_metrics.append(
                    validation_metrics
                )

                print(
                    f"Client {client_id + 1} Validation: "
                    f"Loss={validation_metrics['test_loss']:.4f}, "
                    f"Accuracy={validation_metrics['test_accuracy']:.4f}"
                )

            # ------------------------------------------------
            # ROUND-LEVEL METRICS
            # ------------------------------------------------

            row = {
                "method": method,
                "round": round_num,

                "test_loss": float(
                    test_metrics["test_loss"]
                ),

                "test_accuracy": float(
                    test_metrics["test_accuracy"]
                ),

                "global_update_norm": float(
                    global_update_norm
                ),

                "mean_client_training_accuracy": float(
                    np.mean(client_accuracies)
                ),

                "mean_client_training_loss": float(
                    np.mean(client_losses)
                ),

                "mean_client_validation_accuracy": float(
                    np.mean([
                        metrics["test_accuracy"]
                        for metrics in client_validation_metrics
                    ])
                ),

                "mean_client_validation_loss": float(
                    np.mean([
                        metrics["test_loss"]
                        for metrics in client_validation_metrics
                    ])
                )
            }

            # ------------------------------------------------
            # INDIVIDUAL CLIENT METRICS
            # ------------------------------------------------

            for client_id, metrics in enumerate(update_metrics):

                index = client_id + 1

                row[f"client_{index}_training_loss"] = float(
                    client_losses[client_id]
                )

                row[f"client_{index}_training_accuracy"] = float(
                    client_accuracies[client_id]
                )

                row[f"client_{index}_validation_loss"] = float(
                    client_validation_metrics[client_id]["test_loss"]
                )

                row[f"client_{index}_validation_accuracy"] = float(
                    client_validation_metrics[client_id]["test_accuracy"]
                )

                row[f"client_{index}_update_norm"] = float(
                    metrics["update_norm"]
                )

                row[f"client_{index}_cosine"] = float(
                    metrics["cosine_similarity"]
                )

            history.append(row)

            # ------------------------------------------------
            # W&B METRIC LOGGING
            # ------------------------------------------------

            log_metrics(
                row,
                step=round_num,
                run=wandb_run
            )

            print(
                f"\nGlobal Test Loss: "
                f"{test_metrics['test_loss']:.4f} | "
                f"Global Test Accuracy: "
                f"{test_metrics['test_accuracy']:.4f}"
            )

            print(
                f"Mean Client Validation Accuracy: "
                f"{row['mean_client_validation_accuracy']:.4f}"
            )

            print(
                f"Global Update Norm: "
                f"{global_update_norm:.4f}"
            )

        # ----------------------------------------------------
        # SAVE MODEL CHECKPOINT
        # ----------------------------------------------------

        torch.save(
            global_model.state_dict(),
            f"results/{method.lower()}_model.pth"
        )

        # ----------------------------------------------------
        # SAVE INDIVIDUAL ALGORITHM METRICS
        # ----------------------------------------------------

        pd.DataFrame(history).to_csv(
            f"results/{method.lower()}_metrics.csv",
            index=False
        )

        print(
            f"\n{method} model and metrics saved."
        )

    finally:

        # Finish corresponding W&B run.
        finish_run(wandb_run)

    return history


# ============================================================
# RUN FEDAVG
# ============================================================

fedavg_history = run_experiment(
    "FedAvg",
    mu=0.0
)


# ============================================================
# RUN FEDPROX
# ============================================================

fedprox_history = run_experiment(
    "FedProx",
    mu=FEDPROX_MU
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = fedavg_history + fedprox_history

results_df = pd.DataFrame(all_results)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

results_df.to_csv(
    "results/experiment_results.csv",
    index=False
)

with open(
    "results/experiment_results.json",
    "w"
) as file:

    json.dump(
        all_results,
        file,
        indent=4
    )


# ============================================================
# VISUALIZATION 1: GLOBAL ACCURACY AND LOSS
# ============================================================

for metric, ylabel, title, filename in [

    (
        "test_accuracy",
        "Global Test Accuracy",
        "FedAvg vs FedProx: Global Accuracy",
        "global_test_accuracy.png"
    ),

    (
        "test_loss",
        "Global Test Loss",
        "FedAvg vs FedProx: Global Loss",
        "global_test_loss.png"
    )

]:

    plt.figure(figsize=(9, 5))

    for method in ["FedAvg", "FedProx"]:

        subset = results_df[
            results_df["method"] == method
        ]

        plt.plot(
            subset["round"],
            subset[metric],
            marker="o",
            label=method
        )

    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"plots/{filename}",
        dpi=300
    )

    plt.close()


# ============================================================
# VISUALIZATION 2: FEDAVG CLIENT TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

for client_id in range(1, NUM_CLIENTS + 1):

    column = f"client_{client_id}_training_accuracy"

    plt.plot(
        range(1, len(fedavg_history) + 1),

        [
            row[column]
            for row in fedavg_history
        ],

        marker="o",
        label=f"Client {client_id}"
    )

plt.xlabel("Federated Round")
plt.ylabel("Training Accuracy")
plt.title("FedAvg: Individual Client Training Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/client_training_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 3: FEDAVG CLIENT VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

for client_id in range(1, NUM_CLIENTS + 1):

    column = f"client_{client_id}_validation_accuracy"

    plt.plot(
        range(1, len(fedavg_history) + 1),

        [
            row[column]
            for row in fedavg_history
        ],

        marker="o",
        label=f"Client {client_id}"
    )

plt.xlabel("Federated Round")
plt.ylabel("Clean Validation Accuracy")
plt.title("FedAvg: Individual Client Validation Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/client_validation_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 4: CLIENT UPDATE NORMS
# ============================================================

plt.figure(figsize=(10, 6))

for client_id in range(1, NUM_CLIENTS + 1):

    column = f"client_{client_id}_update_norm"

    plt.plot(
        range(1, len(fedavg_history) + 1),

        [
            row[column]
            for row in fedavg_history
        ],

        marker="o",
        label=f"Client {client_id}"
    )

plt.xlabel("Federated Round")
plt.ylabel("Update Norm")
plt.title("FedAvg: Client Update Norms")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/client_update_norms.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 5: CLIENT-GLOBAL COSINE SIMILARITY
# ============================================================

plt.figure(figsize=(10, 6))

for client_id in range(1, NUM_CLIENTS + 1):

    column = f"client_{client_id}_cosine"

    plt.plot(
        range(1, len(fedavg_history) + 1),

        [
            row[column]
            for row in fedavg_history
        ],

        marker="o",
        label=f"Client {client_id}"
    )

plt.xlabel("Federated Round")
plt.ylabel("Cosine Similarity")
plt.title("FedAvg: Client-Global Update Alignment")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/client_cosine_similarity.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEDERATED LEARNING EXPERIMENTS COMPLETED")
print("=" * 60)

print("\nResults:")

print(
    results_df.to_string(index=False)
)

print("\nModels saved in results/")
print("Metrics saved in results/")
print("Visualizations saved in plots/")

print("\nEXPERIMENT PIPELINE FINISHED SUCCESSFULLY!")
