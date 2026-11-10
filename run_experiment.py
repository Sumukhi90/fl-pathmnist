import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.tracking import start_run, log_metrics, finish_run
from src.data import load_pathmnist, create_client_loader
from src.partition import dirichlet_partition
from src.noise import add_label_noise
from src.models import PathMNISTCNN
from src.client import train_local_model
from src.aggregation import fedavg_aggregate
from src.evaluate import evaluate_model


# CONFIGURATION
SEEDS = [42, 43, 44]
NUM_CLIENTS = 5
ALPHA = 0.3
NOISE_RATE = 0.4
NOISY_CLIENT = 4
CLIENT_VALIDATION_FRACTION = 0.20

ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.01
FEDPROX_MU = 0.1
NUM_CLASSES = 9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_PROJECT = "indiaai-federated-pathmnist"

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# REPRODUCIBILITY
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# PREPARE DATA FOR EACH SEED
def prepare_seed_data(train_dataset, train_labels, test_dataset, seed):

    client_indices = dirichlet_partition(
        train_labels,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA,
        seed=seed
    )

    client_loaders = []
    client_generators = []
    validation_loaders = []
    client_sizes = []

    for client_id, indices in enumerate(client_indices):

        split_rng = np.random.default_rng(seed + 1000 + client_id)

        shuffled_indices = np.asarray(indices).copy()
        split_rng.shuffle(shuffled_indices)

        split_point = int(
            len(shuffled_indices) * (1 - CLIENT_VALIDATION_FRACTION)
        )

        train_indices = shuffled_indices[:split_point]
        validation_indices = shuffled_indices[split_point:]

        client_labels = train_labels[train_indices].copy()
        noisy_count = 0

        # Apply label noise only to Client 5.
        if client_id == NOISY_CLIENT:

            client_labels, noisy_indices = add_label_noise(
                client_labels,
                noise_rate=NOISE_RATE,
                num_classes=NUM_CLASSES,
                seed=seed + 5000
            )

            noisy_count = len(noisy_indices)

        generator = torch.Generator()
        generator.manual_seed(seed + 2000 + client_id)

        train_loader = create_client_loader(
            train_dataset,
            train_indices,
            labels=client_labels,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator
        )

        validation_loader = create_client_loader(
            train_dataset,
            validation_indices,
            labels=None,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        client_loaders.append(train_loader)
        client_generators.append(generator)
        validation_loaders.append(validation_loader)
        client_sizes.append(len(train_indices))

        print(
            f"Seed {seed} | Client {client_id + 1} | "
            f"Training: {len(train_indices)} | "
            f"Validation: {len(validation_indices)} | "
            f"Noisy labels: {noisy_count}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return {
        "client_loaders": client_loaders,
        "client_generators": client_generators,
        "validation_loaders": validation_loaders,
        "client_sizes": client_sizes,
        "test_loader": test_loader
    }


# MODEL UPDATE METRICS
def get_update_vector(new_state, old_state):

    return torch.cat([
        (
            new_state[key].float() -
            old_state[key].float()
        ).reshape(-1)

        for key in old_state
    ])


def calculate_update_metrics(client_updates, global_update):

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


# RUN ONE EXPERIMENT
def run_experiment(method, mu, seed, seed_data):

    run_name = (
        f"fedavg_seed{seed}"
        if method == "FedAvg"
        else f"fedprox_mu{mu}_seed{seed}"
    )

    print("\n" + "=" * 60)
    print(f"STARTING {run_name}")
    print("=" * 60)

    seed_everything(seed)

    for client_id, generator in enumerate(
        seed_data["client_generators"]
    ):
        generator.manual_seed(seed + 2000 + client_id)

    global_model = PathMNISTCNN().to(DEVICE)

    history = []

    wandb_run = start_run(
        project=WANDB_PROJECT,
        run_name=run_name,
        config={
            "method": method,
            "seed": seed,
            "num_clients": NUM_CLIENTS,
            "alpha": ALPHA,
            "noise_rate": NOISE_RATE,
            "noisy_client": NOISY_CLIENT + 1,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "fedprox_mu": mu,
            "device": str(DEVICE)
        }
    )

    try:

        for round_num in range(1, ROUNDS + 1):

            print(
                f"\n{run_name} | "
                f"Round {round_num}/{ROUNDS}"
            )

            old_state = {
                key: value.detach().clone()
                for key, value in global_model.state_dict().items()
            }

            client_weights = []
            client_updates = []
            client_accuracies = []
            client_losses = []

            # LOCAL TRAINING
            for client_id, loader in enumerate(
                seed_data["client_loaders"]
            ):

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
                client_accuracies.append(float(accuracy))
                client_losses.append(float(loss))

                client_updates.append(
                    get_update_vector(weights, old_state)
                )

            # FEDERATED AGGREGATION
            aggregated_weights = fedavg_aggregate(
                client_weights,
                seed_data["client_sizes"]
            )

            global_model.load_state_dict(aggregated_weights)

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

            # GLOBAL EVALUATION
            test_metrics = evaluate_model(
                global_model,
                seed_data["test_loader"],
                device=DEVICE
            )

            # CLIENT VALIDATION
            client_validation_metrics = []

            for validation_loader in seed_data["validation_loaders"]:

                metrics = evaluate_model(
                    global_model,
                    validation_loader,
                    device=DEVICE
                )

                client_validation_metrics.append(metrics)

            clean_clients = [
                i for i in range(NUM_CLIENTS)
                if i != NOISY_CLIENT
            ]

            noisy_accuracy = float(
                client_validation_metrics[NOISY_CLIENT]["test_accuracy"]
            )

            noisy_loss = float(
                client_validation_metrics[NOISY_CLIENT]["test_loss"]
            )

            clean_accuracy = float(np.mean([
                client_validation_metrics[i]["test_accuracy"]
                for i in clean_clients
            ]))

            clean_loss = float(np.mean([
                client_validation_metrics[i]["test_loss"]
                for i in clean_clients
            ]))

            # ROUND METRICS
            row = {
                "method": method,
                "seed": seed,
                "round": round_num,

                "test_loss": float(test_metrics["test_loss"]),
                "test_accuracy": float(test_metrics["test_accuracy"]),

                "global_update_norm": global_update_norm,

                "mean_client_training_accuracy": float(
                    np.mean(client_accuracies)
                ),

                "mean_client_training_loss": float(
                    np.mean(client_losses)
                ),

                "mean_client_validation_accuracy": float(np.mean([
                    m["test_accuracy"]
                    for m in client_validation_metrics
                ])),

                "mean_client_validation_loss": float(np.mean([
                    m["test_loss"]
                    for m in client_validation_metrics
                ])),

                "noisy_client_validation_accuracy": noisy_accuracy,
                "noisy_client_validation_loss": noisy_loss,

                "mean_clean_client_validation_accuracy": clean_accuracy,
                "mean_clean_client_validation_loss": clean_loss,

                "noisy_minus_clean_accuracy": noisy_accuracy - clean_accuracy,
                "noisy_minus_clean_loss": noisy_loss - clean_loss
            }

            # INDIVIDUAL CLIENT METRICS
            for client_id in range(NUM_CLIENTS):

                index = client_id + 1

                validation = client_validation_metrics[client_id]
                update = update_metrics[client_id]

                row[f"client_{index}_training_loss"] = (
                    client_losses[client_id]
                )

                row[f"client_{index}_training_accuracy"] = (
                    client_accuracies[client_id]
                )

                row[f"client_{index}_validation_loss"] = float(
                    validation["test_loss"]
                )

                row[f"client_{index}_validation_accuracy"] = float(
                    validation["test_accuracy"]
                )

                row[f"client_{index}_update_norm"] = (
                    update["update_norm"]
                )

                row[f"client_{index}_cosine"] = (
                    update["cosine_similarity"]
                )

            history.append(row)

            # LIVE W&B LOGGING
            log_metrics(
                row,
                step=round_num,
                run=wandb_run
            )

            print(
                f"Global accuracy: {row['test_accuracy']:.4f} | "
                f"Global loss: {row['test_loss']:.4f}"
            )

            print(
                f"Noisy Client 5 validation accuracy: "
                f"{noisy_accuracy:.4f}"
            )

            print(
                f"Mean clean-client validation accuracy: "
                f"{clean_accuracy:.4f}"
            )

        # SAVE INDIVIDUAL RUN RESULTS
        torch.save(
            global_model.state_dict(),
            RESULTS_DIR / f"{run_name}_model.pth"
        )

        pd.DataFrame(history).to_csv(
            RESULTS_DIR / f"{run_name}_metrics.csv",
            index=False
        )

        with open(
            RESULTS_DIR / f"{run_name}_metrics.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(history, file, indent=2)

        print(f"\nCompleted and saved: {run_name}")

    finally:
        finish_run(wandb_run)

    return history


# AGGREGATE RESULTS
def create_aggregate_results(all_histories):

    all_rows = [
        row
        for history in all_histories.values()
        for row in history
    ]

    all_df = pd.DataFrame(all_rows)

    all_df.to_csv(
        RESULTS_DIR / "all_runs_metrics.csv",
        index=False
    )

    metrics = [
        "test_accuracy",
        "test_loss",
        "mean_client_validation_accuracy",
        "mean_client_validation_loss",
        "global_update_norm",
        "noisy_client_validation_accuracy",
        "noisy_client_validation_loss",
        "mean_clean_client_validation_accuracy",
        "mean_clean_client_validation_loss",
        "noisy_minus_clean_accuracy",
        "noisy_minus_clean_loss"
    ]

    metrics += [
        f"client_{client_id}_update_norm"
        for client_id in range(1, NUM_CLIENTS + 1)
    ]

    metrics += [
        f"client_{client_id}_cosine"
        for client_id in range(1, NUM_CLIENTS + 1)
    ]

    grouped = all_df.groupby(
        ["method", "round"]
    )[metrics].agg(["mean", "std"])

    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat in grouped.columns
    ]

    grouped = grouped.reset_index()

    grouped.to_csv(
        RESULTS_DIR / "mean_std_across_seeds.csv",
        index=False
    )

    final_rows = all_df[all_df["round"] == ROUNDS]

    summary = {}

    for method in ["FedAvg", "FedProx"]:

        subset = final_rows[final_rows["method"] == method]
        summary[method] = {}

        for metric in metrics:

            values = subset[metric].astype(float)

            summary[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "per_seed": {
                    str(int(row["seed"])): float(row[metric])
                    for _, row in subset.iterrows()
                }
            }

    with open(
        RESULTS_DIR / "final_summary_across_seeds.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(summary, file, indent=2)

    return grouped, summary


# PLOT RESULTS
def plot_metric(grouped, metric, ylabel, title, filename):

    plt.figure(figsize=(9, 5))

    for method in ["FedAvg", "FedProx"]:

        subset = grouped[
            grouped["method"] == method
        ].sort_values("round")

        mean = subset[f"{metric}_mean"].to_numpy(dtype=float)

        std = subset[f"{metric}_std"].fillna(0).to_numpy(dtype=float)

        rounds = subset["round"].to_numpy()

        plt.plot(
            rounds,
            mean,
            marker="o",
            markersize=3,
            label=method
        )

        plt.fill_between(
            rounds,
            mean - std,
            mean + std,
            alpha=0.15
        )

    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=300
    )

    plt.close()


# MAIN EXECUTION
def main():

    print("Device:", DEVICE)
    print("Seeds:", SEEDS)
    print("Rounds per run:", ROUNDS)
    print("Total runs:", len(SEEDS) * 2)

    print("\nLoading PathMNIST...")

    train_dataset, test_dataset = load_pathmnist()

    train_labels = np.asarray(
        train_dataset.labels
    ).reshape(-1)

    all_histories = {}

    for seed in SEEDS:

        print("\nPreparing data for seed:", seed)

        seed_everything(seed)

        seed_data = prepare_seed_data(
            train_dataset,
            train_labels,
            test_dataset,
            seed
        )

        # FedAvg
        all_histories[f"fedavg_seed{seed}"] = run_experiment(
            "FedAvg",
            mu=0.0,
            seed=seed,
            seed_data=seed_data
        )

        # FedProx
        all_histories[f"fedprox_mu{FEDPROX_MU}_seed{seed}"] = (
            run_experiment(
                "FedProx",
                mu=FEDPROX_MU,
                seed=seed,
                seed_data=seed_data
            )
        )

    grouped, summary = create_aggregate_results(all_histories)

    plot_metric(
        grouped,
        "test_accuracy",
        "Global Test Accuracy",
        "Global Accuracy: FedAvg vs FedProx",
        "global_accuracy_mean_std.png"
    )

    plot_metric(
        grouped,
        "test_loss",
        "Global Test Loss",
        "Global Loss: FedAvg vs FedProx",
        "global_loss_mean_std.png"
    )

    plot_metric(
        grouped,
        "noisy_client_validation_accuracy",
        "Client 5 Validation Accuracy",
        "Noisy Client Validation Accuracy",
        "noisy_client_accuracy_mean_std.png"
    )

    plot_metric(
        grouped,
        "mean_clean_client_validation_accuracy",
        "Clean Client Validation Accuracy",
        "Clean Client Validation Accuracy",
        "clean_clients_accuracy_mean_std.png"
    )

    plot_metric(
        grouped,
        "noisy_minus_clean_accuracy",
        "Accuracy Difference",
        "Noisy Client vs Clean Clients",
        "noisy_vs_clean_accuracy_gap.png"
    )

    plot_metric(
        grouped,
        "global_update_norm",
        "Global Update Norm",
        "Global Update Norm",
        "global_update_norm_mean_std.png"
    )

    print("\nALL EXPERIMENTS COMPLETED")

    for method, values in summary.items():

        print("\n", method)

        for metric in [
            "test_accuracy",
            "test_loss",
            "noisy_client_validation_accuracy",
            "mean_clean_client_validation_accuracy"
        ]:

            print(
                metric,
                values[metric]["mean"],
                "+/-",
                values[metric]["std"]
            )


if __name__ == "__main__":
    main()
