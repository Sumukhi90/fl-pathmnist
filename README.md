# Federated Learning Benchmark on PathMNIST

## 1. Project Overview

This project implements and evaluates two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, using the PathMNIST medical imaging dataset from MedMNIST.

The objective is to investigate collaborative model training across distributed clients while keeping their training data local. The experiments examine non-IID data distributions and label noise, evaluating their effects on global and client-level model performance.

The project also analyzes client update behavior through update norms and cosine similarity, with experiment tracking using Weights & Biases (W&B).

## 2. Objectives

- Implement and compare FedAvg and FedProx.
- Simulate federated learning across five clients.
- Study non-IID data distribution using Dirichlet partitioning.
- Apply 40% label noise to one client's training data.
- Evaluate global test accuracy and loss.
- Track individual client accuracy and loss.
- Analyze client update norms and cosine similarity.
- Visualize global and client-level performance.
- Evaluate reproducibility across three random seeds.
- Track experimental runs using W&B.

## 3. Technologies Used

- Python
- PyTorch
- Torchvision
- MedMNIST
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Weights & Biases
- Pytest

## 4. Project Structure

```text
fl-pathmnist/
├── configs/
├── data/
├── plots/
├── results/
│   ├── all_runs_metrics.csv
│   ├── final_summary_across_seeds.json
│   ├── mean_std_across_seeds.csv
│   ├── fedavg_seed42_metrics.csv
│   ├── fedavg_seed43_metrics.csv
│   ├── fedavg_seed44_metrics.csv
│   ├── fedprox_mu0.1_seed42_metrics.csv
│   ├── fedprox_mu0.1_seed43_metrics.csv
│   └── fedprox_mu0.1_seed44_metrics.csv
├── src/
│   ├── aggregation.py
│   ├── client.py
│   ├── data.py
│   ├── evaluate.py
│   ├── fedprox.py
│   ├── models.py
│   ├── noise.py
│   ├── partition.py
│   ├── server.py
│   ├── tracking.py
│   └── utils.py
├── tests/
│   └── test_components.py
├── run_experiment.py
├── setup_wandb.py
├── requirements.txt
└── README.md
```

The results directory also contains per-run JSON files and model checkpoints.

## 5. Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | PathMNIST (MedMNIST) |
| Framework | PyTorch |
| Number of clients | 5 |
| Data distribution | Non-IID Dirichlet partition |
| Dirichlet concentration (α) | 0.3 |
| Label noise | 40% on one client's training data |
| Algorithms | FedAvg and FedProx |
| Communication rounds | 30 |
| Random seeds | 42, 43, 44 |
| FedProx coefficient (μ) | 0.1 |
| Experiment tracking | Weights & Biases |
| Evaluation | Global test and client-level validation |

A total of six experiments were conducted: three FedAvg runs and three FedProx runs, using the same three random seeds.

## 6. Implementation Details

The implementation includes:

- **PathMNIST integration:** Dataset loading and preprocessing using MedMNIST.
- **Non-IID partitioning:** Distribution of client data using a Dirichlet distribution with α = 0.3.
- **Label noise:** 40% label noise introduced into one client's training labels.
- **FedAvg:** Aggregation of locally trained client models using federated averaging.
- **FedProx:** Proximal regularization to constrain local model updates relative to the global model.
- **Client-level evaluation:** Tracking of individual client training and validation metrics.
- **Global evaluation:** Evaluation using global test accuracy and loss.
- **Update analysis:** Calculation of client update norms and cosine similarity.
- **Experiment tracking:** Recording experiment metrics across communication rounds using W&B.
- **Visualization:** Generation of plots for global and client-level performance.

## 7. Experimental Results

The following results represent the mean and standard deviation across three independent runs (seeds 42, 43, and 44), evaluated after 30 communication rounds.

### FedAvg vs. FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | **70.43% ± 3.67%** | 68.43% ± 4.45% |
| Global test loss | **0.9545 ± 0.0892** | 1.0362 ± 0.0922 |
| Mean client validation accuracy | **66.01% ± 3.16%** | 61.22% ± 2.93% |
| Global update norm | 0.3211 ± 0.0498 | 0.2285 ± 0.0307 |

### Noisy Client Analysis

Client 5 was assigned 40% label noise in its training data. Its validation accuracy was compared with the mean validation accuracy of the clean clients.

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Noisy client validation accuracy | 61.91% ± 5.73% | 54.45% ± 3.59% |
| Mean clean-client validation accuracy | 67.03% ± 4.00% | 62.91% ± 3.74% |
| Noisy-minus-clean accuracy gap | -5.12 percentage points | -8.46 percentage points |

### Observations

1. **Global performance:** FedAvg achieved higher mean test accuracy than FedProx by approximately 2.01 percentage points and also produced a lower mean test loss.

2. **Client-level performance:** FedAvg achieved higher mean client validation accuracy in this experimental configuration.

3. **Noisy-client behavior:** The noisy client's mean validation accuracy was lower than the clean-client average for both algorithms.

4. **Effect of label noise:** The average noisy-minus-clean accuracy gap was approximately -5.12 percentage points for FedAvg and -8.46 percentage points for FedProx.

5. **Update behavior:** FedProx had a smaller mean global update norm than FedAvg, indicating smaller measured global updates in these experiments.

6. **Variability:** Results varied across random seeds, highlighting the importance of evaluating federated learning algorithms across multiple runs.

These results are specific to the selected dataset, partitioning configuration, noise level, model, and optimization settings. They do not establish that one algorithm will outperform the other under all federated learning conditions.

## 8. Generated Outputs

The experiment pipeline generates:

- Global test accuracy and loss.
- Individual client training and validation accuracy.
- Individual client training and validation loss.
- Client update norms and cosine similarity.
- Per-run CSV and JSON metrics.
- Combined metrics across all six runs
