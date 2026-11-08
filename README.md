
# Federated Learning Benchmark on PathMNIST

## 1. Project Overview

This project implements and evaluates two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, using the PathMNIST medical imaging dataset from MedMNIST.

The objective is to study collaborative model training across multiple clients while keeping their training data decentralized. The experiments simulate non-IID client data distributions and label noise to examine their effects on global and client-level model performance.

The project also analyzes client update behaviour using update norms and cosine similarity, with experiment tracking through Weights & Biases (W&B).

## 2. Objectives

- Implement and compare FedAvg and FedProx.
- Simulate federated learning across five clients.
- Create non-IID client data partitions using Dirichlet distribution.
- Apply 40% label noise to one client's training labels.
- Evaluate global test accuracy and loss.
- Track individual client training and validation accuracy and loss.
- Analyze client update norms and cosine similarity.
- Visualize global and client-level performance.
- Track experimental runs using Weights & Biases.

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
├── plots/
│   ├── client_cosine_similarity.png
│   ├── client_training_accuracy.png
│   ├── client_update_norms.png
│   ├── client_validation_accuracy.png
│   ├── global_test_accuracy.png
│   └── global_test_loss.png
│
├── results/
│   ├── experiment_results.csv
│   ├── experiment_results.json
│   ├── fedavg_metrics.csv
│   ├── fedavg_model.pth
│   ├── fedprox_metrics.csv
│   └── fedprox_model.pth
│
├── src/
│   ├── __init__.py
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
│
├── tests/
│   └── test_components.py
│
├── run_experiment.py
├── setup_wandb.py
├── requirements.txt
└── README.md
```

## 5. Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | PathMNIST (MedMNIST) |
| Framework | PyTorch |
| Number of clients | 5 |
| Data distribution | Non-IID Dirichlet partition |
| Dirichlet concentration (α) | 0.3 |
| Label noise | 40% on Client 5 |
| Noise application | Client 5 training labels only |
| Client validation split | 20% |
| Federated algorithms | FedAvg and FedProx |
| Communication rounds | 2 |
| Local epochs | 1 |
| Batch size | 128 |
| Learning rate | 0.01 |
| FedAvg proximal coefficient | 0.0 |
| FedProx proximal coefficient (μ) | 0.1 |
| Random seed | 42 |
| Experiment tracking | Weights & Biases |

The client training data is partitioned using a Dirichlet distribution with α = 0.3. A 20% validation split is created for each client. Label noise is applied only to the training labels of Client 5, while validation labels remain clean.

## 6. Implementation Details

### FedAvg

Federated Averaging trains local client models and aggregates their model parameters using a sample-size-weighted average.

### FedProx

FedProx adds a proximal regularization term to the local training objective to limit excessive deviation of local models from the global model.

The proximal coefficient used in this experiment is μ = 0.1.

### Additional Implementation Components

- **Dataset integration:** PathMNIST loading and preprocessing using MedMNIST.
- **Client partitioning:** Non-IID data distribution using Dirichlet partitioning.
- **Label noise:** 40% noisy training labels on Client 5.
- **Client-level evaluation:** Individual client training and clean validation metrics.
- **Global evaluation:** Global test accuracy and loss after each communication round.
- **Update analysis:** Client update norms and cosine similarity with the global update.
- **Experiment tracking:** W&B logging of round-level and client-level metrics.
- **Visualization:** Plots comparing global performance and client update behaviour.

## 7. Experimental Results

The following results are from the committed two-round experiment outputs.

### FedAvg vs FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | 37.77% | 35.21% |
| Global test loss | 1.8813 | 2.0590 |
| Mean client validation accuracy | 26.63% | 24.88% |
| Global update norm | 1.1970 | 0.5585 |

### Observations

- FedAvg achieved higher global test accuracy than FedProx in the reported experiment.
- FedAvg also recorded a lower global test loss.
- FedProx produced a smaller global update norm.
- Client-level validation performance varied across the five clients.
- The results provide an initial comparison of both algorithms under non-IID data and label-noise conditions.

**Experimental scope:** These results are preliminary and are based on two communication rounds. They should not be interpreted as evidence of convergence. Additional communication rounds and multiple random seeds would be required for stronger conclusions.

## 8. Generated Outputs

The experiment pipeline generates:

- Global test accuracy and loss.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Client update norms.
- Client-global cosine similarity.
- CSV and JSON experiment result files.
- Saved FedAvg and FedProx model checkpoints.
- Global performance and client-level visualization plots.

## 9. Installation

### Clone the repository

```bash
git clone https://github.com/Sumukhi90/fl-pathmnist.git
cd fl-pathmnist
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 10. Running the Experiments

### Configure Weights & Biases

```bash
python setup_wandb.py
```

Make sure you have authenticated with your W&B account before running the experiments.

### Execute the experiment

```bash
python run_experiment.py
```

The pipeline executes the configured FedAvg and FedProx experiments, evaluates global and client-level performance, logs metrics, saves model checkpoints and generates visualizations.

## 11. Experiment Tracking with Weights & Biases

Experiments are tracked using Weights & Biases.

**W&B Project:**

https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist

The tracking setup includes:

- Global test accuracy and loss.
- Individual client training and validation accuracy.
- Individual client training and validation loss.
- Client update norms.
- Client cosine similarity.
- Communication round information.
- Separate experiment runs for FedAvg and FedProx.

Both algorithms are organized within a single W&B project for comparison and analysis.

## 12. Testing and Verification

Automated tests and environment checks were used to verify the implementation.

### Run automated tests

```bash
python -m pytest -q
```

### Check Python compilation

```bash
python -m compileall -q .
```

### Check package dependencies

```bash
python -m pip check
```

**Previously reported verification results:**

- 9 automated tests passed.
- Python compilation completed successfully.
- No broken package requirements were reported.

## 13. Limitations and Future Work

- The current experiments use two communication rounds and are preliminary.
- Additional communication rounds are needed to study convergence behaviour.
- Multiple random seeds would strengthen reproducibility.
- Further experiments could investigate different non-IID distributions and label-noise levels.
- Different FedProx proximal coefficients could be evaluated.
- More extensive client-level analysis could help understand the effects of heterogeneous data distributions.

## 14. Conclusion

This project implements and compares FedAvg and FedProx for federated learning on PathMNIST using five clients with non-IID data distributions and label noise.

The implementation supports local client training, federated aggregation, noisy-label experimentation, global testing, client-level validation, update analysis, experiment tracking and visualization.

The preliminary results demonstrate differences between FedAvg and FedProx in global accuracy, loss and update behaviour. The project provides an experimental foundation for further investigation of federated optimization under heterogeneous client data conditions.

---

**Repository:** https://github.com/Sumukhi90/fl-pathmnist

