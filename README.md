# Federated Learning Benchmark on PathMNIST

## 1. Project Overview

This project implements and evaluates two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, using the PathMNIST medical imaging dataset from MedMNIST.

The objective is to investigate collaborative model training across multiple distributed clients while keeping their training data local. The experiments examine the effects of non-IID data distributions and label noise on global and client-level model performance.

The project also investigates client update behaviour through update norms and cosine similarity, with experiment tracking using Weights & Biases (W&B).

## 2. Objectives

- Implement and compare FedAvg and FedProx.
- Simulate federated learning across five clients.
- Study non-IID data distribution using Dirichlet partitioning.
- Apply 40% label noise to one client.
- Evaluate global test accuracy and loss.
- Track individual client accuracy and loss.
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
│   ├── client_accuracy.png
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
| Dirichlet concentration parameter (α) | 0.3 |
| Label noise | 40% applied to one client |
| Federated algorithms | FedAvg and FedProx |
| Communication rounds | 2 |
| FedProx proximal coefficient (μ) | 0.1 |
| Experiment tracking | Weights & Biases |
| Evaluation | Global testing and client-level validation |

## 6. Implementation Details and Modifications

The project includes the following implementation components:

- **PathMNIST dataset integration:** Added dataset loading and preprocessing using MedMNIST.
- **Non-IID data partitioning:** Implemented client data distribution using a Dirichlet distribution with α = 0.3.
- **Label noise:** Incorporated 40% label noise into one client's training data to investigate the effect of noisy labels.
- **FedAvg:** Implemented federated averaging for aggregating local client model updates.
- **FedProx:** Implemented proximal regularization to account for differences in client data distributions.
- **Client-level evaluation:** Added individual client training and validation performance tracking.
- **Global evaluation:** Added global test accuracy and loss evaluation.
- **Update analysis:** Added client update norm and cosine similarity calculations.
- **Experiment tracking:** Integrated Weights & Biases to record experiment metrics across communication rounds.
- **Visualization:** Generated plots to compare global and client-level performance.

## 7. Experimental Results

The following results are from the committed two-round experiment output files.

### FedAvg vs FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | 37.77% | 35.21% |
| Global test loss | 1.8813 | 2.0590 |
| Mean client validation accuracy | 26.63% | 24.88% |
| Global update norm | 1.1970 | 0.5585 |

### Observations

- FedAvg achieved higher global test accuracy than FedProx in the reported experiment.
- FedProx produced a smaller global update norm.
- Client-level validation performance varied across the five clients.
- The experiments provide an initial comparison of both algorithms under non-IID data and label-noise conditions.
- The results are preliminary and based on two communication rounds. Additional rounds and multiple random seeds would be needed for stronger conclusions.

## 8. Generated Outputs

The project generates the following outputs:

- Global test accuracy and loss metrics.
- Individual client training and validation accuracy.
- Individual client loss metrics.
- Client update norms.
- Client cosine similarity metrics.
- CSV and JSON experiment result files.
- Saved FedAvg and FedProx model checkpoints.
- Plots showing global and client-level performance.

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

### Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 10. Running the Experiments

Execute the main experiment script:

```bash
python run_experiment.py
```

Configure experiment tracking:

```bash
python setup_wandb.py
```

The experiment pipeline executes the configured federated learning experiments, evaluates global and client-level performance, saves experiment results, and generates visualizations.

## 11. Experiment Tracking with Weights & Biases

Experiments are tracked using Weights & Biases.

**W&B Project:**

https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist

The experiment tracking includes:

- Global test accuracy and loss.
- Individual client accuracy and loss.
- Client update norms.
- Client cosine similarity.
- Communication round information.
- Separate experiment runs for FedAvg and FedProx.

The project uses a single W&B project to organize and compare the federated learning experiments.

## 12. Testing and Verification

The project was verified using automated tests and environment checks.

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

**Reported verification results:**

- Python compilation completed successfully.
- No broken package requirements were reported.
- 9 automated tests passed.

## 13. Limitations and Future Work

- The current experiments use only two communication rounds.
- Additional random seeds are needed to evaluate reproducibility.
- More extensive experiments could investigate different non-IID distributions and label-noise levels.
- Further analysis could examine convergence and client-level performance over additional communication rounds.
- Additional experiments could investigate the impact of different FedProx proximal coefficients.

## 14. Conclusion

This project implements and compares FedAvg and FedProx for federated learning on the PathMNIST dataset using five clients with non-IID data distribution and label noise.

The implementation supports distributed client training, model aggregation, noisy-label experimentation, global evaluation, client-level validation, experiment tracking, and visualization.

The preliminary results demonstrate differences in global accuracy, loss, and update norms between the two algorithms. The project provides an experimental foundation for studying federated optimization under heterogeneous client data conditions.
