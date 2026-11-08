
# Federated Learning Benchmark on PathMNIST

## 1. Project Overview

This project implements and compares two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, using the PathMNIST dataset from MedMNIST.

The experiments simulate collaborative model training across five clients. Each client trains locally, and model updates are aggregated to produce a global model. The experiments investigate non-IID data distributions and the effect of noisy labels on federated learning performance.

The project also tracks global and client-level metrics, client update norms, and cosine similarities using Weights & Biases (W&B).

## 2. Objectives

- Implement FedAvg and FedProx.
- Simulate federated learning with five clients.
- Create non-IID client data partitions using Dirichlet distribution.
- Apply 40% label noise to one client's training labels.
- Evaluate global test accuracy and loss.
- Track individual client training and validation accuracy and loss.
- Analyze client update norms and cosine similarities.
- Visualize global and client-level performance.
- Track experiments using Weights & Biases.

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

The `data/` directory is created automatically when the experiment script runs. Generated files are listed above as expected experiment outputs.

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
| FedProx proximal coefficient (μ) | 0.1 |
| Random seed | 42 |
| Device | Automatically selected: CUDA if available, otherwise CPU |

The validation data retains its original clean labels. Label noise is applied only to Client 5's training labels.

## 6. Implementation Details

### PathMNIST Dataset

The project uses PathMNIST from MedMNIST for nine-class histopathology image classification.

### Non-IID Data Partitioning

The training data is distributed among five clients using Dirichlet partitioning with α = 0.3 to simulate heterogeneous client data.

### Label Noise

A 40% label noise rate is applied to Client 5's training labels. The other clients retain their original training labels.

### FedAvg

FedAvg aggregates locally trained client model parameters using sample-size-weighted averaging.

### FedProx

FedProx incorporates a proximal regularization term during local training to limit excessive deviation from the global model. The proximal coefficient is μ = 0.1.

### Evaluation and Update Analysis

The implementation evaluates global test performance and individual client validation performance. It also calculates client update norms and cosine similarities to analyze model update behaviour.

### Experiment Tracking

Weights & Biases records global and client-level metrics across communication rounds, along with update analysis metrics and experiment configuration.

## 7. Experimental Results

The following values are from the committed two-round experiment results.

### FedAvg vs. FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | 37.77% | 35.21% |
| Global test loss | 1.8813 | 2.0590 |
| Mean client validation accuracy | 26.63% | 24.88% |
| Global update norm | 1.1970 | 0.5585 |

### Observations

- FedAvg achieved higher global test accuracy than FedProx in this experiment.
- FedProx produced a smaller global update norm.
- Client-level validation performance varied across the five clients.
- Both algorithms were evaluated under non-IID data and label-noise conditions.
- These results represent a preliminary comparison based on two communication rounds.

The results should not be interpreted as evidence of convergence or definitive superiority of either algorithm.

## 8. Generated Outputs

The experiment pipeline generates:

- Global test accuracy and loss metrics.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Client update norms.
- Client cosine similarity metrics.
- Combined CSV and JSON experiment results.
- Separate FedAvg and FedProx metrics files.
- Saved FedAvg and FedProx model checkpoints.
- Plots for global and client-level performance.

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

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 10. Running the Experiments

Configure Weights & Biases authentication before running the experiment.

```bash
python setup_wandb.py
```

Run the federated learning experiments:

```bash
python run_experiment.py
```

The script runs the configured FedAvg and FedProx experiments, evaluates the global and client-level models, logs metrics, saves results and checkpoints, and generates visualizations.

The first execution may download the PathMNIST dataset through MedMNIST.

## 11. Experiment Tracking with Weights & Biases

The project uses a single W&B project to track and compare FedAvg and FedProx experiments.

**W&B Project:**

https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist

The tracked metrics include:

- Global test accuracy and loss.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Client update norms.
- Client cosine similarities.
- Communication round information.
- Experiment configuration.

Separate runs are maintained for FedAvg and FedProx.

## 12. Testing and Verification

The project was checked using automated tests and Python environment checks.

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

- The current experiment uses two communication rounds and is preliminary.
- Additional communication rounds are needed to study convergence.
- Multiple random seeds would provide stronger evidence of reproducibility.
- Further experiments could investigate different Dirichlet concentration parameters and label-noise levels.
- Different FedProx proximal coefficients could be evaluated.
- Additional analysis could investigate client-level performance over longer training runs.

## 14. Conclusion

This project implements and compares FedAvg and FedProx on the PathMNIST dataset using five clients with non-IID data distributions and label noise.

The implementation includes client-side training, federated aggregation, noisy-label experimentation, global evaluation, client-level validation, update analysis, experiment tracking, and visualization.

The preliminary results demonstrate differences between FedAvg and FedProx in global accuracy, loss, and update norms. The project provides an experimental foundation for studying federated optimization under heterogeneous client data conditions.


