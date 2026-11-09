# Federated Learning Benchmark on PathMNIST

A comparative implementation of Federated Averaging (FedAvg) and Federated Proximal (FedProx) for medical image classification using PathMNIST, with non-IID client distributions, noisy labels, client-level evaluation, and experiment tracking.

## Project Links

- GitHub Repository: [fl-pathmnist](https://github.com/Sumukhi90/fl-pathmnist)
- Experiment Dashboard: [View Weights & Biases Results](https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist/overview)

The experiment dashboard provides visibility into global and client-level performance metrics, federated optimization behaviour, and experiment configurations.

---

## 1. Project Overview

This project implements and compares two federated learning algorithms, FedAvg and FedProx, using the PathMNIST dataset from MedMNIST.

The system simulates collaborative model training across five clients. Each client performs local training, while model updates are aggregated to produce a global model without sharing raw training data.

The experiments investigate the impact of heterogeneous (non-IID) data distributions and noisy labels on federated learning performance.

The implementation also tracks global and client-level metrics, model update norms, and cosine similarities using Weights & Biases (W&B).

## 2. Objectives

- Implement FedAvg and FedProx.
- Simulate federated learning across five clients.
- Generate non-IID client data partitions using Dirichlet distribution.
- Introduce 40% label noise to one client's training labels.
- Evaluate global test accuracy and loss.
- Monitor individual client training and validation metrics.
- Analyse client update norms and cosine similarities.
- Visualise global and client-level model performance.
- Track and compare experiments using Weights & Biases.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Torchvision |
| Dataset | MedMNIST, PathMNIST |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib |
| Experiment Tracking | Weights & Biases |
| Testing | Pytest |

## 4. Project Architecture

The implementation is organised into modular components:

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

The data directory is generated automatically when the experiment script runs.

## 5. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Dataset | PathMNIST (MedMNIST) |
| Framework | PyTorch |
| Number of clients | 5 |
| Data distribution | Non-IID Dirichlet |
| Dirichlet concentration (α) | 0.3 |
| Label noise | 40% on Client 5 |
| Noise application | Client 5 training labels only |
| Client validation split | 20% |
| Algorithms | FedAvg and FedProx |
| Communication rounds | 2 |
| Local epochs | 1 |
| Batch size | 128 |
| Learning rate | 0.01 |
| FedProx coefficient (μ) | 0.1 |
| Random seed | 42 |
| Compute device | CUDA if available, otherwise CPU |

Data handling: Label noise is applied only to Client 5's training labels. Validation data retains its original clean labels.

## 6. Implementation Details

### PathMNIST Dataset

Uses the PathMNIST dataset for nine-class histopathology image classification.

### Non-IID Data Partitioning

Distributes training samples across five clients using Dirichlet partitioning (α = 0.3), simulating heterogeneous data distributions.

### Noisy Label Experimentation

Applies a 40% label noise rate exclusively to Client 5's training labels to investigate the effect of noisy local data on global model performance.

### Federated Averaging (FedAvg)

Implements sample-size-weighted aggregation of locally trained client model parameters to update the global model.

### Federated Proximal (FedProx)

Introduces a proximal regularisation term during local training to constrain deviations from the global model. The proximal coefficient is set to μ = 0.1.

### Evaluation and Model Update Analysis

Evaluates global test performance and individual client validation performance. Calculates client update norms and cosine similarities to analyse differences in model update behaviour.

### Experiment Tracking

Uses Weights & Biases to record:

- Global test accuracy and loss.
- Client-level training and validation metrics.
- Client update norms.
- Client cosine similarities.
- Communication round information.
- Experiment configurations.

## 7. Experimental Results

The following results are from the committed two-round experiment.

### FedAvg vs. FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | 37.77% | 35.21% |
| Global test loss | 1.8813 | 2.0590 |
| Mean client validation accuracy | 26.63% | 24.88% |
| Global update norm | 1.1970 | 0.5585 |

### Key Observations

- FedAvg achieved higher global test accuracy than FedProx in this experiment.
- FedProx produced a smaller global update norm.
- Client-level validation performance varied across the five clients.
- Both algorithms were evaluated under non-IID data and label-noise conditions.

Note: These are preliminary results based on only two communication rounds. They should not be interpreted as evidence of convergence or definitive superiority of either algorithm.

## 8. Visualisations and Generated Outputs

The experiment pipeline generates:

- Global test accuracy and loss plots.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Client update norm visualisations.
- Client cosine similarity analysis.
- Combined CSV and JSON experiment results.
- Separate FedAvg and FedProx metric files.
- Saved model checkpoints.

## 9. Installation and Setup

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

Configure Weights & Biases authentication:

```bash
python setup_wandb.py
```

Run the federated learning experiments:

```bash
python run_experiment.py
```

The experiment pipeline runs the configured FedAvg and FedProx experiments, performs global and client-level evaluation, logs metrics, saves results and checkpoints, and generates visualisations.

The first execution may download the PathMNIST dataset through MedMNIST.

## 11. Experiment Tracking and Monitoring

Explore the complete experiment dashboard:

[Open PathMNIST Federated Learning Dashboard](https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist/overview)

The dashboard tracks:

- Global test accuracy and loss.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Client update norms.
- Client cosine similarities.
- Communication rounds.
- Experiment configurations.

Separate experiment runs are maintained for FedAvg and FedProx.

## 12. Testing and Verification

The implementation was checked using automated tests and Python environment validation.

### Run automated tests

```bash
python -m pytest -q
```

Previously reported result: 9 automated tests passed.

### Check Python compilation

```bash
python -m compileall -q .
```

### Check package dependencies

```bash
python -m pip check
```

Previously reported verification:

- 9 automated tests passed.
- Python compilation completed successfully.
- No broken package requirements were reported.

## 13. Limitations and Future Work

The current implementation provides a preliminary experimental comparison.

Potential improvements include:

- Increasing communication rounds to investigate convergence.
- Running experiments with multiple random seeds.
- Evaluating different Dirichlet concentration parameters.
- Investigating different label-noise rates.
- Exploring different FedProx proximal coefficients.
- Analysing client-level performance over longer training periods.

## 14. Conclusion

This project implements and compares FedAvg and FedProx for PathMNIST medical image classification across five simulated clients under non-IID data distributions and noisy-label conditions.

It brings together client-side training, federated aggregation, noisy-label experimentation, global evaluation, client-level validation, model update analysis, automated testing, experiment tracking, and visualisation.

The preliminary results demonstrate differences between FedAvg and FedProx in global accuracy, loss, and model update norms, providing an experimental foundation for further investigation of federated optimisation under heterogeneous client data.

---

Project Repository: [Sumukhi90/fl-pathmnist](https://github.com/Sumukhi90/fl-pathmnist)

Experiment Tracking: [Weights & Biases Dashboard](https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist/overview)
