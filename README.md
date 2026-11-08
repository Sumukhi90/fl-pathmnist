# Federated Learning Benchmark on PathMNIST

## Project Overview

This project implements and evaluates two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, using the PathMNIST medical imaging dataset from MedMNIST.

The objective is to investigate collaborative model training across multiple distributed clients while keeping their training data local. The experiments examine the effects of non-IID data distributions and label noise on global and client-level model performance.

## Objectives

- Implement and compare FedAvg and FedProx.
- Simulate federated learning across five clients.
- Study non-IID data distribution using a Dirichlet partition.
- Investigate the effect of label noise on client performance.
- Evaluate global test accuracy and loss.
- Track client-level training and validation performance.
- Analyze client update norms and cosine similarity.
- Record experiment metrics using Weights & Biases.

## Technologies Used

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

## Project Structure

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
├── results/
│   ├── experiment_results.csv
│   ├── experiment_results.json
│   ├── fedavg_metrics.csv
│   ├── fedavg_model.pth
│   ├── fedprox_metrics.csv
│   └── fedprox_model.pth
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
├── tests/
│   └── test_components.py
├── run_experiment.py
├── setup_wandb.py
├── requirements.txt
└── README.md
```

## Experimental Setup

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
| FedProx μ | [INSERT ACTUAL VALUE] |
| Experiment tracking | Weights & Biases |
| Evaluation | Global test performance and client-level validation |

## Experimental Results

The following results were obtained from the two-round experiments.

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
- The experiments provide an initial comparison of the two algorithms under non-IID data and label-noise conditions.
- These results are preliminary and are based on two communication rounds. More rounds and multiple random seeds would be required for stronger conclusions.

## Generated Outputs

The project generates:

- Global test accuracy and loss metrics.
- Client-level training and validation accuracy.
- Client update norms and cosine similarity metrics.
- CSV and JSON experiment result files.
- Saved FedAvg and FedProx model checkpoints.
- Plots showing global and client-level performance.

## Installation

Clone the repository:

```bash
git clone https://github.com/Sumukhi90/fl-pathmnist.git
cd fl-pathmnist
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Run the experiment script:

```bash
python run_experiment.py
```

Configure experiment tracking:

```bash
python setup_wandb.py
```

## Experiment Tracking

Experiments are tracked using Weights & Biases.

**W&B Project:**  
https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist

The project is intended to provide experiment-level tracking of federated learning performance and training metrics.

## Testing and Verification

Run the test suite:

```bash
pytest -q
```

Check Python compilation:

```bash
python -m compileall src run_experiment.py setup_wandb.py
```

Check installed dependencies:

```bash
pip check
```

## Limitations and Future Work

- The current experiments use only two communication rounds.
- Additional random seeds are needed to evaluate reproducibility.
- More extensive experiments could investigate different non-IID distributions and label-noise levels.
- Further analysis could examine convergence and client-level performance over additional communication rounds.

## Conclusion

This project implements and compares FedAvg and FedProx for federated learning on PathMNIST using five clients with non-IID data and label noise.

The preliminary results show differences in global accuracy, loss, and update norms between the two algorithms. The implementation provides a starting point for further investigation of federated optimization under heterogeneous client data conditions.
