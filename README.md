# Federated Learning Benchmark on PathMNIST: FedAvg vs. FedProx

## Project Overview

This project implements and evaluates two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, on the PathMNIST medical imaging dataset from MedMNIST.

The benchmark investigates federated model training under heterogeneous client data distributions and client-specific label noise. Five clients participate in collaborative training while keeping their local training data separate. A non-IID data distribution is created using Dirichlet partitioning, and 40% label noise is introduced into the training labels of one client.

Both algorithms are evaluated over **30 communication rounds and three random seeds (42, 43, and 44)**. The experiments track global model performance, individual client performance, and model update behaviour.

Experiment metrics are logged to **Weights & Biases (W&B)** for tracking and analysis.

## Objectives

- Implement Federated Averaging (FedAvg).
- Implement Federated Proximal (FedProx).
- Simulate federated learning with five clients.
- Create non-IID client data partitions using a Dirichlet distribution.
- Introduce 40% label noise into one client's training labels.
- Compare global test accuracy and loss.
- Track individual client training and validation metrics.
- Analyse model update norms and cosine similarity.
- Evaluate performance across three random seeds.
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

## Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | PathMNIST (MedMNIST) |
| Learning framework | PyTorch |
| Number of clients | 5 |
| Data distribution | Non-IID |
| Partitioning method | Dirichlet distribution |
| Dirichlet concentration (α) | 0.3 |
| Label noise | 40% on one client's training labels |
| Noisy client | Client 5 |
| Algorithms | FedAvg and FedProx |
| Communication rounds | 30 |
| Random seeds | 42, 43, 44 |
| FedProx proximal coefficient (μ) | 0.1 |
| Total experiment runs | 6 |
| Experiment tracking | Weights & Biases |

Three runs were conducted for each algorithm using the same set of random seeds. This resulted in six experiment runs in total: three FedAvg runs and three FedProx runs.

## Algorithms

### 1. Federated Averaging (FedAvg)

FedAvg is a federated optimization algorithm in which clients train their models locally and send their model updates to a central server. The server aggregates the local models, typically using a weighted average based on the number of training samples at each client.

### 2. Federated Proximal (FedProx)

FedProx extends FedAvg by adding a proximal regularization term to the local training objective. This term penalizes deviations from the current global model and is intended to help manage statistical and systems heterogeneity among clients.

The proximal coefficient used in these experiments is:

**μ = 0.1**

## Implementation Details

The experiment pipeline includes the following stages:

1. **Dataset preparation:** Loading and preprocessing PathMNIST using MedMNIST.
2. **Client partitioning:** Distributing training data across five clients using a Dirichlet distribution with α = 0.3.
3. **Label noise:** Introducing 40% label noise into the training labels of Client 5.
4. **Local training:** Training client models independently during each communication round.
5. **Server aggregation:** Applying FedAvg or FedProx according to the selected algorithm.
6. **Global evaluation:** Evaluating the aggregated global model on the global test dataset.
7. **Client-level evaluation:** Recording individual client training and validation accuracy and loss.
8. **Update analysis:** Tracking model update norms and cosine similarity.
9. **Experiment tracking:** Logging metrics across communication rounds to Weights & Biases.

## Experimental Results

The following results represent the **mean ± standard deviation across three runs (seeds 42, 43, and 44)** after 30 communication rounds.

### FedAvg vs. FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | 70.43% ± 3.67% | 68.43% ± 4.45% |
| Global test loss | 0.9545 ± 0.0892 | 1.0362 ± 0.0922 |
| Mean client validation accuracy | 66.01% ± 3.16% | 61.22% ± 2.93% |
| Global update norm | 0.3211 ± 0.0498 | 0.2285 ± 0.0307 |

### Noisy Client Analysis

Client 5 received 40% label noise in its training labels. Its validation performance was compared with the average validation performance of the four clean clients.

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Noisy client validation accuracy | 61.91% ± 5.73% | 54.45% ± 3.59% |
| Mean clean-client validation accuracy | 67.03% ± 4.00% | 62.91% ± 3.74% |
| Noisy-minus-clean accuracy gap | -5.12 percentage points | -8.46 percentage points |

### Client Update Norm and Cosine Similarity Analysis

Client 5's update behaviour was compared with the average behaviour of the four clean clients. The following values are averages across the recorded communication rounds and three random seeds.

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Client 5 mean update norm | 1.1682 | 0.7630 |
| Mean update norm of clean clients | 0.9613 | 0.7074 |
| Client 5 mean cosine similarity | 0.0879 | 0.1079 |
| Mean cosine similarity of clean clients | 0.4847 | 0.5410 |

The noisy client generally exhibited larger update norms and lower cosine similarity than the clean clients under both algorithms.

The higher update norms suggest that Client 5 produced larger model updates, while its lower cosine similarity indicates that its updates were less aligned with the reference updates used in the cosine-similarity analysis.

FedProx reduced the difference between the noisy and clean clients' update norms. However, the directional difference in cosine similarity remained noticeable.

These measurements provide an indication of differences in client update behaviour; they do not independently establish convergence or generalization quality.

## Results and Observations

**Global performance:** FedAvg achieved a mean global test accuracy of 70.43%, compared with 68.43% for FedProx. FedAvg also recorded a lower mean test loss in this experimental configuration.

**Client-level performance:** FedAvg achieved higher mean client validation accuracy than FedProx across the three runs.

**Noisy-client behaviour:** Client 5 showed lower validation accuracy than the average of the four clean clients under both algorithms.

**Impact of label noise:** The average noisy-minus-clean validation accuracy gap was -5.12 percentage points for FedAvg and -8.46 percentage points for FedProx.

**Model update behaviour:** FedProx recorded a smaller mean global update norm than FedAvg. At the client level, Client 5 generally showed larger update norms and lower cosine similarity than the clean clients.

**Variation across runs:** The standard deviations indicate that performance varied across random seeds, reinforcing the importance of evaluating federated learning methods across multiple runs.

**Overall observation:** FedAvg performed better on the reported global test and client validation metrics in this particular setup. FedProx produced smaller global update norms, but this did not translate into higher test accuracy or better noisy-client validation accuracy in these experiments.

These findings are specific to the dataset, partitioning, noise level, model, and optimization settings used in this benchmark. They should not be interpreted as a universal ranking of the two algorithms.

## Experiment Tracking

All six experiment runs were logged to the Weights & Biases project.

**W&B Project:** IndiaAI Federated PathMNIST Benchmark

**Project path:** `sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist`

### Run Details

| Algorithm | Random seed | Run name |
|---|---:|---|
| FedAvg | 42 | `fedavg_seed42` |
| FedProx | 42 | `fedprox_mu0.1_seed42` |
| FedAvg | 43 | `fedavg_seed43` |
| FedProx | 43 | `fedprox_mu0.1_seed43` |
| FedAvg | 44 | `fedavg_seed44` |
| FedProx | 44 | `fedprox_mu0.1_seed44` |

The tracking pipeline records global and client-level metrics across communication rounds, including accuracy, loss, update norms, and cosine similarity where implemented.

## Generated Outputs

The experiment pipeline generates:

- Global test accuracy and loss.
- Per-client training accuracy and loss.
- Per-client validation accuracy and loss.
- Client update norms and cosine similarity.
- Per-run CSV and JSON metric files.
- Combined metrics across all six runs.
- Mean and standard deviation summaries across seeds.
- Experiment plots and model checkpoints, where generated by the configured pipeline.

### Key Result Files

```text
results/all_runs_metrics.csv
results/final_summary_across_seeds.json
results/mean_std_across_seeds.csv
```

## Project Structure

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

The results directory may also contain per-run JSON files, experiment plots, and model checkpoints, depending on the configured outputs.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sumukhi90/fl-pathmnist.git
cd fl-pathmnist
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure experiment tracking

Configure your Weights & Biases account and project settings using the project's tracking configuration.

Authenticate with W&B before running experiments if live logging is required.

```bash
wandb login
```

### 5. Run the experiment

```bash
python run_experiment.py
```

The experiment script runs the configured FedAvg and FedProx experiments and generates the corresponding metrics and result files.

**Note:** The exact experiment configuration, including communication rounds, random seeds, and algorithm parameters, is determined by the project's configuration and experiment script.

## Limitations

- The benchmark uses one dataset, PathMNIST, and one model and optimization configuration.
- The experiments use five simulated clients rather than physically distributed devices.
- Label noise is applied to one client's training labels; other forms and distributions of noise are not evaluated.
- Only three random seeds are used, so the reported variability is based on a limited number of runs.
- The results do not establish that FedAvg or FedProx is universally superior.
- The benchmark does not evaluate real-world communication costs, network failures, or large-scale client participation.
- Further experiments with different datasets, client distributions, noise levels, and hyperparameters would be required to establish broader conclusions.

## Conclusion

This benchmark compares FedAvg and FedProx under non-IID client data and client-specific label noise using the PathMNIST dataset.

Across three seeds and 30 communication rounds, FedAvg achieved higher mean global test accuracy and lower test loss in the tested configuration. The noisy client performed below the clean-client average under both algorithms.

Client update analysis showed that the noisy client generally produced larger update norms and lower cosine similarity than the clean clients. FedProx produced smaller global update norms and reduced the difference in client update norms, but this did not translate into better test performance in these experiments.

The results highlight the importance of evaluating federated optimization methods using both global and client-level metrics, particularly when client data distributions and label quality differ.

## Repository

**GitHub:** https://github.com/Sumukhi90/fl-pathmnist
