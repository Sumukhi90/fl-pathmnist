# Federated Learning Benchmark on PathMNIST

A comparative implementation of Federated Averaging (FedAvg) and Federated Proximal (FedProx) for medical image classification using PathMNIST, featuring non-IID client distributions, noisy labels, client-level evaluation, model update analysis, and experiment tracking.

## Project Links

- **GitHub Repository:** https://github.com/Sumukhi90/fl-pathmnist
- **Public Weights & Biases Dashboard:** https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist/overview

The W&B dashboard provides experiment configurations, global and client-level performance metrics, model update norms, cosine similarity measurements, and round-wise visualisations.

---

## 1. Project Overview

This project implements and compares two federated learning algorithms, Federated Averaging (FedAvg) and Federated Proximal (FedProx), using the PathMNIST dataset from MedMNIST.

The system simulates collaborative model training across five clients. Each client performs local training using its assigned dataset, while model updates are aggregated to produce a global model without sharing raw training images.

The experiments investigate the effects of heterogeneous (non-IID) data distributions and noisy labels on federated learning performance.

The implementation evaluates global and client-level performance and analyses model update norms and cosine similarities using Weights & Biases (W&B).

## 2. Objectives

- Implement and compare FedAvg and FedProx.
- Simulate federated learning across five clients.
- Generate non-IID client data partitions using Dirichlet distribution.
- Introduce 40% label noise into one client's training labels.
- Evaluate global test accuracy and loss.
- Monitor individual client training and validation performance.
- Analyse global and client-level update norms.
- Calculate cosine similarity values to study model update behaviour.
- Visualise global and client-level performance.
- Track experiments using a public Weights & Biases project.

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
│
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

The dataset directory is generated automatically when the experiment script runs.

## 5. Experimental Configuration

The principal experimental setup uses the following configuration:

| Parameter | Configuration |
|---|---|
| Dataset | PathMNIST (MedMNIST) |
| Framework | PyTorch |
| Number of clients | 5 |
| Data distribution | Non-IID Dirichlet |
| Dirichlet concentration (α) | 0.3 |
| Label noise | 40% |
| Noisy client | Client 5 |
| Noise application | Client 5 training labels only |
| Client validation fraction | 20% |
| Algorithms | FedAvg and FedProx |
| Communication rounds | 2 |
| Local epochs | 1 |
| Batch size | 128 |
| Learning rate | 0.01 |
| FedProx coefficient (μ) | 0.1 for the corresponding FedProx run |
| Random seed | 42 for the verified seed-42 runs |

**Verified FedProx configuration:** The `fedprox_mu0.1_seed42` W&B run records α = 0.3, five clients, 40% label noise on Client 5, two communication rounds, one local epoch, batch size 128, learning rate 0.01, μ = 0.1, seed 42 and CPU execution.

**Data handling:** Label noise is applied exclusively to Client 5's training labels. Validation data retains its original clean labels.

The current comparison uses two communication rounds and should be interpreted as a preliminary experiment rather than a convergence study.

## 6. Implementation Details

### 6.1 PathMNIST Dataset

The project uses PathMNIST from MedMNIST for nine-class histopathology image classification.

### 6.2 Non-IID Data Partitioning

Training samples are distributed across five clients using Dirichlet partitioning with a concentration parameter of α = 0.3.

This creates heterogeneous client data distributions to simulate differences in locally available data.

### 6.3 Noisy Label Experimentation

A 40% label noise rate is introduced exclusively into Client 5's training labels.

This experiment investigates how incorrect local labels affect client-level learning and global model performance.

### 6.4 Federated Averaging (FedAvg)

FedAvg performs local training on participating clients and aggregates their model parameters using sample-size-weighted averaging to update the global model.

### 6.5 Federated Proximal (FedProx)

FedProx introduces a proximal regularisation term during local training to constrain deviations from the global model.

The proximal coefficient is set to μ = 0.1 for the corresponding FedProx experiment.

### 6.6 Evaluation and Model Update Analysis

The implementation evaluates:

- Global test accuracy and loss.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Global model update norms.
- Individual client update norms.
- Cosine similarity values.

These metrics help analyse differences in learning behaviour across heterogeneous clients.

### 6.7 Experiment Tracking

Weights & Biases is used to record experiment configurations, performance metrics and model update statistics.

The tracked metrics include:

- Global test accuracy and loss.
- Client-level training accuracy and loss.
- Client-level validation accuracy and loss.
- Global and client update norms.
- Client cosine similarity values.
- Communication round information.

## 7. Experimental Results

The following results represent the recorded two-round FedAvg and FedProx comparison.

### 7.1 FedAvg vs. FedProx

| Metric | FedAvg | FedProx |
|---|---:|---:|
| Global test accuracy | 37.77% | 35.21% |
| Global test loss | 1.8813 | 2.0590 |
| Mean client validation accuracy | 26.63% | 24.88% |
| Global update norm | 1.1970 | 0.5585 |

### 7.2 Key Observations

- FedAvg achieved higher global test accuracy than FedProx in the recorded comparison.
- FedProx produced a smaller global update norm.
- Client-level validation performance varied across the five clients.
- Both algorithms were evaluated under non-IID data and label-noise conditions.
- Global and client-level update statistics provide additional insight into optimisation behaviour.

**Interpretation:** These results are preliminary and based on two communication rounds. They should not be interpreted as evidence of convergence or definitive superiority of either algorithm.

## 8. Visualisations and Generated Outputs

The experiment pipeline generates the following outputs:

- Global test accuracy plot.
- Global test loss plot.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Global and client update norm visualisations.
- Client cosine similarity analysis.
- Combined CSV and JSON experiment results.
- Separate FedAvg and FedProx metric files.
- Saved model checkpoints.

The corresponding plots are available in the `plots/` directory.

## 9. Installation and Setup

### 9.1 Clone the repository

```bash
git clone https://github.com/Sumukhi90/fl-pathmnist.git
cd fl-pathmnist
```

### 9.2 Create a virtual environment

```bash
python -m venv .venv
```

### 9.3 Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 9.4 Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 10. Running the Experiments

### 10.1 Configure Weights & Biases

```bash
python setup_wandb.py
```

Complete the authentication process if prompted.

### 10.2 Run the experiments

```bash
python run_experiment.py
```

The experiment pipeline executes the configured FedAvg and FedProx experiments, performs global and client-level evaluation, logs metrics to W&B, saves results and checkpoints, and generates visualisations.

The first execution may download the PathMNIST dataset through MedMNIST.

## 11. Experiment Tracking and Monitoring

All experiment tracking is maintained in a public Weights & Biases project.

**Project Dashboard:**

https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist/overview

The dashboard includes:

- Global test accuracy and loss.
- Individual client training accuracy and loss.
- Individual client validation accuracy and loss.
- Global and client update norms.
- Client cosine similarity values.
- Communication round information.
- Experiment configurations and run summaries.

Separate experiment runs are maintained for FedAvg and FedProx, alongside setup and tracking verification runs.

## 12. Testing and Verification

The implementation was checked using automated tests and Python environment validation.

### 12.1 Run automated tests

```bash
python -m pytest -q
```

Previously reported test result: **9 automated tests passed.**

### 12.2 Check Python compilation

```bash
python -m compileall -q .
```

### 12.3 Check package dependencies

```bash
python -m pip check
```

Previously reported verification:

- 9 automated tests passed.
- Python compilation completed successfully.
- No broken package requirements were reported.

## 13. Limitations and Future Work

The current implementation provides a preliminary experimental comparison of FedAvg and FedProx.

The main limitations are:

- The experiments use only two communication rounds.
- Results are based on a limited experimental configuration.
- Additional random seeds are needed to study run-to-run variability.
- Longer training is required to investigate convergence behaviour.

Potential future improvements include:

- Increasing communication rounds.
- Running experiments with multiple random seeds.
- Evaluating different Dirichlet concentration parameters.
- Investigating different label-noise rates.
- Exploring different FedProx proximal coefficients.
- Analysing client-level performance over longer training periods.
- Investigating adaptive optimisation strategies for heterogeneous federated environments.

## 14. Conclusion

This project implements and compares Federated Averaging (FedAvg) and Federated Proximal (FedProx) for PathMNIST medical image classification across five simulated clients under non-IID data distributions and noisy-label conditions.

It brings together client-side training, federated aggregation, noisy-label experimentation, global evaluation, client-level validation, model update analysis, automated testing, experiment tracking and visualisation.

The preliminary results demonstrate differences between FedAvg and FedProx in global accuracy, loss and model update norms.

The implementation provides an experimental foundation for further investigation of federated optimisation under heterogeneous client data.

---

**Project Repository:** https://github.com/Sumukhi90/fl-pathmnist

**Experiment Tracking:** https://forge.coreweave.com/wandb/sumukhipandey22-k-r-mangalam-university/indiaai-federated-pathmnist/overview
