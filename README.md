# Federated Learning Benchmark on PathMNIST



## Project Overview



This project implements and evaluates two federated learning algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)**, using the PathMNIST medical imaging dataset.



The objective is to study collaborative model training across multiple distributed clients while keeping their training data local.



The experiments also investigate client-level performance, noisy-label effects, model updates, and global model performance.



## Objectives



- Implement FedAvg and FedProx.

- Simulate distributed training across five clients.

- Evaluate global and client-level model performance.

- Investigate the impact of noisy labels on federated learning.

- Track model updates using update norms and cosine similarity.

- Compare global test accuracy and loss.

- Track experiments using Weights \& Biases (W\&B).



## Technologies Used



- Python

- PyTorch

- Torchvision

- MedMNIST

- NumPy

- Pandas

- Matplotlib

- Scikit-learn

- Weights \& Biases

- Pytest



## Project Structure



```text

fl-pathmnist/

â”‚

â”œâ”€â”€ configs/

â”œâ”€â”€ data/

â”œâ”€â”€ plots/

â”œâ”€â”€ plots\_pilot\_backup/

â”œâ”€â”€ results/

â”œâ”€â”€ results\_pilot\_backup/

â”œâ”€â”€ src/

â”‚   â”œâ”€â”€ aggregation.py

â”‚   â”œâ”€â”€ client.py

â”‚   â”œâ”€â”€ data.py

â”‚   â”œâ”€â”€ evaluate.py

â”‚   â”œâ”€â”€ fedprox.py

â”‚   â”œâ”€â”€ models.py

â”‚   â”œâ”€â”€ noise.py

â”‚   â”œâ”€â”€ partition.py

â”‚   â”œâ”€â”€ server.py

â”‚   â””â”€â”€ tracking.py

â”‚

â”œâ”€â”€ tests/

â”œâ”€â”€ run\_experiment.py

â”œâ”€â”€ setup\_wandb.py

â”œâ”€â”€ requirements.txt

â””â”€â”€ README.md

```



## Experimental Setup



| Parameter | Configuration |

|---|---|

| Dataset | PathMNIST |

| Framework | PyTorch |

| Federated clients | 5 |

| Algorithms | FedAvg and FedProx |

| Communication rounds | 2 |

| Experiment tracking | Weights \& Biases |

| Evaluation | Global testing and client validation |



The project includes client-specific training and validation, label-noise experimentation, global aggregation, and model evaluation.



## Experimental Results



### Round 2: FedAvg vs FedProx



| Metric | FedAvg | FedProx |

|---|---:|---:|

| Global test accuracy | 37.77% | 35.21% |

| Global test loss | 1.8813 | 2.0590 |

| Mean client validation accuracy | 26.63% | 24.88% |

| Global update norm | 1.1970 | 0.5585 |



### Observations



- Both algorithms improved global test accuracy between the first and second communication rounds.

- FedAvg achieved higher global test accuracy than FedProx in the second round.

- FedProx produced a smaller global update norm.

- Client-level validation performance varied across the five clients.

- The noisy-label client showed comparatively low validation performance.



These results are preliminary and should be interpreted in the context of the limited two-round experiment.



## Generated Outputs



The `results/` directory contains:



- `experiment\_results.csv`

- `experiment\_results.json`

- `fedavg\_metrics.csv`

- `fedavg\_model.pth`

- `fedprox\_metrics.csv`

- `fedprox\_model.pth`



The `plots/` directory contains:



- Global test accuracy

- Global test loss

- Client training accuracy

- Client validation accuracy

- Client update norms

- Client cosine similarity

- Client accuracy



## Installation



Create and activate a virtual environment:



```bash

python -m venv .venv

```



Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



## Running the Experiments



Execute:



```bash

python run\_experiment.py

```



The experiment pipeline runs the configured federated learning experiments, evaluates the models, saves results and generates visualizations.



## Experiment Tracking



Weights \& Biases is used to track training metrics, validation metrics, client updates, and experiment summaries.



The project uses a configured W\&B project for experiment logging.



Configure your own W\&B credentials securely before running experiments. Never commit API keys or passwords to the repository.



## Testing and Verification



The project was verified using:



```bash

python -m compileall -q .

python -m pip check

python -m pytest -q

```



**Verification results:**



- Python compilation completed successfully.

- No broken package requirements were reported.

- 9 automated tests passed.



## Limitations and Future Work



- Increase the number of communication rounds.

- Conduct experiments with multiple random seeds.

- Evaluate different client partitioning strategies.

- Investigate different label-noise levels.

- Compare FedAvg and FedProx across additional experimental settings.

- Perform further analysis of client-level performance and convergence.



## Conclusion



This project establishes an experimental federated learning pipeline for PathMNIST using FedAvg and FedProx.



It supports distributed client training, model aggregation, noisy-label experimentation, client-level validation, global evaluation, experiment tracking, and visualization.



The initial results provide a foundation for further investigation into federated learning performance under heterogeneous client conditions.
