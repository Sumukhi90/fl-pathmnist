
import os
import wandb

WANDB_PROJECT = "indiaai-federated-pathmnist"
WANDB_ENTITY = "sumukhipandey22-k-r-mangalam-university"
WANDB_BASE_URL = "https://api.forge.coreweave.com"


def start_run(project=None, run_name=None, config=None):
    os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL

    try:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=project or WANDB_PROJECT,
            name=run_name,
            config=config or {},
            reinit="finish_previous"
        )

        print("W&B run initialized successfully.")
        return run

    except Exception as exc:
        print("W&B unavailable. Continuing with local logging.")
        print("Reason:", exc)
        return None


def log_metrics(metrics, step=None, run=None):
    if run is None:
        return

    run.log(metrics, step=step)


def finish_run(run=None):
    if run is not None:
        run.finish()
