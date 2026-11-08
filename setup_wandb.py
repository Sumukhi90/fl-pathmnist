
import os
import wandb

os.environ["WANDB_BASE_URL"] = "https://api.forge.coreweave.com"

print("W&B version:", wandb.__version__)

# Authenticate using credentials stored in .netrc
success = wandb.login(
    host="https://api.forge.coreweave.com"
)

if success:
    print("SUCCESS: W&B authentication completed!")
else:
    print("Authentication unsuccessful.")
