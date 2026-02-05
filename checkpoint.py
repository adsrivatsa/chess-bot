import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim

from replay_buffer import ReplayBuffer


def save(
    path: str,
    epoch: int,
    wandb_id: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer": replay_buffer.buffer,
        "wandb_id": wandb_id,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(checkpoint_data, path)


def load(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    device: torch.device,
) -> Tuple[int, str | None]:
    if not os.path.exists(path):
        return 0, None

    print(f"--> Found checkpoint! Resuming from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    replay_buffer.buffer = checkpoint["replay_buffer"]

    # Restore RNGs if available
    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        torch.set_rng_state(rng["torch"].cpu())
        cuda_states = [state.cpu() for state in rng["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1
    wandb_id = checkpoint.get("wandb_id", None)

    print(f"--> Resumed at Epoch {start_epoch}")
    return start_epoch, wandb_id
