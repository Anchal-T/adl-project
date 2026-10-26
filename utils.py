import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_name: str,
    base_channels: int,
    image_size: int,
    lambda_dice: float,
    epoch: int,
    val_dice: float,
) -> None:
    torch.save(
        {
            "model_name": model_name,
            "base_channels": base_channels,
            "image_size": image_size,
            "lambda_dice": lambda_dice,
            "epoch": epoch,
            "val_dice": val_dice,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
