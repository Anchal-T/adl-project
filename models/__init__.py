from torch import nn

from models.attention_unet import AttentionUNet
from models.unet import UNet

MODELS = {"unet": UNet, "attention_unet": AttentionUNet}
MODEL_NAMES = tuple(MODELS)


def build_model(name: str, base_channels: int = 64) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', expected one of {MODEL_NAMES}")
    return MODELS[name](base_channels=base_channels)


__all__ = ["AttentionUNet", "MODEL_NAMES", "UNet", "build_model"]
