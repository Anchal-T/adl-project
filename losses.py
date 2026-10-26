import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return (1 - dice).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, lambda_dice: float = 1.0) -> torch.Tensor:
    """L = BCE + lambda * Dice, both averaged over the batch."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + lambda_dice * dice_loss(logits, targets)
