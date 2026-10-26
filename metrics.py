import torch

METRIC_NAMES = ("dice", "iou", "precision", "recall")
DEFAULT_THRESHOLD = 0.5


def binarize(logits: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor, empty_value: float) -> torch.Tensor:
    # Counts are integers after binarization, so clamping the denominator at 1 is exact.
    safe = numerator / denominator.clamp_min(1.0)
    return torch.where(denominator > 0, safe, torch.full_like(safe, empty_value))


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Per-image Dice, IoU, precision and recall for tensors shaped (B, 1, H, W).

    Empty-mask conventions: Dice and IoU are 1 when prediction and target are both empty,
    precision and recall are 0 when their denominator is 0.
    """
    preds = binarize(logits, threshold)
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)
    return {
        "dice": _ratio(2 * tp, 2 * tp + fp + fn, empty_value=1.0),
        "iou": _ratio(tp, tp + fp + fn, empty_value=1.0),
        "precision": _ratio(tp, tp + fp, empty_value=0.0),
        "recall": _ratio(tp, tp + fn, empty_value=0.0),
    }


def summarize(per_image: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for name in METRIC_NAMES:
        values = torch.tensor([row[name] for row in per_image], dtype=torch.float64)
        summary[name] = {"mean": values.mean().item(), "std": values.std(unbiased=False).item()}
    return summary
