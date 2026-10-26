"""Seg-Grad-CAM heatmaps, attention-gate maps, the E_in localisation score and explanation figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import KvasirSegDataset, load_or_create_split
from metrics import DEFAULT_THRESHOLD, segmentation_metrics
from models import AttentionUNet
from utils import get_device, load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a trained checkpoint on the test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/Kvasir-SEG")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--out_dir", default="outputs/explain")
    parser.add_argument("--num_figures", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    array = array - array.min()
    peak = array.max()
    return array / peak if peak > 0 else array


def _capture_output(store: dict):
    def hook(_module, _inputs, output):
        store["activation"] = output

    return hook


def seg_grad_cam(model, image: torch.Tensor, threshold: float = DEFAULT_THRESHOLD):
    """Seg-Grad-CAM (Vinogradova et al., 2020) for one image shaped (1, 3, H, W).

    The target score is the sum of foreground logits over the predicted mask, so the
    gradients explain the pixels the model segments as polyp. The heatmap is taken at the
    last decoder stage. Returns the heatmap in [0, 1] at input resolution and the logits.
    """
    store = {}
    handle = model.decoders[-1].register_forward_hook(_capture_output(store))
    try:
        logits = model(image)
    finally:
        handle.remove()
    activation = store["activation"]

    mask = (torch.sigmoid(logits) >= threshold).float()
    score = (logits * mask).sum() if mask.sum() > 0 else logits.sum()
    (gradients,) = torch.autograd.grad(score, activation)

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    heatmap = min_max_normalize(cam[0, 0].detach().cpu().numpy())
    return heatmap, logits.detach()


@torch.no_grad()
def attention_maps(model: AttentionUNet, image: torch.Tensor) -> list[np.ndarray]:
    """Attention maps resized to input resolution, ordered from full resolution to the deepest gate."""
    _, maps = model(image, return_attention=True)
    resized = [
        F.interpolate(alpha, size=image.shape[-2:], mode="bilinear", align_corners=False)
        for alpha in maps
    ]
    return [alpha[0, 0].cpu().numpy() for alpha in resized[::-1]]


def energy_inside_ratio(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """E_in = sum of heatmap over ground-truth pixels / sum of heatmap over all pixels."""
    total = heatmap.sum()
    if total <= 0:
        return 0.0
    return float(heatmap[gt_mask > 0.5].sum() / total)


def save_figure(path, image, heatmap, probability, gt, attention, title, threshold):
    overlay = 0.5 * image + 0.5 * matplotlib.colormaps["jet"](heatmap)[..., :3]
    panels = [
        ("Image", image, None),
        ("Seg-Grad-CAM", overlay, None),
        ("Predicted mask", (probability >= threshold).astype(float), "gray"),
        ("Ground truth", gt, "gray"),
    ]
    rows = 2 if attention else 1
    cols = max(len(panels), len(attention))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()

    for ax, (name, data, cmap) in zip(axes[0], panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(name)
    for index, (ax, alpha) in enumerate(zip(axes[-1], attention)):
        ax.imshow(alpha, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Attention gate {index + 1}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_explanation(args: argparse.Namespace) -> dict:
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    name = checkpoint["model_name"]
    is_attention = isinstance(model, AttentionUNet)

    split = load_or_create_split(args.data_dir, args.split_file)
    dataset = KvasirSegDataset(args.data_dir, split["test"], image_size=checkpoint["image_size"])
    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(len(dataset)):
        image, gt, sample_id = dataset[index]
        batch = image.unsqueeze(0).to(device)
        heatmap, logits = seg_grad_cam(model, batch, args.threshold)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        dice = segmentation_metrics(logits, gt.unsqueeze(0).to(device), args.threshold)["dice"].item()
        e_in = energy_inside_ratio(heatmap, gt[0].numpy())
        rows.append({"image_id": sample_id, "e_in": e_in, "dice": dice})

        if index < args.num_figures:
            maps = attention_maps(model, batch) if is_attention else []
            save_figure(
                figure_dir / f"{name}_{sample_id}.png",
                image.permute(1, 2, 0).numpy(),
                heatmap,
                probability,
                gt[0].numpy(),
                maps,
                title=f"{name} | {sample_id} | E_in = {e_in:.3f} | Dice = {dice:.3f}",
                threshold=args.threshold,
            )

    with (out_dir / f"{name}_e_in.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "e_in", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    e_in_values = np.array([row["e_in"] for row in rows])
    summary = {
        "model": name,
        "checkpoint": str(args.checkpoint),
        "num_images": len(rows),
        "mean_e_in": float(e_in_values.mean()),
        "std_e_in": float(e_in_values.std()),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
    }
    (out_dir / f"{name}_explain_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> dict:
    return run_explanation(parse_args(argv))


if __name__ == "__main__":
    main()
