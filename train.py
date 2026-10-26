"""Train U-Net or Attention U-Net on the Kvasir-SEG training split with early stopping on validation Dice."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import IMAGE_SIZE, build_loader, load_or_create_split
from losses import bce_dice_loss
from metrics import segmentation_metrics, summarize
from models import MODEL_NAMES, build_model
from utils import get_device, save_checkpoint, set_seed

MIN_DELTA = 1e-4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net or Attention U-Net on Kvasir-SEG.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--data_dir", default="data/Kvasir-SEG")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def train_one_epoch(model, loader, optimizer, lambda_dice: float, device: torch.device) -> float:
    model.train()
    total_loss, count = 0.0, 0
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = bce_dice_loss(model(images), masks, lambda_dice)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        count += images.size(0)
    return total_loss / count


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, lambda_dice: float = 1.0):
    """Return per-image metric rows and the mean loss over the loader."""
    model.eval()
    rows, total_loss = [], 0.0
    for images, masks, sample_ids in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += bce_dice_loss(logits, masks, lambda_dice).item() * images.size(0)
        per_image = segmentation_metrics(logits, masks)
        for i, sample_id in enumerate(sample_ids):
            rows.append({"image_id": sample_id, **{k: v[i].item() for k, v in per_image.items()}})
    return rows, total_loss / len(rows)


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run_training(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.model}_config.json").write_text(json.dumps(vars(args), indent=2))

    split = load_or_create_split(args.data_dir, args.split_file, seed=args.seed)
    train_loader = build_loader(
        args.data_dir, split["train"], args.batch_size, args.image_size,
        augment=True, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = build_loader(
        args.data_dir, split["val"], args.batch_size, args.image_size, num_workers=args.num_workers,
    )

    model = build_model(args.model, args.base_channels).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_path = out_dir / f"{args.model}_best.pt"
    history_path = out_dir / f"{args.model}_history.csv"

    best_dice, best_epoch, epochs_without_gain = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, args.lambda_dice, device)
        val_rows, val_loss = evaluate_model(model, val_loader, device, args.lambda_dice)
        val_summary = summarize(val_rows)
        val_dice = val_summary["dice"]["mean"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_summary["iou"]["mean"],
        })
        write_history(history_path, history)
        print(f"epoch {epoch:03d} train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
              f"val_dice {val_dice:.4f} lr {lr:.2e}")

        if val_dice > best_dice + MIN_DELTA:
            best_dice, best_epoch, epochs_without_gain = val_dice, epoch, 0
            save_checkpoint(
                checkpoint_path,
                model,
                model_name=args.model,
                base_channels=args.base_channels,
                image_size=args.image_size,
                lambda_dice=args.lambda_dice,
                epoch=epoch,
                val_dice=val_dice,
            )
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"early stopping at epoch {epoch}, best epoch {best_epoch}")
                break

    return {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "best_val_dice": best_dice}


def main(argv: list[str] | None = None) -> dict:
    return run_training(parse_args(argv))


if __name__ == "__main__":
    main()
