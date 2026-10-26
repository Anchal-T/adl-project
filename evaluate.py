"""Evaluate a trained checkpoint on the held-out test split. Run once, after model selection is finished."""

import argparse
import csv
import json
from pathlib import Path

from data import build_loader, load_or_create_split
from metrics import DEFAULT_THRESHOLD, METRIC_NAMES, summarize
from train import evaluate_model
from utils import get_device, load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the Kvasir-SEG test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/Kvasir-SEG")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict:
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    split = load_or_create_split(args.data_dir, args.split_file)
    loader = build_loader(
        args.data_dir, split["test"], args.batch_size, checkpoint["image_size"],
        num_workers=args.num_workers,
    )
    rows, test_loss = evaluate_model(model, loader, device, checkpoint["lambda_dice"])

    name = checkpoint["model_name"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_test_per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": name,
        "checkpoint": str(args.checkpoint),
        "best_epoch": checkpoint["epoch"],
        "num_test_images": len(rows),
        "threshold": DEFAULT_THRESHOLD,
        "test_loss": test_loss,
        "metrics": summarize(rows),
    }
    (out_dir / f"{name}_test_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    return summary


def main(argv: list[str] | None = None) -> dict:
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
