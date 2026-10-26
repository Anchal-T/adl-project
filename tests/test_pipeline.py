import json
from pathlib import Path

import pytest

from evaluate import main as evaluate_main
from explain import main as explain_main
from metrics import METRIC_NAMES
from train import main as train_main

TINY = ["--device", "cpu", "--image_size", "64", "--base_channels", "4", "--batch_size", "4", "--num_workers", "0"]


@pytest.mark.parametrize("model_name", ["unet", "attention_unet"])
def test_train_evaluate_explain_on_synthetic_data(synthetic_kvasir, tmp_path, model_name):
    out_dir = tmp_path / "outputs"

    train_summary = train_main([
        "--model", model_name, "--epochs", "2", "--patience", "5",
        "--data_dir", str(synthetic_kvasir), "--out_dir", str(out_dir), *TINY,
    ])
    checkpoint = Path(train_summary["checkpoint"])
    assert checkpoint.exists()
    assert train_summary["best_epoch"] in (1, 2)
    assert (out_dir / f"{model_name}_history.csv").exists()

    test_summary = evaluate_main([
        "--checkpoint", str(checkpoint), "--data_dir", str(synthetic_kvasir),
        "--out_dir", str(out_dir), "--device", "cpu", "--num_workers", "0",
    ])
    assert test_summary["num_test_images"] == 2
    assert set(test_summary["metrics"]) == set(METRIC_NAMES)
    assert json.loads((out_dir / f"{model_name}_test_summary.json").read_text())["model"] == model_name

    explain_dir = tmp_path / "explain"
    explain_summary = explain_main([
        "--checkpoint", str(checkpoint), "--data_dir", str(synthetic_kvasir),
        "--out_dir", str(explain_dir), "--device", "cpu", "--num_figures", "1",
    ])
    assert explain_summary["num_images"] == 2
    assert 0.0 <= explain_summary["mean_e_in"] <= 1.0
    assert len(list((explain_dir / "figures").glob("*.png"))) == 1
    assert (explain_dir / f"{model_name}_e_in.csv").exists()


def test_both_models_share_the_same_saved_split(synthetic_kvasir, tmp_path):
    for model_name in ("unet", "attention_unet"):
        train_main([
            "--model", model_name, "--epochs", "1",
            "--data_dir", str(synthetic_kvasir), "--out_dir", str(tmp_path / model_name), *TINY,
        ])
    split = json.loads((synthetic_kvasir / "split.json").read_text())
    assert len(split["train"]) == 16
    assert len(split["val"]) == 2
    assert len(split["test"]) == 2
