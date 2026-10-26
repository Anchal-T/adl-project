import math

import pytest
import torch

from losses import bce_dice_loss, dice_loss
from metrics import segmentation_metrics, summarize


def test_dice_loss_matches_hand_computed_value():
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.ones(1, 1, 2, 2)
    # probs = 0.5 everywhere: intersection = 2, denominator = 2 + 4, dice = (4 + 1) / (6 + 1)
    assert dice_loss(logits, targets).item() == pytest.approx(1 - 5 / 7)


def test_dice_loss_is_near_zero_for_confident_correct_prediction():
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = (targets * 2 - 1) * 20
    assert dice_loss(logits, targets).item() < 1e-3


def test_bce_dice_loss_reduces_to_bce_when_lambda_is_zero():
    logits = torch.zeros(1, 1, 4, 4)
    targets = torch.randint(0, 2, (1, 1, 4, 4)).float()
    assert bce_dice_loss(logits, targets, lambda_dice=0.0).item() == pytest.approx(math.log(2), rel=1e-5)


def test_bce_dice_loss_adds_weighted_dice():
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.ones(1, 1, 2, 2)
    expected = math.log(2) + 0.5 * (1 - 5 / 7)
    assert bce_dice_loss(logits, targets, lambda_dice=0.5).item() == pytest.approx(expected, rel=1e-5)


def test_metrics_match_hand_computed_confusion_counts():
    # Flattened: preds = [1, 0, 1, 0], targets = [1, 1, 0, 0] -> tp = 1, fn = 1, fp = 1
    logits = torch.tensor([[[[5.0, -5.0], [5.0, -5.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    metrics = segmentation_metrics(logits, targets)

    assert metrics["dice"].item() == pytest.approx(0.5)
    assert metrics["iou"].item() == pytest.approx(1 / 3)
    assert metrics["precision"].item() == pytest.approx(0.5)
    assert metrics["recall"].item() == pytest.approx(0.5)


def test_threshold_controls_binarization():
    logits = torch.tensor([[[[0.0, 0.0]]]])
    targets = torch.ones(1, 1, 1, 2)
    assert segmentation_metrics(logits, targets, threshold=0.5)["recall"].item() == pytest.approx(1.0)
    assert segmentation_metrics(logits, targets, threshold=0.6)["recall"].item() == pytest.approx(0.0)


def test_empty_mask_conventions():
    logits = torch.full((1, 1, 2, 2), -10.0)
    targets = torch.zeros(1, 1, 2, 2)
    metrics = segmentation_metrics(logits, targets)
    assert metrics["dice"].item() == pytest.approx(1.0)
    assert metrics["iou"].item() == pytest.approx(1.0)
    assert metrics["precision"].item() == pytest.approx(0.0)
    assert metrics["recall"].item() == pytest.approx(0.0)


def test_summarize_reports_mean_and_population_std():
    rows = [
        {"dice": 0.2, "iou": 0.1, "precision": 1.0, "recall": 0.0},
        {"dice": 0.6, "iou": 0.5, "precision": 0.0, "recall": 1.0},
    ]
    summary = summarize(rows)
    assert summary["dice"]["mean"] == pytest.approx(0.4)
    assert summary["dice"]["std"] == pytest.approx(0.2)
