import numpy as np
import pytest
import torch

from explain import attention_maps, energy_inside_ratio, min_max_normalize, seg_grad_cam
from models import build_model


def test_energy_inside_ratio_matches_hand_computed_value():
    heatmap = np.array([[1.0, 0.0], [0.0, 1.0]])
    gt = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert energy_inside_ratio(heatmap, gt) == pytest.approx(0.5)


def test_energy_inside_ratio_is_one_when_all_energy_is_on_gt():
    heatmap = np.array([[0.0, 2.0], [0.0, 0.0]])
    gt = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert energy_inside_ratio(heatmap, gt) == pytest.approx(1.0)


def test_energy_inside_ratio_handles_zero_heatmap():
    assert energy_inside_ratio(np.zeros((2, 2)), np.ones((2, 2))) == 0.0


def test_min_max_normalize_maps_to_unit_range():
    normalized = min_max_normalize(np.array([[2.0, 4.0], [3.0, 6.0]]))
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    assert np.all(min_max_normalize(np.zeros((2, 2))) == 0.0)


@pytest.mark.parametrize("name", ["unet", "attention_unet"])
def test_seg_grad_cam_returns_heatmap_at_input_resolution(name):
    torch.manual_seed(0)
    model = build_model(name, base_channels=4).eval()
    image = torch.rand(1, 3, 64, 64)
    heatmap, logits = seg_grad_cam(model, image)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert logits.shape == (1, 1, 64, 64)


def test_attention_maps_are_full_resolution_and_ordered_shallow_to_deep():
    model = build_model("attention_unet", base_channels=4).eval()
    maps = attention_maps(model, torch.rand(1, 3, 64, 64))
    assert len(maps) == 4
    assert all(m.shape == (64, 64) for m in maps)
