import pytest
import torch

from models import MODEL_NAMES, build_model
from models.attention_unet import AttentionGate, AttentionUNet
from models.unet import UNet

SMALL_CHANNELS = 8


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_output_logits_with_input_resolution(name):
    model = build_model(name, base_channels=SMALL_CHANNELS)
    images = torch.rand(2, 3, 256, 256)
    logits = model(images)
    assert logits.shape == (2, 1, 256, 256)


def test_unet_has_four_encoder_and_decoder_stages():
    model = UNet(base_channels=SMALL_CHANNELS)
    assert len(model.encoders) == 4
    assert len(model.decoders) == 4


def test_attention_unet_returns_one_map_per_stage_at_stage_resolution():
    model = AttentionUNet(base_channels=SMALL_CHANNELS).eval()
    images = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        logits, maps = model(images, return_attention=True)

    assert logits.shape == (1, 1, 256, 256)
    assert len(maps) == 4
    expected_sizes = [32, 64, 128, 256]
    for alpha, size in zip(maps, expected_sizes):
        assert alpha.shape == (1, 1, size, size)
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_attention_gate_scales_skip_by_its_map():
    gate = AttentionGate(gate_channels=8, skip_channels=8, inter_channels=4)
    skip = torch.rand(2, 8, 16, 16)
    gating_signal = torch.rand(2, 8, 16, 16)
    gated, alpha = gate(skip, gating_signal)
    assert alpha.shape == (2, 1, 16, 16)
    assert torch.allclose(gated, skip * alpha)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("resnet")
