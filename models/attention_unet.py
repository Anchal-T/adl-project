import torch
from torch import nn

from models.unet import UNet


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018) that re-weights a skip connection."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.relu(self.theta_x(skip) + self.phi_g(gate))
        alpha = torch.sigmoid(self.psi(combined))
        return skip * alpha, alpha


class AttentionUNet(UNet):
    """U-Net whose skip connections pass through attention gates gated by the decoder features.

    forward returns logits, or (logits, attention maps) when return_attention is True.
    Attention maps are ordered from the deepest decoder stage to the full-resolution one.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__(in_channels, out_channels, base_channels, depth)
        self.gates = nn.ModuleList(
            AttentionGate(width, width, max(width // 2, 1)) for width in self._decoder_widths()
        )

    def _decoder_widths(self) -> list[int]:
        return [up.out_channels for up in self.upsamplers]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x, skips = self.encode(x)
        attention_maps = []
        for up, gate, decoder, skip in zip(self.upsamplers, self.gates, self.decoders, reversed(skips)):
            gating_signal = up(x)
            gated_skip, alpha = gate(skip, gating_signal)
            attention_maps.append(alpha)
            x = decoder(torch.cat([gated_skip, gating_signal], dim=1))
        logits = self.head(x)
        return (logits, attention_maps) if return_attention else logits
