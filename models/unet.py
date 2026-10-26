import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) with batch norm; outputs logits, apply a sigmoid for probabilities."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        widths = [base_channels * 2**i for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for width in widths[:-1]:
            self.encoders.append(DoubleConv(prev, width))
            prev = width
        self.bottleneck = DoubleConv(prev, widths[-1])
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = widths[-1]
        for width in reversed(widths[:-1]):
            self.upsamplers.append(nn.ConvTranspose2d(prev, width, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(2 * width, width))
            prev = width

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        return self.bottleneck(x), skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encode(x)
        for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = decoder(torch.cat([skip, up(x)], dim=1))
        return self.head(x)
