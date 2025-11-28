"""
Encoder Network for Deepfake Autoencoder (Lightweight Version)

The encoder learns to extract facial features into a compact latent representation.
This is SHARED between both persons - it learns universal facial features.

Architecture (Lightweight):
    Input: 128x128x3 RGB image
    Output: 4x4x128 latent tensor

Key Changes from Heavy Version:
    - Reduced from 256x256 to 128x128 input
    - Reduced channels: 512 -> 128
    - Removed FC layer (direct conv output)
    - ~20M params total (was ~1B)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolutional block with LeakyReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class Encoder(nn.Module):
    """
    Shared Encoder for Deepfake model (Lightweight Version).

    This encoder is shared between both persons A and B.
    It learns to extract universal facial features that can be
    reconstructed by person-specific decoders.

    Input: (B, 3, 128, 128) - Batch of RGB face images
    Output: (B, 128, 4, 4) - Latent representation
    """

    def __init__(self, input_size: int = 128, latent_dim: int = 128):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # Downsampling path: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            # 128x128x3 -> 64x64x32
            ConvBlock(3, 32),
            # 64x64x32 -> 32x32x64
            ConvBlock(32, 64),
            ResidualBlock(64),
            # 32x32x64 -> 16x16x128
            ConvBlock(64, 128),
            ResidualBlock(128),
            # 16x16x128 -> 8x8x128
            ConvBlock(128, 128),
            ResidualBlock(128),
            # 8x8x128 -> 4x4x128
            ConvBlock(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image to latent representation.

        Args:
            x: Input tensor of shape (B, 3, 128, 128)

        Returns:
            Latent tensor of shape (B, 128, 4, 4)
        """
        return self.encoder(x)


if __name__ == "__main__":
    # Test encoder
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = Encoder().to(device)
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    output = encoder(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
