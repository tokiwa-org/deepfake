"""
Decoder Network for Deepfake Autoencoder (Lightweight Version)

The decoder learns to reconstruct a SPECIFIC person's face from the latent space.
Each person has their OWN decoder - this is the key to face swapping.

Architecture (Lightweight):
    Input: 4x4x128 latent tensor
    Output: 128x128x3 RGB image

Key Changes from Heavy Version:
    - Reduced from 256x256 to 128x128 output
    - Reduced channels: 512 -> 128
    - Simpler architecture
    - ~10M params per decoder (was ~500M)
"""

import torch
import torch.nn as nn


class PixelShuffleUpBlock(nn.Module):
    """
    Upsampling using PixelShuffle.
    Reduces checkerboard artifacts compared to transposed conv.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * (scale_factor**2), 3, 1, 1, bias=False
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pixel_shuffle(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for decoder."""

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


class Decoder(nn.Module):
    """
    Person-specific Decoder for Deepfake model (Lightweight Version).

    Each person (A and B) has their own decoder.
    The decoder learns to reconstruct that specific person's face
    from the shared latent representation.

    Face Swap works by:
    1. Encode person A's face -> latent
    2. Decode with person B's decoder -> B's face with A's expression

    Input: (B, 128, 4, 4) - Latent representation
    Output: (B, 3, 128, 128) - Reconstructed RGB face image
    """

    def __init__(self, latent_dim: int = 128, output_size: int = 128):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_size = output_size

        # Initial processing of latent space
        self.initial = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
        )

        # Upsampling path: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            # 4x4x128 -> 8x8x128
            PixelShuffleUpBlock(128, 128),
            ResidualBlock(128),
            # 8x8x128 -> 16x16x128
            PixelShuffleUpBlock(128, 128),
            ResidualBlock(128),
            # 16x16x128 -> 32x32x64
            PixelShuffleUpBlock(128, 64),
            ResidualBlock(64),
            # 32x32x64 -> 64x64x32
            PixelShuffleUpBlock(64, 32),
            # 64x64x32 -> 128x128x32
            PixelShuffleUpBlock(32, 32),
        )

        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to face image.

        Args:
            latent: Latent tensor of shape (B, 128, 4, 4)

        Returns:
            Reconstructed image of shape (B, 3, 128, 128)
        """
        x = self.initial(latent)
        x = self.decoder(x)
        return self.output(x)


if __name__ == "__main__":
    # Test decoder
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    decoder = Decoder().to(device)
    dummy_latent = torch.randn(2, 128, 4, 4).to(device)
    output = decoder(dummy_latent)

    print(f"Input shape: {dummy_latent.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
