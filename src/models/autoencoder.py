"""
Complete Autoencoder and Deepfake Model (Lightweight Version)

This module combines the encoder and decoder into complete models:
1. AutoEncoder: Single encoder-decoder pair for reconstruction
2. DeepfakeModel: Shared encoder + two person-specific decoders for face swap

The magic of deepfake:
    - Train with: Encoder + Decoder_A on Person A's faces
    - Train with: Encoder + Decoder_B on Person B's faces
    - The SHARED encoder learns universal face features
    - Each decoder learns person-specific reconstruction

    To swap faces:
    - Encode A's face -> Get expression/pose in latent space
    - Decode with B's decoder -> B's face with A's expression!
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from .encoder import Encoder
from .decoder import Decoder


class AutoEncoder(nn.Module):
    """
    Basic Autoencoder for face reconstruction.

    Used for:
    - Understanding the basic concept
    - Pre-training encoder/decoder pairs
    - Single-person face reconstruction
    """

    def __init__(self, input_size: int = 128, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(latent_dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode input image."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(latent)


class DeepfakeModel(nn.Module):
    """
    Complete Deepfake Model with shared encoder and two decoders (Lightweight).

    Architecture:
        Person A Image ─┐
                        ├─> Shared Encoder ─> Latent ─┬─> Decoder A ─> Reconstructed A
        Person B Image ─┘                             └─> Decoder B ─> Reconstructed B

    Face Swap (after training):
        Person A Image ─> Encoder ─> Latent ─> Decoder B ─> A's expression on B's face

    Training Strategy:
        1. Feed A's image -> Encoder -> Decoder A -> Compare with A's image
        2. Feed B's image -> Encoder -> Decoder B -> Compare with B's image
        3. Encoder learns universal features, decoders learn person-specific features
    """

    def __init__(self, input_size: int = 128, latent_dim: int = 128):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # Shared encoder
        self.encoder = Encoder(input_size, latent_dim)

        # Person-specific decoders
        self.decoder_a = Decoder(latent_dim, input_size)
        self.decoder_b = Decoder(latent_dim, input_size)

    def forward(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x_a: Batch of person A's face images
            x_b: Batch of person B's face images

        Returns:
            Tuple of (reconstructed_a, reconstructed_b)
        """
        # Encode both persons with shared encoder
        latent_a = self.encoder(x_a)
        latent_b = self.encoder(x_b)

        # Decode with respective decoders
        recon_a = self.decoder_a(latent_a)
        recon_b = self.decoder_b(latent_b)

        return recon_a, recon_b

    def swap_face(
        self, source: torch.Tensor, target_person: str = "b"
    ) -> torch.Tensor:
        """
        Perform face swap: source expression on target person's face.

        This is the actual deepfake operation!

        Args:
            source: Source face image (person whose expression to use)
            target_person: 'a' or 'b' - whose face to generate

        Returns:
            Swapped face: target person's face with source's expression
        """
        # Encode source face to get expression/pose
        latent = self.encoder(source)

        # Decode with target person's decoder
        if target_person.lower() == "a":
            swapped = self.decoder_a(latent)
        else:
            swapped = self.decoder_b(latent)

        return swapped

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation of any face."""
        return self.encoder(x)

    def get_training_outputs(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get all outputs needed for training.

        Returns dict with:
        - recon_a: A reconstructed from A
        - recon_b: B reconstructed from B
        - latent_a: A's latent representation
        - latent_b: B's latent representation
        - swap_a_to_b: A's expression on B's face
        - swap_b_to_a: B's expression on A's face
        """
        latent_a = self.encoder(x_a)
        latent_b = self.encoder(x_b)

        return {
            "recon_a": self.decoder_a(latent_a),
            "recon_b": self.decoder_b(latent_b),
            "latent_a": latent_a,
            "latent_b": latent_b,
            "swap_a_to_b": self.decoder_b(latent_a),  # A's expression on B
            "swap_b_to_a": self.decoder_a(latent_b),  # B's expression on A
        }


def get_device() -> torch.device:
    """Get the best available device for M4 Mac."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    # Test the complete model
    device = get_device()
    print(f"Using device: {device}")

    # Create model
    model = DeepfakeModel().to(device)

    # Test inputs
    batch_size = 4
    x_a = torch.randn(batch_size, 3, 128, 128).to(device)
    x_b = torch.randn(batch_size, 3, 128, 128).to(device)

    # Forward pass
    recon_a, recon_b = model(x_a, x_b)

    print(f"\n=== Model Test ===")
    print(f"Input A shape: {x_a.shape}")
    print(f"Input B shape: {x_b.shape}")
    print(f"Reconstructed A shape: {recon_a.shape}")
    print(f"Reconstructed B shape: {recon_b.shape}")

    # Face swap test
    swapped = model.swap_face(x_a, target_person="b")
    print(f"Swapped (A -> B) shape: {swapped.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder_a.parameters())

    print(f"\n=== Parameter Count ===")
    print(f"Encoder: {encoder_params:,}")
    print(f"Decoder (each): {decoder_params:,}")
    print(f"Total: {total_params:,}")
