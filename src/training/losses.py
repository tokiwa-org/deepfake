"""
Loss Functions for Deepfake Training (Lightweight Version)

Multiple loss functions are combined for better results:

1. Reconstruction Loss (L1/L2):
   - Pixel-level difference between input and output
   - L1 (MAE) produces sharper images than L2 (MSE)

2. Perceptual Loss (VGG) - OPTIONAL:
   - Compares high-level features from VGG network
   - Captures structural similarity better than pixel loss
   - Makes faces look more natural
   - NOTE: Significantly slower, disabled by default for lightweight training

For educational purposes, we use L1 loss by default.
Enable perceptual loss for better quality at the cost of speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


class ReconstructionLoss(nn.Module):
    """
    Basic reconstruction loss.

    Combines L1 and L2 losses with configurable weights.
    L1 (MAE) is preferred for face generation as it produces sharper images.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss.

        Args:
            predicted: Generated image (B, 3, H, W)
            target: Original image (B, 3, H, W)

        Returns:
            Weighted sum of L1 and L2 losses
        """
        loss = 0.0
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1_loss(predicted, target)
        if self.l2_weight > 0:
            loss += self.l2_weight * self.l2_loss(predicted, target)
        return loss


class VGGFeatureExtractor(nn.Module):
    """
    Extract features from VGG19 for perceptual loss.

    VGG19 is pre-trained on ImageNet and has learned
    hierarchical features that are useful for comparing images.

    Lower layers: edges, colors, textures
    Higher layers: shapes, objects, faces
    """

    def __init__(
        self,
        feature_layers: List[int] = [3, 8, 17, 26],  # relu1_2, relu2_2, relu3_4, relu4_4
        use_input_norm: bool = True,
    ):
        super().__init__()

        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children()))

        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False

        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features at multiple layers.

        Args:
            x: Input image (B, 3, H, W) in [0, 1] range

        Returns:
            List of feature tensors at specified layers
        """
        # Normalize to ImageNet stats
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)

        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Compares high-level features instead of raw pixels.
    This makes generated faces look more natural and
    preserves important facial structures.

    The loss is computed as:
        L_perceptual = sum( ||VGG(pred)_i - VGG(target)_i||_1 )

    where i indexes the selected VGG layers.
    """

    def __init__(
        self,
        feature_layers: List[int] = [3, 8, 17, 26],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.vgg = VGGFeatureExtractor(feature_layers)
        self.vgg.eval()

        # Layer weights (higher layers weighted more)
        if weights is None:
            weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]
        self.weights = weights

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate perceptual loss.

        Args:
            predicted: Generated image (B, 3, H, W)
            target: Original image (B, 3, H, W)

        Returns:
            Weighted sum of feature losses
        """
        pred_features = self.vgg(predicted)
        target_features = self.vgg(target)

        loss = 0.0
        for pred_feat, target_feat, weight in zip(
            pred_features, target_features, self.weights
        ):
            loss += weight * F.l1_loss(pred_feat, target_feat)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for deepfake training.

    Total loss = λ_recon * L_recon + λ_perceptual * L_perceptual

    By default, perceptual loss is DISABLED for faster training.
    Set use_perceptual=True for better quality at the cost of speed.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        use_perceptual: bool = False,  # Disabled by default for speed
    ):
        super().__init__()
        self.recon_loss = ReconstructionLoss(l1_weight=1.0)
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual

        # Only create perceptual loss if needed
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> dict:
        """
        Calculate combined loss.

        Returns dict with individual losses and total.
        """
        recon = self.recon_loss(predicted, target)

        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(predicted, target)
            total = self.recon_weight * recon + self.perceptual_weight * perceptual
        else:
            perceptual = torch.tensor(0.0, device=predicted.device)
            total = self.recon_weight * recon

        return {
            "total": total,
            "recon": recon,
            "perceptual": perceptual,
        }


if __name__ == "__main__":
    # Test losses
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy images (128x128 for lightweight version)
    pred = torch.rand(2, 3, 128, 128).to(device)
    target = torch.rand(2, 3, 128, 128).to(device)

    # Test reconstruction loss
    recon_loss = ReconstructionLoss()
    print(f"Reconstruction loss: {recon_loss(pred, target):.4f}")

    # Test combined loss (without perceptual - fast)
    combined_fast = CombinedLoss(use_perceptual=False).to(device)
    losses_fast = combined_fast(pred, target)
    print(f"Combined loss (fast): {losses_fast}")

    # Test combined loss (with perceptual - slow but better quality)
    combined_quality = CombinedLoss(use_perceptual=True).to(device)
    losses_quality = combined_quality(pred, target)
    print(f"Combined loss (quality): {losses_quality}")
