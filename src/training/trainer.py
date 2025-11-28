"""
Training Loop for Deepfake Model (Lightweight Version)

The trainer handles:
1. Training loop with both persons' images
2. Loss calculation and backpropagation
3. Model checkpointing
4. TensorBoard logging
5. Learning rate scheduling

Training Strategy:
    For each batch:
    1. Get images from both persons (img_a, img_b)
    2. Encode both -> latent_a, latent_b
    3. Decode with respective decoders -> recon_a, recon_b
    4. Calculate loss: L(recon_a, img_a) + L(recon_b, img_b)
    5. Backpropagate and update all parameters

The shared encoder learns features common to both faces,
while each decoder specializes in reconstructing one person.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import DeepfakeModel
from .losses import CombinedLoss


class DeepfakeTrainer:
    """
    Trainer class for deepfake model.

    Handles the complete training pipeline including:
    - Model training
    - Validation
    - Checkpointing
    - Logging

    Args:
        model: DeepfakeModel instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to train on (mps/cuda/cpu)
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs
        use_perceptual: Whether to use perceptual loss (slower but better quality)
    """

    def __init__(
        self,
        model: DeepfakeModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        learning_rate: float = 1e-4,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        use_perceptual: bool = False,  # Disabled by default for speed
    ):
        # Device setup
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        print(f"Training on: {device}")

        # Model
        self.model = model.to(device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.use_perceptual = use_perceptual
        self.criterion = CombinedLoss(
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            use_perceptual=use_perceptual,
        ).to(device)

        if use_perceptual:
            print("Using perceptual loss (slower but better quality)")
        else:
            print("Using L1 loss only (fast mode)")

        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=learning_rate / 10
        )

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/deepfake_{timestamp}")

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {"total": 0, "recon": 0, "perceptual": 0}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (img_a, img_b) in enumerate(pbar):
            # Move to device
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)

            # Forward pass
            recon_a, recon_b = self.model(img_a, img_b)

            # Calculate losses for both persons
            losses_a = self.criterion(recon_a, img_a)
            losses_b = self.criterion(recon_b, img_b)

            # Total loss is sum of both
            total_loss = losses_a["total"] + losses_b["total"]

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["recon"] += (losses_a["recon"] + losses_b["recon"]).item()
            if self.use_perceptual:
                epoch_losses["perceptual"] += (
                    losses_a["perceptual"] + losses_b["perceptual"]
                ).item()

            # Update progress bar
            pbar.set_postfix(
                loss=total_loss.item(),
                recon=(losses_a["recon"] + losses_b["recon"]).item(),
            )

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar("Train/Loss", total_loss.item(), self.global_step)
                self.writer.add_scalar(
                    "Train/Recon", (losses_a["recon"] + losses_b["recon"]).item(), self.global_step
                )

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of average validation losses
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {"total": 0, "recon": 0, "perceptual": 0}
        num_batches = len(self.val_loader)

        for img_a, img_b in self.val_loader:
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)

            recon_a, recon_b = self.model(img_a, img_b)

            losses_a = self.criterion(recon_a, img_a)
            losses_b = self.criterion(recon_b, img_b)

            val_losses["total"] += (losses_a["total"] + losses_b["total"]).item()
            val_losses["recon"] += (losses_a["recon"] + losses_b["recon"]).item()
            if self.use_perceptual:
                val_losses["perceptual"] += (
                    losses_a["perceptual"] + losses_b["perceptual"]
                ).item()

        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    @torch.no_grad()
    def log_images(self, num_images: int = 4):
        """Log sample reconstructions and face swaps to TensorBoard."""
        self.model.eval()

        # Get a batch
        img_a, img_b = next(iter(self.val_loader or self.train_loader))
        img_a = img_a[:num_images].to(self.device)
        img_b = img_b[:num_images].to(self.device)

        # Get all outputs
        outputs = self.model.get_training_outputs(img_a, img_b)

        # Log original images
        self.writer.add_images("Images/Original_A", img_a, self.epoch)
        self.writer.add_images("Images/Original_B", img_b, self.epoch)

        # Log reconstructions
        self.writer.add_images("Images/Recon_A", outputs["recon_a"], self.epoch)
        self.writer.add_images("Images/Recon_B", outputs["recon_b"], self.epoch)

        # Log face swaps (the interesting part!)
        self.writer.add_images("Images/Swap_A_to_B", outputs["swap_a_to_b"], self.epoch)
        self.writer.add_images("Images/Swap_B_to_A", outputs["swap_b_to_a"], self.epoch)

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save epoch checkpoint
        if self.epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch}.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self, num_epochs: int, save_every: int = 10):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\n{'='*50}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*50}\n")

        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.4f}")

            # Validate
            val_losses = self.validate()
            if val_losses:
                print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f}")

                # Log to TensorBoard
                self.writer.add_scalar("Val/Loss", val_losses["total"], epoch)
                self.writer.add_scalar("Val/Recon", val_losses["recon"], epoch)

                # Check for best model
                is_best = val_losses["total"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses["total"]

            # Log images
            if epoch % 5 == 0:
                self.log_images()

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(is_best=is_best if val_losses else False)

            # Update learning rate
            self.scheduler.step()

        # Final save
        self.save_checkpoint()
        self.writer.close()
        print("\nTraining complete!")


if __name__ == "__main__":
    print("Trainer module ready.")
    print("\nUsage:")
    print("  from src.training import DeepfakeTrainer")
    print("  trainer = DeepfakeTrainer(model, train_loader, val_loader)")
    print("  trainer.train(num_epochs=100)")
