#!/usr/bin/env python3
"""
Main Training Script for Deepfake Model (Lightweight Version)

Usage:
    # Fast training (L1 loss only, ~3-5 sec/batch)
    python train.py --data-a data/person_a --data-b data/person_b

    # Quality training (with perceptual loss, slower)
    python train.py --data-a data/person_a --data-b data/person_b --perceptual

This script:
1. Loads face images for two persons
2. Creates the DeepfakeModel (lightweight: 128x128, ~20M params)
3. Trains using the specified configuration
4. Saves checkpoints and logs to TensorBoard

For M4 Mac:
- Uses MPS (Metal Performance Shaders) for GPU acceleration
- Optimized batch size for unified memory
- Default: L1 loss only for fast iteration
"""

import argparse
from pathlib import Path

import torch

from src.models import DeepfakeModel
from src.data import create_dataloaders
from src.training import DeepfakeTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Deepfake model (lightweight version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--data-a",
        type=str,
        required=True,
        help="Directory containing person A's face images",
    )
    parser.add_argument(
        "--data-b",
        type=str,
        required=True,
        help="Directory containing person B's face images",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,  # Increased for smaller model
        help="Batch size (16-32 recommended for lightweight model)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,  # Reduced from 256 for speed
        help="Image size (128 for fast training, 256 for quality)",
    )

    # Loss weights
    parser.add_argument(
        "--recon-weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--perceptual-weight",
        type=float,
        default=0.1,
        help="Weight for perceptual loss (only used with --perceptual)",
    )
    parser.add_argument(
        "--perceptual",
        action="store_true",
        help="Enable perceptual loss (slower but better quality)",
    )

    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def get_device() -> torch.device:
    """Get the best available device for M4 Mac."""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) - M4 Mac GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU (training will be slow)")
        return torch.device("cpu")


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("Deepfake Training - Educational Project (Lightweight)")
    print("=" * 60)

    # Check data directories
    data_a = Path(args.data_a)
    data_b = Path(args.data_b)

    if not data_a.exists():
        print(f"Error: Data directory for person A not found: {data_a}")
        print("Please create this directory and add face images.")
        return

    if not data_b.exists():
        print(f"Error: Data directory for person B not found: {data_b}")
        print("Please create this directory and add face images.")
        return

    # Get device
    device = get_device()

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        dir_a=str(data_a),
        dir_b=str(data_b),
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=0,  # MPS doesn't support multiprocessing well
    )

    # Create model (lightweight: 128 latent dim)
    print("\nCreating model...")
    model = DeepfakeModel(
        input_size=args.image_size,
        latent_dim=128,  # Reduced from 512
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        recon_weight=args.recon_weight,
        perceptual_weight=args.perceptual_weight,
        use_perceptual=args.perceptual,  # Disabled by default
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train!
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Perceptual loss: {'Enabled' if args.perceptual else 'Disabled (fast mode)'}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  TensorBoard: {args.log_dir}")
    print()

    trainer.train(num_epochs=args.epochs)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print(f"\nTo use the trained model:")
    print(f"  from src.models import DeepfakeModel")
    print(f"  model = DeepfakeModel()")
    print(f"  model.load_state_dict(torch.load('{args.checkpoint_dir}/best.pt')['model_state_dict'])")


if __name__ == "__main__":
    main()
