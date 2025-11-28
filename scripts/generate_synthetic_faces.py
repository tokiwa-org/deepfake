#!/usr/bin/env python3
"""
Generate Synthetic Face Data for Testing

This script generates simple synthetic "face-like" images for testing the pipeline.
These are NOT real faces - just colored patterns that simulate face structure.

For actual training, you should use:
1. StyleGAN-generated faces (thispersondoesnotexist.com style)
2. CelebA dataset (with proper attribution)
3. Your own photos (with consent)

Usage:
    python scripts/generate_synthetic_faces.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import random


def generate_synthetic_face(
    size: int = 256,
    skin_color: tuple = None,
    hair_color: tuple = None,
    eye_color: tuple = None,
    seed: int = None,
) -> Image.Image:
    """
    Generate a simple synthetic face-like image.
    
    This creates abstract face patterns, NOT photorealistic faces.
    Useful for testing the training pipeline.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Random colors if not specified
    if skin_color is None:
        # Various skin tone ranges
        base = random.randint(180, 240)
        skin_color = (base, base - random.randint(20, 40), base - random.randint(30, 60))
    
    if hair_color is None:
        hair_type = random.choice(['dark', 'brown', 'blonde', 'red'])
        if hair_type == 'dark':
            hair_color = (random.randint(20, 50), random.randint(15, 40), random.randint(10, 30))
        elif hair_type == 'brown':
            hair_color = (random.randint(80, 120), random.randint(50, 80), random.randint(30, 50))
        elif hair_type == 'blonde':
            hair_color = (random.randint(200, 230), random.randint(180, 210), random.randint(100, 150))
        else:  # red
            hair_color = (random.randint(150, 200), random.randint(50, 80), random.randint(30, 50))
    
    if eye_color is None:
        eye_type = random.choice(['brown', 'blue', 'green', 'gray'])
        if eye_type == 'brown':
            eye_color = (random.randint(60, 100), random.randint(40, 70), random.randint(20, 40))
        elif eye_type == 'blue':
            eye_color = (random.randint(50, 100), random.randint(100, 150), random.randint(180, 220))
        elif eye_type == 'green':
            eye_color = (random.randint(50, 100), random.randint(120, 160), random.randint(60, 100))
        else:  # gray
            eye_color = (random.randint(100, 140), random.randint(100, 140), random.randint(110, 150))
    
    # Create base image with background
    bg_color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    center_x = size // 2
    center_y = size // 2
    
    # Draw hair (top of head)
    hair_top = int(size * 0.1)
    hair_width = int(size * 0.4)
    draw.ellipse([
        center_x - hair_width, hair_top,
        center_x + hair_width, center_y + int(size * 0.2)
    ], fill=hair_color)
    
    # Draw face (oval)
    face_width = int(size * 0.32)
    face_height = int(size * 0.38)
    face_top = int(size * 0.22)
    draw.ellipse([
        center_x - face_width, face_top,
        center_x + face_width, face_top + face_height * 2
    ], fill=skin_color)
    
    # Draw eyes
    eye_y = int(size * 0.42)
    eye_spacing = int(size * 0.12)
    eye_size = int(size * 0.06)
    
    # Eye whites
    for offset in [-eye_spacing, eye_spacing]:
        draw.ellipse([
            center_x + offset - eye_size, eye_y - eye_size // 2,
            center_x + offset + eye_size, eye_y + eye_size // 2
        ], fill=(255, 255, 255))
    
    # Eye pupils
    pupil_size = eye_size // 2
    # Random gaze direction
    gaze_x = random.randint(-2, 2)
    gaze_y = random.randint(-1, 1)
    for offset in [-eye_spacing, eye_spacing]:
        draw.ellipse([
            center_x + offset - pupil_size + gaze_x, eye_y - pupil_size // 2 + gaze_y,
            center_x + offset + pupil_size + gaze_x, eye_y + pupil_size // 2 + gaze_y
        ], fill=eye_color)
    
    # Draw nose (simple triangle/line)
    nose_top = int(size * 0.48)
    nose_bottom = int(size * 0.58)
    nose_width = int(size * 0.03)
    draw.polygon([
        (center_x, nose_top),
        (center_x - nose_width, nose_bottom),
        (center_x + nose_width, nose_bottom)
    ], fill=tuple(max(0, c - 20) for c in skin_color))
    
    # Draw mouth
    mouth_y = int(size * 0.68)
    mouth_width = int(size * 0.1)
    mouth_height = int(size * 0.02)
    
    # Random expression
    expression = random.choice(['neutral', 'smile', 'slight_smile'])
    if expression == 'smile':
        draw.arc([
            center_x - mouth_width, mouth_y - mouth_height * 2,
            center_x + mouth_width, mouth_y + mouth_height * 4
        ], start=0, end=180, fill=(180, 100, 100), width=2)
    elif expression == 'slight_smile':
        draw.arc([
            center_x - mouth_width, mouth_y - mouth_height,
            center_x + mouth_width, mouth_y + mouth_height * 2
        ], start=0, end=180, fill=(180, 100, 100), width=2)
    else:
        draw.line([
            (center_x - mouth_width, mouth_y),
            (center_x + mouth_width, mouth_y)
        ], fill=(180, 100, 100), width=2)
    
    # Draw eyebrows
    brow_y = int(size * 0.35)
    brow_width = int(size * 0.08)
    brow_height = int(size * 0.01)
    for offset in [-eye_spacing, eye_spacing]:
        # Random eyebrow angle
        angle_offset = random.randint(-2, 2)
        draw.line([
            (center_x + offset - brow_width, brow_y + angle_offset),
            (center_x + offset + brow_width, brow_y - angle_offset)
        ], fill=hair_color, width=2)
    
    # Add some blur for smoothness
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Add slight noise for texture
    img_array = np.array(img)
    noise = np.random.normal(0, 3, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img


def generate_person_dataset(
    output_dir: str,
    num_images: int = 100,
    base_skin_color: tuple = None,
    base_hair_color: tuple = None,
    base_eye_color: tuple = None,
    person_seed: int = None,
):
    """
    Generate a dataset of synthetic faces for one "person".
    
    The faces will have consistent base features (skin, hair, eye color)
    but vary in expression and slight details.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if person_seed is not None:
        random.seed(person_seed)
        np.random.seed(person_seed)
    
    # Set consistent base colors for this "person"
    if base_skin_color is None:
        base = random.randint(180, 240)
        base_skin_color = (base, base - random.randint(20, 40), base - random.randint(30, 60))
    
    if base_hair_color is None:
        hair_type = random.choice(['dark', 'brown', 'blonde', 'red'])
        if hair_type == 'dark':
            base_hair_color = (random.randint(20, 50), random.randint(15, 40), random.randint(10, 30))
        elif hair_type == 'brown':
            base_hair_color = (random.randint(80, 120), random.randint(50, 80), random.randint(30, 50))
        elif hair_type == 'blonde':
            base_hair_color = (random.randint(200, 230), random.randint(180, 210), random.randint(100, 150))
        else:
            base_hair_color = (random.randint(150, 200), random.randint(50, 80), random.randint(30, 50))
    
    if base_eye_color is None:
        eye_type = random.choice(['brown', 'blue', 'green', 'gray'])
        if eye_type == 'brown':
            base_eye_color = (random.randint(60, 100), random.randint(40, 70), random.randint(20, 40))
        elif eye_type == 'blue':
            base_eye_color = (random.randint(50, 100), random.randint(100, 150), random.randint(180, 220))
        elif eye_type == 'green':
            base_eye_color = (random.randint(50, 100), random.randint(120, 160), random.randint(60, 100))
        else:
            base_eye_color = (random.randint(100, 140), random.randint(100, 140), random.randint(110, 150))
    
    print(f"Generating {num_images} images for person in {output_dir}")
    print(f"  Skin color: {base_skin_color}")
    print(f"  Hair color: {base_hair_color}")
    print(f"  Eye color: {base_eye_color}")
    
    for i in range(num_images):
        # Add slight variation to colors
        skin_var = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in base_skin_color)
        hair_var = tuple(max(0, min(255, c + random.randint(-5, 5))) for c in base_hair_color)
        eye_var = tuple(max(0, min(255, c + random.randint(-5, 5))) for c in base_eye_color)
        
        img = generate_synthetic_face(
            size=256,
            skin_color=skin_var,
            hair_color=hair_var,
            eye_color=eye_var,
        )
        
        img.save(output_path / f"face_{i:04d}.jpg", quality=95)
    
    print(f"  Saved {num_images} images")


def main():
    """Generate synthetic datasets for two persons."""
    print("=" * 60)
    print("Generating Synthetic Face Datasets")
    print("=" * 60)
    print("\nNote: These are abstract face patterns, not photorealistic.")
    print("Use for testing the training pipeline.\n")
    
    # Person A - lighter features
    generate_person_dataset(
        output_dir="data/person_a",
        num_images=200,
        person_seed=42,
    )
    
    print()
    
    # Person B - different features
    generate_person_dataset(
        output_dir="data/person_b",
        num_images=200,
        person_seed=123,
    )
    
    print("\n" + "=" * 60)
    print("Done! Datasets created in data/person_a and data/person_b")
    print("=" * 60)
    print("\nTo train with this data:")
    print("  python train.py --data-a data/person_a --data-b data/person_b --epochs 50")


if __name__ == "__main__":
    main()
