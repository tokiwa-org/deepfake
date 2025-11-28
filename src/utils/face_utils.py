"""
Face Detection and Alignment Utilities

For deepfake training, we need:
1. Face detection: Find faces in images
2. Face alignment: Rotate/scale so eyes are horizontal and centered
3. Face extraction: Crop the face region

This ensures consistent input to the model.

Note: For educational purposes, we use a simple approach.
Production systems use more sophisticated alignment.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import torch


def detect_faces(
    image: np.ndarray,
    min_face_size: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using OpenCV's Haar Cascade.

    This is a simple approach for educational purposes.
    Production systems typically use:
    - MTCNN
    - RetinaFace
    - dlib's HOG or CNN detector

    Args:
        image: BGR image array (OpenCV format)
        min_face_size: Minimum face size to detect

    Returns:
        List of (x, y, w, h) tuples for each detected face
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load cascade classifier
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )

    return [(x, y, w, h) for (x, y, w, h) in faces]


def align_face(
    image: np.ndarray,
    face_rect: Tuple[int, int, int, int],
    target_size: int = 256,
    padding: float = 0.3,
) -> np.ndarray:
    """
    Extract and align a face from an image.

    Simple alignment by:
    1. Expanding bounding box with padding
    2. Cropping the region
    3. Resizing to target size

    More sophisticated alignment would:
    - Detect facial landmarks
    - Calculate rotation angle from eye positions
    - Apply affine transformation

    Args:
        image: BGR image array
        face_rect: (x, y, w, h) face bounding box
        target_size: Output size (square)
        padding: Extra padding around face (0.3 = 30%)

    Returns:
        Aligned face image of shape (target_size, target_size, 3)
    """
    x, y, w, h = face_rect
    img_h, img_w = image.shape[:2]

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    # Calculate new bounding box with padding
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    # Crop face region
    face = image[y1:y2, x1:x2]

    # Resize to target size
    face_resized = cv2.resize(face, (target_size, target_size))

    return face_resized


def extract_face(
    image_path: str,
    target_size: int = 256,
    padding: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Extract the largest face from an image file.

    This is a convenience function that:
    1. Loads the image
    2. Detects faces
    3. Extracts and aligns the largest face

    Args:
        image_path: Path to image file
        target_size: Output size (square)
        padding: Extra padding around face

    Returns:
        Aligned face image or None if no face detected
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Detect faces
    faces = detect_faces(image)
    if not faces:
        print(f"No faces detected in: {image_path}")
        return None

    # Get largest face (by area)
    largest_face = max(faces, key=lambda f: f[2] * f[3])

    # Align and extract
    aligned = align_face(image, largest_face, target_size, padding)

    return aligned


def batch_extract_faces(
    image_paths: List[str],
    output_dir: str,
    target_size: int = 256,
) -> int:
    """
    Extract faces from multiple images and save to output directory.

    Args:
        image_paths: List of image file paths
        output_dir: Directory to save extracted faces
        target_size: Output size

    Returns:
        Number of successfully extracted faces
    """
    import os
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for i, img_path in enumerate(image_paths):
        face = extract_face(img_path, target_size)
        if face is not None:
            output_file = output_path / f"face_{i:04d}.jpg"
            cv2.imwrite(str(output_file), face)
            success_count += 1

    print(f"Extracted {success_count}/{len(image_paths)} faces to {output_dir}")
    return success_count


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert OpenCV BGR image to PyTorch tensor."""
    # BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # HWC to CHW, scale to [0, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to OpenCV BGR image."""
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    # CHW to HWC, scale to [0, 255]
    image = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # RGB to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr


if __name__ == "__main__":
    print("Face utilities module ready.")
    print("\nUsage:")
    print("  from src.utils import extract_face, batch_extract_faces")
    print("  face = extract_face('photo.jpg')")
    print("  batch_extract_faces(['img1.jpg', 'img2.jpg'], 'output/')")
