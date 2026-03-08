"""Image loading utilities for real image benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """Check if a file is a supported image format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def scan_directory(directory: Path, recursive: bool = False) -> List[Path]:
    """Scan directory for image files.

    Args:
        directory: Path to the directory to scan
        recursive: If True, scan subdirectories recursively

    Returns:
        List of paths to image files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    images: List[Path] = []
    if recursive:
        for root, _, files in os.walk(directory):
            for f in files:
                path = Path(root) / f
                if is_image_file(path):
                    images.append(path)
    else:
        for f in directory.iterdir():
            if f.is_file() and is_image_file(f):
                images.append(f)

    return sorted(images)


def preprocess_image(
    image: np.ndarray,
    target_size: int = 300,
    crop_to_square: bool = True,
) -> np.ndarray:
    """Preprocess image for puzzle reconstruction.

    Args:
        image: Input image in BGR format (OpenCV)
        target_size: Target size for the longer edge
        crop_to_square: If True, center-crop to square before resizing

    Returns:
        Preprocessed image in BGR format
    """
    h, w = image.shape[:2]

    if crop_to_square:
        # Center crop to square
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        image = image[start_h : start_h + size, start_w : start_w + size]

    # Resize to target size
    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return image


class DirectoryImageLoader:
    """Generic directory-based image loader for benchmarks."""

    def __init__(
        self,
        directory: Path,
        target_size: int = 300,
        crop_to_square: bool = True,
        recursive: bool = False,
    ):
        """Initialize the image loader.

        Args:
            directory: Path to directory containing images
            target_size: Target size for the longer edge
            crop_to_square: If True, center-crop to square before resizing
            recursive: If True, scan subdirectories recursively
        """
        self.directory = Path(directory)
        self.target_size = target_size
        self.crop_to_square = crop_to_square
        self.recursive = recursive

        self._image_paths = scan_directory(self.directory, recursive=self.recursive)
        if not self._image_paths:
            raise ValueError(f"No images found in {self.directory}")

    def __len__(self) -> int:
        """Return the number of available images."""
        return len(self._image_paths)

    def get_image_path(self, index: int) -> Path:
        """Get the path to an image by index."""
        return self._image_paths[index]

    def load_image(self, index: int) -> Tuple[np.ndarray, Path]:
        """Load and preprocess an image by index.

        Args:
            index: Image index

        Returns:
            Tuple of (preprocessed image, original path)
        """
        path = self._image_paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"Failed to load image: {path}")

        image = preprocess_image(
            image,
            target_size=self.target_size,
            crop_to_square=self.crop_to_square,
        )
        return image, path

    def iterate(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, Path]]:
        """Iterate over a range of images.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive), None means to the end

        Yields:
            Tuples of (preprocessed image, original path)
        """
        if end is None:
            end = len(self)
        for i in range(start, min(end, len(self))):
            yield self.load_image(i)
