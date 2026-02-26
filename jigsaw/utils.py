"""Utility helpers for reproducible jigsaw experiments."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .splitter import Patch

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def set_random_seed(seed: int = 42) -> np.random.Generator:
    """Create a deterministic numpy random generator."""
    return np.random.default_rng(seed)


def shuffle_patches(
    patches: List[Patch], seed: int = 42
) -> Tuple[List[Patch], np.ndarray]:
    """Return a shuffled copy of patches and the applied permutation."""
    rng = set_random_seed(seed)
    order = rng.permutation(len(patches))
    return [patches[i] for i in order], order


def compose_image_from_grid(grid: np.ndarray, patches: List[Patch]) -> np.ndarray:
    """Reconstruct full image from a grid of patch indices."""
    rows, cols = grid.shape
    patch_h, patch_w, channels = patches[0].image.shape
    canvas = np.zeros((rows * patch_h, cols * patch_w, channels), dtype=patches[0].image.dtype)
    for r in range(rows):
        for c in range(cols):
            idx = int(grid[r, c])
            y0 = r * patch_h
            x0 = c * patch_w
            canvas[y0 : y0 + patch_h, x0 : x0 + patch_w, :] = patches[idx].image
    return canvas


def compose_image_row_major(patches: List[Patch], rows: int, cols: int) -> np.ndarray:
    """Compose image by placing list-order patches in row-major coordinates."""
    grid = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    return compose_image_from_grid(grid, patches)


def generate_random_image(size: int = 300, seed: int = 42) -> np.ndarray:
    """Generate a purely random RGB image."""
    rng = set_random_seed(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def generate_gradient_image(size: int = 300) -> np.ndarray:
    """Generate a smooth RGB gradient image."""
    x = np.linspace(0, 255, size, dtype=np.float32)
    y = np.linspace(0, 255, size, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    r = xv
    g = yv
    b = 0.5 * (xv + yv)
    img = np.stack([r, g, b], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_natural_like_image(size: int = 300, seed: int = 42) -> np.ndarray:
    """Generate a deterministic texture-rich image resembling a natural scene."""
    rng = set_random_seed(seed)
    base = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    if cv2 is not None:
        smooth = cv2.GaussianBlur(base, (0, 0), sigmaX=6, sigmaY=6)
        detail = cv2.Canny(smooth, 60, 120)
        detail_rgb = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(smooth, 0.85, detail_rgb, 0.15, 0)

    base_f = base.astype(np.float32)
    padded = np.pad(base_f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    smooth = (
        padded[1:-1, 1:-1, :]
        + padded[:-2, 1:-1, :]
        + padded[2:, 1:-1, :]
        + padded[1:-1, :-2, :]
        + padded[1:-1, 2:, :]
    ) / 5.0
    dx = np.zeros_like(smooth)
    dx[:, 1:, :] = np.abs(smooth[:, 1:, :] - smooth[:, :-1, :])
    edges = dx.mean(axis=2, keepdims=True)
    edges = np.clip(edges * 2.0, 0, 255)
    blended = 0.85 * smooth + 0.15 * np.repeat(edges, 3, axis=2)
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_or_generate_image(path: Optional[str], size: int = 300, seed: int = 42) -> np.ndarray:
    """Load image from path, or generate a natural-like fallback image."""
    if path:
        if cv2 is not None:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image from path: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] != size or image.shape[1] != size:
                image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            return image

        import matplotlib.image as mpimg

        image = mpimg.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {path}")
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if image.shape[0] != size or image.shape[1] != size:
            y_idx = np.linspace(0, image.shape[0] - 1, size).astype(np.int32)
            x_idx = np.linspace(0, image.shape[1] - 1, size).astype(np.int32)
            image = image[y_idx][:, x_idx]
        return image
    return generate_natural_like_image(size=size, seed=seed)
