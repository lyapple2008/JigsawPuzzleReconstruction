"""Puzzle image splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Patch:
    """A square jigsaw patch with image data and precomputed borders."""

    image: np.ndarray
    original_index: int
    edges: Dict[str, np.ndarray]

    @classmethod
    def from_image(cls, image: np.ndarray, original_index: int) -> "Patch":
        """Create a patch and extract its 4 directional edges."""
        edges = {
            "top": image[0, :, :].astype(np.float32),
            "bottom": image[-1, :, :].astype(np.float32),
            "left": image[:, 0, :].astype(np.float32),
            "right": image[:, -1, :].astype(np.float32),
        }
        return cls(image=image, original_index=original_index, edges=edges)


class PuzzleSplitter:
    """Split an image into a grid of equal-size square patches."""

    def split(self, image: np.ndarray, rows: int, cols: int) -> List[Patch]:
        """Split image into `rows x cols` patches in row-major order."""
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HxWx3 numpy array")

        height, width, _ = image.shape
        if height % rows != 0 or width % cols != 0:
            raise ValueError("image size must be divisible by rows and cols")

        patch_h = height // rows
        patch_w = width // cols

        if patch_h != patch_w:
            raise ValueError("patches must be square; choose rows/cols accordingly")

        patches: List[Patch] = []
        index = 0
        for r in range(rows):
            for c in range(cols):
                y0 = r * patch_h
                y1 = y0 + patch_h
                x0 = c * patch_w
                x1 = x0 + patch_w
                patch_img = image[y0:y1, x0:x1, :].copy()
                patches.append(Patch.from_image(patch_img, original_index=index))
                index += 1
        return patches
