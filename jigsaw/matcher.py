"""Edge matching and pairwise cost matrix computation."""

from __future__ import annotations

from enum import IntEnum
from typing import List

import numpy as np

from .splitter import Patch


class Direction(IntEnum):
    """Supported relative directions between neighboring pieces."""

    RIGHT = 0
    DOWN = 1


class EdgeMatcher:
    """Compute edge compatibility scores for jigsaw patches."""

    def __init__(
        self,
        strip_width: int = 3,
        color_weight: float = 0.85,
        gradient_weight: float = 0.15,
    ) -> None:
        """Initialize matcher parameters for robust edge comparison."""
        self.strip_width = max(1, int(strip_width))
        self.color_weight = float(color_weight)
        self.gradient_weight = float(gradient_weight)

    def _get_strip(self, patch: Patch, side: str) -> np.ndarray:
        """Return edge-adjacent strip with seam-aligned orientation."""
        img = patch.image.astype(np.float32)
        h, w, _ = img.shape
        width = min(self.strip_width, h if side in {"top", "bottom"} else w)
        if side == "right":
            # Flip so seam is at index 0 and inward direction is positive.
            return np.flip(img[:, w - width : w, :], axis=1)
        if side == "left":
            return img[:, :width, :]
        if side == "bottom":
            return np.flip(img[h - width : h, :, :], axis=0)
        if side == "top":
            return img[:width, :, :]
        raise ValueError(f"Unsupported side: {side}")

    @staticmethod
    def _safe_mean_sq(diff: np.ndarray) -> float:
        return float(np.mean(diff * diff)) if diff.size else 0.0

    def _color_strip_distance(self, patch_a: Patch, patch_b: Patch, direction: Direction) -> float:
        """Compute strip-based color distance for adjacency direction."""
        if direction == Direction.RIGHT:
            strip_a = self._get_strip(patch_a, "right")
            strip_b = self._get_strip(patch_b, "left")
        elif direction == Direction.DOWN:
            strip_a = self._get_strip(patch_a, "bottom")
            strip_b = self._get_strip(patch_b, "top")
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        return self._safe_mean_sq(strip_a - strip_b)

    def _seam_gradient_distance(self, patch_a: Patch, patch_b: Patch, direction: Direction) -> float:
        """Compare first-order gradients at seam to penalize edge discontinuities."""
        a = patch_a.image.astype(np.float32)
        b = patch_b.image.astype(np.float32)

        if direction == Direction.RIGHT:
            if a.shape[1] < 2 or b.shape[1] < 2:
                return 0.0
            grad_a = a[:, -1, :] - a[:, -2, :]
            grad_b = b[:, 1, :] - b[:, 0, :]
        elif direction == Direction.DOWN:
            if a.shape[0] < 2 or b.shape[0] < 2:
                return 0.0
            grad_a = a[-1, :, :] - a[-2, :, :]
            grad_b = b[1, :, :] - b[0, :, :]
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return self._safe_mean_sq(grad_a - grad_b)

    def edge_distance(
        self,
        patch_a: Patch,
        patch_b: Patch,
        direction: Direction,
        normalize: bool = False,
    ) -> float:
        """Return fused edge compatibility distance for directional adjacency."""
        if direction == Direction.RIGHT:
            edge_a = patch_a.edges["right"]
            edge_b = patch_b.edges["left"]
        elif direction == Direction.DOWN:
            edge_a = patch_a.edges["bottom"]
            edge_b = patch_b.edges["top"]
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        edge_dist = float(np.sum((edge_a - edge_b) ** 2))
        strip_dist = self._color_strip_distance(patch_a, patch_b, direction)
        grad_dist = self._seam_gradient_distance(patch_a, patch_b, direction)
        fused = self.color_weight * strip_dist + self.gradient_weight * grad_dist
        dist = 0.5 * edge_dist + 0.5 * fused
        if normalize:
            denom = float(edge_a.size) if edge_a.size else 1.0
            return dist / denom
        return dist

    def build_cost_matrix(self, patches: List[Patch], normalize: bool = False) -> np.ndarray:
        """Build full pairwise directional cost matrix: [i, j, direction]."""
        n = len(patches)
        cost = np.full((n, n, 2), np.inf, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cost[i, j, Direction.RIGHT] = self.edge_distance(
                    patches[i], patches[j], Direction.RIGHT, normalize=normalize
                )
                cost[i, j, Direction.DOWN] = self.edge_distance(
                    patches[i], patches[j], Direction.DOWN, normalize=normalize
                )
        return cost

    def total_grid_cost(self, grid: np.ndarray, cost: np.ndarray) -> float:
        """Compute sum of all right/down adjacency costs in a solved grid."""
        rows, cols = grid.shape
        total = 0.0
        for r in range(rows):
            for c in range(cols):
                cur = int(grid[r, c])
                if c + 1 < cols:
                    right = int(grid[r, c + 1])
                    total += float(cost[cur, right, Direction.RIGHT])
                if r + 1 < rows:
                    down = int(grid[r + 1, c])
                    total += float(cost[cur, down, Direction.DOWN])
        return total
