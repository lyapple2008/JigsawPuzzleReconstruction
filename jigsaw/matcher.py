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

    def edge_distance(
        self,
        patch_a: Patch,
        patch_b: Patch,
        direction: Direction,
        normalize: bool = False,
    ) -> float:
        """Return L2 edge distance for a directional adjacency."""
        if direction == Direction.RIGHT:
            edge_a = patch_a.edges["right"]
            edge_b = patch_b.edges["left"]
        elif direction == Direction.DOWN:
            edge_a = patch_a.edges["bottom"]
            edge_b = patch_b.edges["top"]
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        dist = float(np.sum((edge_a - edge_b) ** 2))
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
