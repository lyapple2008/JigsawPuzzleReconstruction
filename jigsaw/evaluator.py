"""Evaluation metrics for jigsaw reconstruction quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .matcher import EdgeMatcher
from .splitter import Patch


@dataclass
class EvaluationResult:
    """Container for reconstruction metrics."""

    position_accuracy: float
    neighbor_accuracy: float
    total_cost: float


class PuzzleEvaluator:
    """Compute core quality metrics for reconstructed puzzles."""

    def __init__(self) -> None:
        """Initialize evaluator dependencies."""
        self.matcher = EdgeMatcher()

    def compute_position_accuracy(self, grid: np.ndarray, patches: List[Patch]) -> float:
        """Fraction of patches restored to their original coordinates."""
        rows, cols = grid.shape
        correct = 0
        total = rows * cols
        for r in range(rows):
            for c in range(cols):
                expected = r * cols + c
                patch_index = int(grid[r, c])
                if patches[patch_index].original_index == expected:
                    correct += 1
        return correct / total if total else 0.0

    def compute_neighbor_accuracy(self, grid: np.ndarray, patches: List[Patch]) -> float:
        """Fraction of right/down neighbors matching original adjacency."""
        rows, cols = grid.shape
        correct = 0
        total = 0

        for r in range(rows):
            for c in range(cols):
                cur_patch = patches[int(grid[r, c])]
                cur_idx = cur_patch.original_index
                cur_row, cur_col = divmod(cur_idx, cols)

                if c + 1 < cols:
                    total += 1
                    right_patch = patches[int(grid[r, c + 1])]
                    right_idx = right_patch.original_index
                    if right_idx == cur_row * cols + (cur_col + 1):
                        correct += 1

                if r + 1 < rows:
                    total += 1
                    down_patch = patches[int(grid[r + 1, c])]
                    down_idx = down_patch.original_index
                    if down_idx == (cur_row + 1) * cols + cur_col:
                        correct += 1

        return correct / total if total else 0.0

    def evaluate(self, grid: np.ndarray, patches: List[Patch], cost_matrix: np.ndarray) -> EvaluationResult:
        """Calculate all required metrics for a reconstructed puzzle."""
        return EvaluationResult(
            position_accuracy=self.compute_position_accuracy(grid, patches),
            neighbor_accuracy=self.compute_neighbor_accuracy(grid, patches),
            total_cost=self.matcher.total_grid_cost(grid, cost_matrix),
        )
