"""Jigsaw puzzle solving algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .matcher import Direction, EdgeMatcher
from .splitter import Patch


@dataclass
class SolverConfig:
    """Configuration for the puzzle solver."""

    rows: int
    cols: int
    seed: int = 42
    local_opt_iters: int = 1000
    row_fill_left_weight: float = 0.25
    use_multi_start: bool = True


class JigsawSolver:
    """Solve shuffled patches using edge compatibility heuristics."""

    def __init__(self, config: SolverConfig) -> None:
        """Initialize solver with reproducible random generator."""
        self.config = config
        self.matcher = EdgeMatcher()
        self.rng = np.random.default_rng(config.seed)

    def solve(self, patches: List[Patch], cost_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a solved grid containing patch indices in current patch list."""
        n = len(patches)
        expected = self.config.rows * self.config.cols
        if n != expected:
            raise ValueError(f"Expected {expected} patches, got {n}")

        cost = cost_matrix if cost_matrix is not None else self.matcher.build_cost_matrix(patches)
        grid = self._build_best_initial_solution(cost)
        if self.config.local_opt_iters > 0:
            grid = self._local_swap_optimize(grid, cost, self.config.local_opt_iters)
        return grid

    def _build_best_initial_solution(self, cost: np.ndarray) -> np.ndarray:
        """Generate one random-seeded solution and optionally improve with multi-start."""
        rows, cols = self.config.rows, self.config.cols
        n = rows * cols

        random_start = int(self.rng.integers(0, n))
        best_grid = self._greedy_initial_solution(cost, start_piece=random_start)
        best_cost = self.matcher.total_grid_cost(best_grid, cost)

        if not self.config.use_multi_start:
            return best_grid

        for start_piece in range(n):
            candidate = self._greedy_initial_solution(cost, start_piece=start_piece)
            candidate_cost = self.matcher.total_grid_cost(candidate, cost)
            if candidate_cost < best_cost:
                best_grid = candidate
                best_cost = candidate_cost
        return best_grid

    def _greedy_initial_solution(self, cost: np.ndarray, start_piece: int) -> np.ndarray:
        """Build an initial arrangement using the required greedy strategy."""
        rows, cols = self.config.rows, self.config.cols
        grid = np.full((rows, cols), -1, dtype=np.int32)
        n = rows * cols
        unused = set(range(n))

        grid[0, 0] = start_piece
        unused.remove(start_piece)

        for c in range(1, cols):
            left = int(grid[0, c - 1])
            candidates = np.array(sorted(unused), dtype=np.int32)
            scores = cost[left, candidates, Direction.RIGHT]
            choice = int(candidates[int(np.argmin(scores))])
            grid[0, c] = choice
            unused.remove(choice)

        for r in range(1, rows):
            for c in range(cols):
                above = int(grid[r - 1, c])
                candidates = np.array(sorted(unused), dtype=np.int32)
                down_scores = cost[above, candidates, Direction.DOWN]
                if c == 0:
                    scores = down_scores
                else:
                    left = int(grid[r, c - 1])
                    right_scores = cost[left, candidates, Direction.RIGHT]
                    scores = down_scores + self.config.row_fill_left_weight * right_scores
                choice = int(candidates[int(np.argmin(scores))])
                grid[r, c] = choice
                unused.remove(choice)

        return grid

    def _local_swap_optimize(self, grid: np.ndarray, cost: np.ndarray, iterations: int) -> np.ndarray:
        """Improve solution by accepting random swaps that reduce global cost."""
        best = grid.copy()
        best_cost = self.matcher.total_grid_cost(best, cost)
        rows, cols = best.shape
        n = rows * cols

        for _ in range(iterations):
            i, j = self.rng.choice(n, size=2, replace=False)
            r1, c1 = divmod(int(i), cols)
            r2, c2 = divmod(int(j), cols)

            candidate = best.copy()
            candidate[r1, c1], candidate[r2, c2] = candidate[r2, c2], candidate[r1, c1]
            candidate_cost = self.matcher.total_grid_cost(candidate, cost)
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
        return best
