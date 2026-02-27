"""Jigsaw puzzle solving algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

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
    use_beam_init: bool = True
    beam_width: int = 6
    beam_candidate_pool: int = 12
    max_start_pieces: int = 12
    sa_initial_temp_ratio: float = 0.08
    sa_cooling: float = 0.995
    border_bonus_weight: float = 0.15


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
        border_scores = self._compute_border_scores(cost)
        grid = self._build_best_initial_solution(cost, border_scores)
        if self.config.local_opt_iters > 0:
            grid = self._local_swap_optimize(grid, cost, self.config.local_opt_iters)
        return grid

    def _build_best_initial_solution(self, cost: np.ndarray, border_scores: np.ndarray) -> np.ndarray:
        """Generate one random-seeded solution and optionally improve with multi-start."""
        rows, cols = self.config.rows, self.config.cols
        n = rows * cols

        start_pieces = self._select_start_pieces(n, border_scores)
        first_start = start_pieces[0]
        if self.config.use_beam_init:
            best_grid = self._beam_initial_solution(
                cost, border_scores=border_scores, start_piece=first_start
            )
        else:
            best_grid = self._greedy_initial_solution(cost, start_piece=first_start)
        best_cost = self.matcher.total_grid_cost(best_grid, cost)

        if not self.config.use_multi_start:
            return best_grid

        for start_piece in start_pieces[1:]:
            if self.config.use_beam_init:
                candidate = self._beam_initial_solution(
                    cost, border_scores=border_scores, start_piece=start_piece
                )
            else:
                candidate = self._greedy_initial_solution(cost, start_piece=start_piece)
            candidate_cost = self.matcher.total_grid_cost(candidate, cost)
            if candidate_cost < best_cost:
                best_grid = candidate
                best_cost = candidate_cost
        return best_grid

    def _select_start_pieces(self, n: int, border_scores: np.ndarray) -> List[int]:
        """Select start-piece candidates for multi-start without blowing up runtime."""
        # For row-major fill from top-left, prioritize pieces likely to be TL corners.
        tl_corner_score = border_scores[:, 0] + border_scores[:, 2]
        ranked = np.argsort(-tl_corner_score)

        if not self.config.use_multi_start:
            return [int(ranked[0])]
        if n <= self.config.max_start_pieces:
            return [int(x) for x in ranked.tolist()]
        return [int(x) for x in ranked[: self.config.max_start_pieces].tolist()]

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

    def _compute_border_scores(self, cost: np.ndarray) -> np.ndarray:
        """Estimate how likely each piece side belongs to outer boundary."""
        n = cost.shape[0]
        scores = np.zeros((n, 4), dtype=np.float64)  # left, right, top, bottom

        # Right/Down use forward costs directly.
        scores[:, 1] = np.min(cost[:, :, Direction.RIGHT], axis=1)
        scores[:, 3] = np.min(cost[:, :, Direction.DOWN], axis=1)
        # Left/Top use inverse directional costs from other pieces.
        scores[:, 0] = np.min(cost[:, :, Direction.RIGHT], axis=0)
        scores[:, 2] = np.min(cost[:, :, Direction.DOWN], axis=0)

        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        if sigma > 1e-9:
            scores = (scores - mu) / sigma
        else:
            scores = scores - mu
        return scores

    def _boundary_bonus(
        self, border_scores: np.ndarray, piece: int, r: int, c: int, rows: int, cols: int
    ) -> float:
        """Return bonus for placing likely boundary sides on puzzle boundary."""
        bonus = 0.0
        if c == 0:
            bonus += float(border_scores[piece, 0])
        if c == cols - 1:
            bonus += float(border_scores[piece, 1])
        if r == 0:
            bonus += float(border_scores[piece, 2])
        if r == rows - 1:
            bonus += float(border_scores[piece, 3])
        return self.config.border_bonus_weight * bonus

    def _beam_initial_solution(
        self, cost: np.ndarray, border_scores: np.ndarray, start_piece: int
    ) -> np.ndarray:
        """Build initial arrangement with beam search over row-major placement."""
        rows, cols = self.config.rows, self.config.cols
        n = rows * cols

        grid0 = np.full((rows, cols), -1, dtype=np.int32)
        grid0[0, 0] = start_piece
        beams: List[Tuple[float, np.ndarray, Set[int]]] = [(0.0, grid0, set(range(n)) - {start_piece})]

        for pos in range(1, n):
            r, c = divmod(pos, cols)
            expanded: List[Tuple[float, np.ndarray, Set[int]]] = []
            for score, grid, unused in beams:
                candidates = self._rank_candidates(cost, border_scores, grid, unused, r, c)
                for piece, local_cost in candidates[: self.config.beam_candidate_pool]:
                    next_grid = grid.copy()
                    next_grid[r, c] = piece
                    next_unused = set(unused)
                    next_unused.remove(piece)
                    expanded.append((score + local_cost, next_grid, next_unused))

            if not expanded:
                break
            expanded.sort(key=lambda x: x[0])
            beams = expanded[: self.config.beam_width]

        return beams[0][1]

    def _rank_candidates(
        self,
        cost: np.ndarray,
        border_scores: np.ndarray,
        grid: np.ndarray,
        unused: Set[int],
        r: int,
        c: int,
    ) -> List[Tuple[int, float]]:
        """Rank candidate pieces by local left/up compatibility score."""
        scores: List[Tuple[int, float]] = []
        up = int(grid[r - 1, c]) if r > 0 else None
        left = int(grid[r, c - 1]) if c > 0 else None

        rows, cols = grid.shape
        for piece in unused:
            value = 0.0
            if up is not None:
                value += float(cost[up, piece, Direction.DOWN])
            if left is not None:
                value += self.config.row_fill_left_weight * float(cost[left, piece, Direction.RIGHT])
            value -= self._boundary_bonus(border_scores, piece, r, c, rows, cols)
            scores.append((piece, value))
        scores.sort(key=lambda x: x[1])
        return scores

    def _local_swap_optimize(self, grid: np.ndarray, cost: np.ndarray, iterations: int) -> np.ndarray:
        """Improve solution with simulated annealing swaps to escape local minima."""
        current = grid.copy()
        current_cost = self.matcher.total_grid_cost(current, cost)
        best = current.copy()
        best_cost = current_cost

        rows, cols = current.shape
        n = rows * cols
        finite = cost[np.isfinite(cost)]
        base_scale = float(np.mean(finite)) if finite.size else 1.0
        temp = max(1e-6, self.config.sa_initial_temp_ratio * base_scale)

        for _ in range(iterations):
            i, j = self.rng.choice(n, size=2, replace=False)
            r1, c1 = divmod(int(i), cols)
            r2, c2 = divmod(int(j), cols)

            candidate = current.copy()
            candidate[r1, c1], candidate[r2, c2] = candidate[r2, c2], candidate[r1, c1]
            candidate_cost = self.matcher.total_grid_cost(candidate, cost)
            delta = candidate_cost - current_cost

            if delta < 0 or float(self.rng.random()) < float(np.exp(-delta / max(temp, 1e-6))):
                current = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

            temp *= self.config.sa_cooling
            if temp < 1e-9:
                temp = 1e-9

        return best
