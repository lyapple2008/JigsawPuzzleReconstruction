"""Gymnasium environment for jigsaw puzzle solving via piece swaps.

The agent learns to swap pairs of puzzle pieces to reconstruct the original image.
Observation includes the current grid state and pre-computed edge cost matrix.
Action is a single integer encoding a swap pair (i, j) where i < j.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import shuffle_patches


class PuzzleSwapEnv(gym.Env):
    """Jigsaw puzzle environment where the agent swaps piece pairs.

    Observation Space:
        Dict:
        - "grid": int32 array of shape (n,) — flat grid of piece indices
          grid[row*col + col] = piece_index at that position
        - "edge_costs": float32 array of shape (n, n, 2) — pre-computed
          directional edge costs [i, j, direction]

    Action Space:
        Discrete(n*(n-1)//2) — index into upper-triangle swap pairs.
        Decoded as (pos_a, pos_b) where pos_a < pos_b in row-major order.

    Reward:
        improvement in total edge cost + accuracy bonus + solve bonus - step penalty

    Episode:
        Starts with shuffled puzzle.
        Terminates when all pieces are in correct positions.
        Truncates at max_steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        image: np.ndarray,
        rows: int = 6,
        cols: int = 6,
        max_steps: int = 500,
        seed: int = 42,
        render_mode: Optional[str] = None,
        cost_improvement_weight: float = 1.0,
        accuracy_bonus_weight: float = 10.0,
        solve_bonus: float = 100.0,
        step_penalty: float = 0.01,
    ) -> None:
        """Initialize the puzzle environment.

        Args:
            image: Input image as HxWx3 uint8 numpy array.
            rows: Number of grid rows.
            cols: Number of grid columns.
            max_steps: Maximum steps per episode before truncation.
            seed: Random seed for reproducibility.
            render_mode: Render mode ("human", "rgb_array", or None).
            cost_improvement_weight: Weight for cost delta in reward.
            accuracy_bonus_weight: Weight for accuracy improvement in reward.
            solve_bonus: Bonus reward for completing the puzzle.
            step_penalty: Small penalty per step to encourage efficiency.
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._seed = seed

        # Reward weights
        self.cost_improvement_weight = cost_improvement_weight
        self.accuracy_bonus_weight = accuracy_bonus_weight
        self.solve_bonus = solve_bonus
        self.step_penalty = step_penalty

        # Split image into patches (once)
        splitter = PuzzleSplitter()
        self.patches: list = splitter.split(image, rows, cols)

        # Evaluator (uses raw cost matrix for total_grid_cost)
        self.evaluator = PuzzleEvaluator()

        # Pre-compute cost matrix (once, expensive O(n^2))
        matcher = EdgeMatcher()
        raw_cost: np.ndarray = matcher.build_cost_matrix(self.patches)

        # Replace inf with large finite value and normalize to [0, 1] range
        finite_vals = raw_cost[np.isfinite(raw_cost)]
        self._cost_max = float(finite_vals.max()) if finite_vals.size > 0 else 1.0
        self._cost_fill = self._cost_max * 10.0  # value for inf entries
        self.cost_matrix = np.where(np.isfinite(raw_cost), raw_cost, self._cost_fill)
        # Normalize to [0, 1]
        if self._cost_max > 0:
            self.cost_matrix = self.cost_matrix / self._cost_fill

        # Evaluator uses the original (un-normalized) cost for total_grid_cost
        self._raw_cost_matrix = raw_cost

        # Pre-compute all valid swap pairs
        self._swap_pairs: List[Tuple[int, int]] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self._swap_pairs.append((i, j))
        self.n_swaps = len(self._swap_pairs)

        # Spaces
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0, high=self.n - 1, shape=(self.n,), dtype=np.int32
                ),
                "edge_costs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.n, self.n, 2),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(self.n_swaps)

        # Internal state
        self.grid: Optional[np.ndarray] = None
        self.step_count: int = 0
        self._prev_cost: float = 0.0
        self._prev_accuracy: float = 0.0

    def _encode_grid(self) -> np.ndarray:
        """Flatten grid to 1D observation array."""
        return self.grid.flatten().astype(np.int32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dict."""
        return {
            "grid": self._encode_grid(),
            "edge_costs": self.cost_matrix.astype(np.float32),
        }

    def _compute_total_cost(self) -> float:
        """Sum all right/down adjacency costs in current grid (using raw costs)."""
        return float(self.evaluator.matcher.total_grid_cost(self.grid, self._raw_cost_matrix))

    def _compute_accuracy(self) -> float:
        """Fraction of patches in their original positions."""
        return float(self.evaluator.compute_position_accuracy(self.grid, self.patches))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment with a new shuffled puzzle.

        Args:
            seed: Optional seed override.
            options: Unused, for Gymnasium compatibility.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed

        # Create identity grid then shuffle
        self.grid = np.arange(self.n, dtype=np.int32).reshape(self.rows, self.cols)
        rng = np.random.default_rng(effective_seed)
        flat = self.grid.flatten()
        rng.shuffle(flat)
        self.grid = flat.reshape(self.rows, self.cols)

        self.step_count = 0
        self._prev_cost = self._compute_total_cost()
        self._prev_accuracy = self._compute_accuracy()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute a swap action.

        Args:
            action: Integer index into pre-computed swap pairs.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        # Decode action
        pos_a, pos_b = self._swap_pairs[action]
        r1, c1 = divmod(pos_a, self.cols)
        r2, c2 = divmod(pos_b, self.cols)

        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        self.step_count += 1

        # Compute new metrics
        new_cost = self._compute_total_cost()
        new_accuracy = self._compute_accuracy()

        # Reward
        cost_delta = self._prev_cost - new_cost
        accuracy_delta = new_accuracy - self._prev_accuracy

        reward = self.cost_improvement_weight * cost_delta
        reward += self.accuracy_bonus_weight * accuracy_delta
        reward -= self.step_penalty

        terminated = False
        if new_accuracy >= 1.0:
            reward += self.solve_bonus
            terminated = True

        truncated = self.step_count >= self.max_steps

        # Update state
        self._prev_cost = new_cost
        self._prev_accuracy = new_accuracy

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict for current step."""
        return {
            "total_cost": self._prev_cost,
            "position_accuracy": self._prev_accuracy,
            "step": self.step_count,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render current state as image (rgb_array mode)."""
        if self.render_mode != "rgb_array":
            return None
        from jigsaw.utils import compose_image_from_grid

        return compose_image_from_grid(self.grid, self.patches)

    def decode_action(self, action: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Decode action index to grid position pairs.

        Args:
            action: Integer action index.

        Returns:
            ((row1, col1), (row2, col2)) position tuples.
        """
        pos_a, pos_b = self._swap_pairs[action]
        r1, c1 = divmod(pos_a, self.cols)
        r2, c2 = divmod(pos_b, self.cols)
        return (r1, c1), (r2, c2)

    def encode_action(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
        """Encode grid position pair to action index.

        Args:
            pos_a: (row, col) of first piece.
            pos_b: (row, col) of second piece.

        Returns:
            Integer action index.
        """
        idx_a = pos_a[0] * self.cols + pos_a[1]
        idx_b = pos_b[0] * self.cols + pos_b[1]
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a
        # Find index in swap_pairs
        # Use formula: for pairs (i,j) where i<j, index = i*n - i*(i+1)/2 + j - i - 1
        return idx_a * self.n - idx_a * (idx_a + 1) // 2 + idx_b - idx_a - 1
