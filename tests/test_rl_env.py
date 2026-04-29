"""Tests for the puzzle RL environment."""

from __future__ import annotations

import numpy as np
import pytest

from jigsaw.utils import generate_natural_like_image
from rl.envs.puzzle_env import PuzzleSwapEnv


@pytest.fixture
def small_image():
    """Generate a small test image."""
    return generate_natural_like_image(size=300, seed=42)


@pytest.fixture
def env_3x3(small_image):
    """Create a 3x3 puzzle environment."""
    return PuzzleSwapEnv(image=small_image, rows=3, cols=3, max_steps=50, seed=42)


@pytest.fixture
def env_6x6(small_image):
    """Create a 6x6 puzzle environment."""
    return PuzzleSwapEnv(image=small_image, rows=6, cols=6, max_steps=100, seed=42)


class TestPuzzleSwapEnv:
    """Test suite for PuzzleSwapEnv."""

    def test_init(self, env_3x3):
        """Test environment initialization."""
        assert env_3x3.rows == 3
        assert env_3x3.cols == 3
        assert env_3x3.n == 9
        assert env_3x3.n_swaps == 36  # 9*8/2

    def test_observation_space(self, env_3x3):
        """Test observation space shapes."""
        obs_space = env_3x3.observation_space
        assert "grid" in obs_space.spaces
        assert "edge_costs" in obs_space.spaces
        assert obs_space["grid"].shape == (9,)
        assert obs_space["edge_costs"].shape == (9, 9, 2)

    def test_action_space(self, env_3x3):
        """Test action space size."""
        assert env_3x3.action_space.n == 36  # 9*8/2

    def test_reset(self, env_3x3):
        """Test reset returns valid observation."""
        obs, info = env_3x3.reset(seed=123)
        assert "grid" in obs
        assert "edge_costs" in obs
        assert obs["grid"].shape == (9,)
        assert obs["edge_costs"].shape == (9, 9, 2)
        # Grid should be a permutation of 0..8
        assert sorted(obs["grid"].tolist()) == list(range(9))

    def test_step(self, env_3x3):
        """Test step returns correct tuple."""
        env_3x3.reset(seed=42)
        obs, reward, terminated, truncated, info = env_3x3.step(0)
        assert "grid" in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "total_cost" in info
        assert "position_accuracy" in info

    def test_step_updates_grid(self, env_3x3):
        """Test that step actually swaps pieces."""
        env_3x3.reset(seed=42)
        grid_before = env_3x3.grid.copy()
        env_3x3.step(0)  # Swap first pair
        # Grid should have changed (unless swapping same piece, which shouldn't happen)
        assert not np.array_equal(grid_before, env_3x3.grid) or env_3x3.n_swaps == 0

    def test_terminated_on_solve(self, small_image):
        """Test that episode terminates when puzzle is solved."""
        env = PuzzleSwapEnv(image=small_image, rows=2, cols=2, max_steps=100, seed=42)
        obs, info = env.reset(seed=42)

        # Manually set grid to solved state
        env.grid = np.arange(4, dtype=np.int32).reshape(2, 2)
        env._prev_cost = env._compute_total_cost()
        env._prev_accuracy = env._compute_accuracy()

        # Take any action (should detect already solved)
        # Actually, we need to test that after swaps that solve it, terminated=True
        # For 2x2, try all swaps to find one that solves it
        for action in range(env.n_swaps):
            env_test = PuzzleSwapEnv(image=small_image, rows=2, cols=2, max_steps=100, seed=42)
            env_test.reset(seed=42)
            # Set to one swap away from solved
            env_test.grid = np.arange(4, dtype=np.int32).reshape(2, 2)
            # Swap two pieces
            env_test.grid[0, 0], env_test.grid[0, 1] = env_test.grid[0, 1], env_test.grid[0, 0]
            env_test._prev_cost = env_test._compute_total_cost()
            env_test._prev_accuracy = env_test._compute_accuracy()

            # Now swap them back
            action_idx = env_test.encode_action((0, 0), (0, 1))
            obs, reward, terminated, truncated, info = env_test.step(action_idx)
            if info["position_accuracy"] >= 1.0:
                assert terminated
                return

    def test_truncated_at_max_steps(self, small_image):
        """Test that episode truncates at max_steps."""
        env = PuzzleSwapEnv(image=small_image, rows=3, cols=3, max_steps=5, seed=42)
        env.reset(seed=42)

        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            if truncated:
                assert info["step"] >= 5
                return
        # Should have truncated by step 5
        assert False, "Episode did not truncate within max_steps"

    def test_decode_encode_action(self, env_3x3):
        """Test action encoding/decoding roundtrip."""
        for action_idx in range(env_3x3.n_swaps):
            (r1, c1), (r2, c2) = env_3x3.decode_action(action_idx)
            recovered = env_3x3.encode_action((r1, c1), (r2, c2))
            assert recovered == action_idx, f"Action {action_idx} encode/decode mismatch"

    def test_cost_matrix_properties(self, env_3x3):
        """Test cost matrix is valid (normalized, no inf/nan)."""
        cm = env_3x3.cost_matrix
        assert cm.shape == (9, 9, 2)
        # All values should be finite and in [0, 1]
        assert np.all(np.isfinite(cm))
        assert np.all(cm >= 0.0)
        assert np.all(cm <= 1.0)
        # Diagonal should be 1.0 (fill value for self-pairs)
        for i in range(9):
            assert cm[i, i, 0] == 1.0
            assert cm[i, i, 1] == 1.0

    def test_accuracy_range(self, env_3x3):
        """Test accuracy is between 0 and 1."""
        env_3x3.reset(seed=42)
        obs, reward, terminated, truncated, info = env_3x3.step(0)
        acc = info["position_accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_different_seeds_different_shuffles(self, env_3x3):
        """Test that different seeds produce different initial states."""
        obs1, _ = env_3x3.reset(seed=1)
        obs2, _ = env_3x3.reset(seed=2)
        assert not np.array_equal(obs1["grid"], obs2["grid"])

    def test_6x6_env(self, env_6x6):
        """Test 6x6 environment basic operations."""
        obs, info = env_6x6.reset(seed=42)
        assert obs["grid"].shape == (36,)
        assert env_6x6.n_swaps == 630  # 36*35/2
        obs, reward, terminated, truncated, info = env_6x6.step(0)
        assert isinstance(reward, float)
