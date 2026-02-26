"""Accuracy and runtime smoke tests on multiple image types."""

from __future__ import annotations

import time

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import (
    generate_gradient_image,
    generate_natural_like_image,
    generate_random_image,
    shuffle_patches,
)


def _evaluate_case(image_type: str, image, grid_size: int = 5) -> None:
    splitter = PuzzleSplitter()
    patches = splitter.split(image, grid_size, grid_size)
    shuffled, _ = shuffle_patches(patches, seed=42)

    matcher = EdgeMatcher()
    cost = matcher.build_cost_matrix(shuffled)
    solver = JigsawSolver(
        SolverConfig(rows=grid_size, cols=grid_size, seed=42, local_opt_iters=1000)
    )

    t0 = time.perf_counter()
    grid = solver.solve(shuffled, cost_matrix=cost)
    elapsed = time.perf_counter() - t0

    evaluator = PuzzleEvaluator()
    result = evaluator.evaluate(grid, shuffled, cost)

    print(
        f"{image_type}: accuracy={result.position_accuracy:.4f}, "
        f"neighbor_accuracy={result.neighbor_accuracy:.4f}, "
        f"runtime={elapsed:.4f}s, total_cost={result.total_cost:.2f}"
    )
    assert 0.0 <= result.position_accuracy <= 1.0
    assert 0.0 <= result.neighbor_accuracy <= 1.0
    assert elapsed >= 0.0
    assert result.total_cost >= 0.0


def test_random_image_metrics() -> None:
    """Report metrics for random image case."""
    image = generate_random_image(size=300, seed=42)
    _evaluate_case("random", image)


def test_gradient_image_metrics() -> None:
    """Report metrics for gradient image case."""
    image = generate_gradient_image(size=300)
    _evaluate_case("gradient", image)


def test_natural_image_metrics() -> None:
    """Report metrics for natural-like image case."""
    image = generate_natural_like_image(size=300, seed=42)
    _evaluate_case("natural", image)
