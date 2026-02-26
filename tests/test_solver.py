"""Solver integration tests."""

from __future__ import annotations

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import generate_natural_like_image, shuffle_patches


def _run_case(grid_size: int) -> float:
    image_size = grid_size * 60
    image = generate_natural_like_image(size=image_size, seed=42)
    splitter = PuzzleSplitter()
    patches = splitter.split(image, grid_size, grid_size)
    shuffled, _ = shuffle_patches(patches, seed=42)

    matcher = EdgeMatcher()
    cost = matcher.build_cost_matrix(shuffled)
    solver = JigsawSolver(
        SolverConfig(rows=grid_size, cols=grid_size, seed=42, local_opt_iters=1000)
    )
    grid = solver.solve(shuffled, cost_matrix=cost)

    evaluator = PuzzleEvaluator()
    return evaluator.compute_position_accuracy(grid, shuffled)


def test_solver_3x3_accuracy() -> None:
    """3x3 puzzle should reconstruct with high accuracy."""
    acc = _run_case(3)
    assert acc > 0.8


def test_solver_5x5_accuracy() -> None:
    """5x5 puzzle should reconstruct with high accuracy."""
    acc = _run_case(5)
    assert acc > 0.8
