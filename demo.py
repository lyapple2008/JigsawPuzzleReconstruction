"""Demo script for square jigsaw puzzle reconstruction."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import (
    compose_image_from_grid,
    compose_image_row_major,
    load_or_generate_image,
    shuffle_patches,
)


def run_demo(image_path: str | None = None, grid_size: int = 5) -> None:
    """Run full pipeline and display original/shuffled/reconstructed images."""
    image = load_or_generate_image(image_path, size=300, seed=42)
    splitter = PuzzleSplitter()
    patches = splitter.split(image, grid_size, grid_size)
    shuffled_patches, _ = shuffle_patches(patches, seed=42)

    matcher = EdgeMatcher()
    cost_matrix = matcher.build_cost_matrix(shuffled_patches)
    solver = JigsawSolver(
        SolverConfig(rows=grid_size, cols=grid_size, seed=42, local_opt_iters=1000)
    )

    start = time.perf_counter()
    solved_grid = solver.solve(shuffled_patches, cost_matrix=cost_matrix)
    duration = time.perf_counter() - start

    evaluator = PuzzleEvaluator()
    result = evaluator.evaluate(solved_grid, shuffled_patches, cost_matrix)

    shuffled_image = compose_image_row_major(shuffled_patches, grid_size, grid_size)
    reconstructed_image = compose_image_from_grid(solved_grid, shuffled_patches)

    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Position accuracy: {result.position_accuracy:.4f}")
    print(f"Neighbor accuracy: {result.neighbor_accuracy:.4f}")
    print(f"Total matching cost: {result.total_cost:.2f}")
    print(f"Solve time: {duration:.4f}s")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(shuffled_image)
    axes[1].set_title("Shuffled")
    axes[2].imshow(reconstructed_image)
    axes[2].set_title("Reconstructed")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Square jigsaw puzzle reconstruction demo")
    parser.add_argument("--image", type=str, default=None, help="Optional input image path")
    parser.add_argument("--grid-size", type=int, default=5, help="Puzzle grid size, default=5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(image_path=args.image, grid_size=args.grid_size)
