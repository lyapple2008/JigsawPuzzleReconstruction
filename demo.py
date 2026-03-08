"""Demo script for square jigsaw puzzle reconstruction."""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import matplotlib.pyplot as plt

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.puzzle_roi import extract_puzzle_region_with_metadata
from jigsaw.solver import SolverFactory
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import (
    compose_image_row_major,
    load_or_generate_image,
    shuffle_patches,
)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def parse_grid(value: str) -> Tuple[int, int]:
    """Parse grid value in format ROWSxCOLS, e.g. 4x6."""
    text = value.strip().lower()
    if "x" not in text:
        raise argparse.ArgumentTypeError("grid must be in format ROWSxCOLS, e.g. 5x5")
    rows_text, cols_text = text.split("x", maxsplit=1)
    try:
        rows = int(rows_text)
        cols = int(cols_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid rows/cols must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("grid rows/cols must be positive")
    return rows, cols


def run_demo(
    image_path: str | None = None,
    grid: Tuple[int, int] = (5, 5),
    extract_roi: bool = False,
    solver_name: str = "default",
) -> None:
    """Run full pipeline and display original/shuffled/reconstructed images."""
    rows, cols = grid
    grid_size = max(rows, cols)  # For backward compatibility

    if extract_roi and image_path and cv2 is not None:
        raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        roi_result = extract_puzzle_region_with_metadata(raw)
        image = roi_result.image
        if roi_result.rows is not None and roi_result.cols is not None:
            print(f"Extracted puzzle ROI: bbox={roi_result.bbox}, inferred grid={roi_result.rows}x{roi_result.cols}")
        side = (300 // grid_size) * grid_size
        if side < grid_size:
            side = grid_size * 60
        image = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    else:
        image = load_or_generate_image(image_path, size=300, seed=42)

    # Handle non-divisible image dimensions
    h, w = image.shape[:2]
    patch_height = h // rows
    patch_width = w // cols

    if patch_height == 0 or patch_width == 0:
        raise ValueError(
            f"Image size ({h}x{w}) is too small for grid ({rows}x{cols}). "
            f"Each patch would be {patch_height}x{patch_width} pixels."
        )

    # Crop image to exact divisible size
    crop_h = patch_height * rows
    crop_w = patch_width * cols

    if crop_h < h or crop_w < w:
        # Center crop
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
        if rows != patch_height or cols != patch_width:
            print(f"Note: Image cropped from {h}x{w} to {crop_h}x{crop_w} for grid {rows}x{cols}")

    splitter = PuzzleSplitter()
    patches = splitter.split(image, rows, cols)
    shuffled_patches, _ = shuffle_patches(patches, seed=42)

    matcher = EdgeMatcher()
    cost_matrix = matcher.build_cost_matrix(shuffled_patches)

    # Create solver using factory
    solver = SolverFactory.create(
        solver_name,
        rows=rows,
        cols=cols,
        seed=42,
    )

    start = time.perf_counter()
    solve_result = solver.solve(shuffled_patches, cost_matrix=cost_matrix)
    duration = time.perf_counter() - start

    solved_grid = solve_result.grid
    reconstructed_image = solve_result.reconstructed_image

    evaluator = PuzzleEvaluator()
    result = evaluator.evaluate(solved_grid, shuffled_patches, cost_matrix)

    shuffled_image = compose_image_row_major(shuffled_patches, rows, cols)

    print(f"Solver: {solver_name}")
    print(f"Grid: {rows}x{cols}")
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
    parser.add_argument(
        "--grid",
        type=parse_grid,
        default=(5, 5),
        help="Grid format: ROWSxCOLS (e.g., 5x5, 4x6)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="default",
        choices=["default", "gaps"],
        help="Solver algorithm to use (default or gaps)",
    )
    parser.add_argument(
        "--extract-roi",
        action="store_true",
        help="Extract puzzle region from screenshot before running demo",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        image_path=args.image,
        grid=args.grid,
        extract_roi=args.extract_roi,
        solver_name=args.solver,
    )
