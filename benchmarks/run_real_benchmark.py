"""Real image benchmark runner for jigsaw puzzle reconstruction.

This script evaluates the solver on real-world images from a directory,
providing statistical analysis across multiple images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import SolverFactory, SolverConfig
from jigsaw.solver.base import SolveResult
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import shuffle_patches

from image_loader import DirectoryImageLoader


@dataclass
class ImageResult:
    """Result for a single image."""

    image_index: int
    image_name: str
    rows: int
    cols: int
    patch_height: int
    patch_width: int
    position_accuracy: float
    neighbor_accuracy: float
    total_cost: float
    runtime_seconds: float


@dataclass
class GridStats:
    """Statistics for a specific grid configuration."""

    rows: int
    cols: int
    patch_height: int
    patch_width: int
    num_images: int
    position_accuracy_mean: float
    position_accuracy_std: float
    position_accuracy_min: float
    position_accuracy_max: float
    neighbor_accuracy_mean: float
    neighbor_accuracy_std: float
    neighbor_accuracy_min: float
    neighbor_accuracy_max: float
    total_cost_mean: float
    total_cost_std: float
    total_cost_min: float
    total_cost_max: float
    runtime_mean: float
    runtime_std: float
    runtime_min: float
    runtime_max: float
    image_results: List[ImageResult] = field(default_factory=list)


def parse_grid_size(grid_str: str) -> Tuple[int, int]:
    """Parse grid size string like '3x5' into (rows, cols).

    Args:
        grid_str: Grid size string like '3x5' or '5'

    Returns:
        Tuple of (rows, cols)

    Raises:
        ValueError: If grid size is invalid (zero, negative, or non-numeric)
    """
    if 'x' in grid_str.lower():
        parts = grid_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid grid format: '{grid_str}'. Expected 'rowsxcols' (e.g., '3x5')")
        try:
            rows = int(parts[0])
            cols = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid grid size: '{grid_str}'. Must be numeric.")
    else:
        # If just a number, use square grid
        try:
            n = int(grid_str)
            rows, cols = n, n
        except ValueError:
            raise ValueError(f"Invalid grid size: '{grid_str}'. Must be numeric.")

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Grid size must be positive: rows={rows}, cols={cols}")

    return rows, cols


def run_single_image(
    image: np.ndarray,
    rows: int,
    cols: int,
    seed: int,
    solver_config: SolverConfig,
    solver_name: str = "default",
    verbose: bool = False,
) -> ImageResult:
    """Run puzzle reconstruction on a single image.

    Args:
        image: Input image (BGR format)
        rows: Requested number of rows
        cols: Requested number of columns
        seed: Random seed
        solver_config: Solver configuration
        solver_name: Name of the solver to use
        verbose: Print verbose output

    Returns:
        ImageResult with metrics
    """
    h, w = image.shape[:2]

    # Validate grid size vs image dimensions
    if rows > h or cols > w:
        raise ValueError(
            f"Grid size ({rows}x{cols}) is too large for image ({h}x{w}). "
            f"Each patch would be {h // rows}x{w // cols} pixels, which is invalid."
        )

    # Calculate patch dimensions
    patch_height = h // rows
    patch_width = w // cols

    if patch_height == 0 or patch_width == 0:
        raise ValueError(
            f"Grid size ({rows}x{cols}) produces zero-sized patches for image ({h}x{w})."
        )

    # Crop image to exact divisible size
    crop_h = patch_height * rows
    crop_w = patch_width * cols

    if crop_h < h or crop_w < w:
        # Center crop to get divisible dimensions
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

    # Split image into patches
    splitter = PuzzleSplitter()
    patches = splitter.split(image, rows, cols)

    # Shuffle patches
    shuffled, _ = shuffle_patches(patches, seed=seed)

    # Build cost matrix
    matcher = EdgeMatcher()
    cost_matrix = matcher.build_cost_matrix(shuffled)

    # Solve puzzle using factory
    solver = SolverFactory.create(
        solver_name,
        rows=solver_config.rows,
        cols=solver_config.cols,
        seed=solver_config.seed,
        local_opt_iters=solver_config.local_opt_iters,
        use_position_prior=solver_config.use_position_prior,
        auto_position_prior=solver_config.auto_position_prior,
    )

    start_time = time.perf_counter()
    # Pass original image to solver to prevent label leakage (for gaps solver)
    solve_result = solver.solve(shuffled, original_image=image, cost_matrix=cost_matrix)
    runtime = time.perf_counter() - start_time

    # Handle different return types (np.ndarray or SolveResult)
    if isinstance(solve_result, SolveResult):
        solved_grid = solve_result.grid
    else:
        solved_grid = solve_result

    # Evaluate
    evaluator = PuzzleEvaluator()
    result = evaluator.evaluate(solved_grid, shuffled, cost_matrix)

    if verbose:
        print(
            f"  Grid {rows}x{cols} (patch: {patch_height}x{patch_width}): "
            f"PosAcc={result.position_accuracy:.4f}, "
            f"NbrAcc={result.neighbor_accuracy:.4f}, "
            f"Cost={result.total_cost:.2f}, "
            f"Time={runtime:.3f}s"
        )

    return ImageResult(
        image_index=0,
        image_name="",
        rows=rows,
        cols=cols,
        patch_height=patch_height,
        patch_width=patch_width,
        position_accuracy=result.position_accuracy,
        neighbor_accuracy=result.neighbor_accuracy,
        total_cost=result.total_cost,
        runtime_seconds=runtime,
    )


def run_benchmark(
    loader: DirectoryImageLoader,
    grid_configs: List[Tuple[int, int]],
    num_images: int,
    skip_images: int,
    solver_config: SolverConfig,
    solver_name: str = "default",
    verbose: bool = False,
    progress_callback: Optional[callable] = None,
) -> List[GridStats]:
    """Run benchmark on a directory of images.

    Args:
        loader: DirectoryImageLoader instance
        grid_configs: List of (rows, cols) tuples to test
        num_images: Number of images to test
        skip_images: Number of images to skip from the start
        solver_config: Solver configuration
        solver_name: Name of the solver to use
        verbose: Print verbose output
        progress_callback: Optional callback for progress updates

    Returns:
        List of GridStats for each grid configuration
    """
    all_results: Dict[Tuple[int, int], List[ImageResult]] = {g: [] for g in grid_configs}

    total_tasks = num_images * len(grid_configs)
    completed = 0

    # Load images
    images = []
    image_names = []
    end_idx = min(skip_images + num_images, len(loader))

    if verbose:
        print(f"Loading {num_images} images (skipping first {skip_images})...")

    for idx in range(skip_images, end_idx):
        img, path = loader.load_image(idx)
        images.append(img)
        image_names.append(path.name)

    # Run benchmark for each grid configuration
    for rows, cols in grid_configs:
        if verbose:
            print(f"\n=== Testing {rows}x{cols} grid ===")

        for img_idx, (image, img_name) in enumerate(zip(images, image_names)):
            # Vary seed based on image index for diversity
            seed = solver_config.seed + img_idx

            # Create solver config for this grid size
            config = SolverConfig(
                rows=rows,
                cols=cols,
                seed=seed,
                local_opt_iters=solver_config.local_opt_iters,
                use_position_prior=solver_config.use_position_prior,
                auto_position_prior=solver_config.auto_position_prior,
            )

            result = run_single_image(
                image=image,
                rows=rows,
                cols=cols,
                seed=seed,
                solver_config=config,
                solver_name=solver_name,
                verbose=False,
            )
            result.image_index = img_idx
            result.image_name = img_name
            all_results[(rows, cols)].append(result)

            completed += 1
            if progress_callback:
                progress_callback(completed, total_tasks)

    # Compute statistics
    stats_list = []
    for rows, cols in grid_configs:
        results = all_results[(rows, cols)]
        if not results:
            continue

        pos_acc = np.array([r.position_accuracy for r in results])
        nbr_acc = np.array([r.neighbor_accuracy for r in results])
        costs = np.array([r.total_cost for r in results])
        runtimes = np.array([r.runtime_seconds for r in results])

        # Use actual rows/cols from first result
        patch_height = results[0].patch_height
        patch_width = results[0].patch_width

        stats = GridStats(
            rows=rows,
            cols=cols,
            patch_height=patch_height,
            patch_width=patch_width,
            num_images=len(results),
            position_accuracy_mean=float(np.mean(pos_acc)),
            position_accuracy_std=float(np.std(pos_acc)),
            position_accuracy_min=float(np.min(pos_acc)),
            position_accuracy_max=float(np.max(pos_acc)),
            neighbor_accuracy_mean=float(np.mean(nbr_acc)),
            neighbor_accuracy_std=float(np.std(nbr_acc)),
            neighbor_accuracy_min=float(np.min(nbr_acc)),
            neighbor_accuracy_max=float(np.max(nbr_acc)),
            total_cost_mean=float(np.mean(costs)),
            total_cost_std=float(np.std(costs)),
            total_cost_min=float(np.min(costs)),
            total_cost_max=float(np.max(costs)),
            runtime_mean=float(np.mean(runtimes)),
            runtime_std=float(np.std(runtimes)),
            runtime_min=float(np.min(runtimes)),
            runtime_max=float(np.max(runtimes)),
            image_results=results,
        )
        stats_list.append(stats)

    return stats_list


def print_table(stats_list: List[GridStats], dataset_path: Path) -> None:
    """Print benchmark results as a table."""
    print(f"\n{'='*60}")
    print("Real Image Benchmark Results")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    if stats_list:
        print(f"Images per grid: {stats_list[0].num_images}")
    print()
    print(
        f"{'Grid':<12}{'Patch':<12}{'Images':>8}{'PosAcc(%)':>12}{'NbrAcc(%)':>12}"
        f"{'Cost':>14}{'Time(s)':>10}"
    )
    print("-" * 80)

    for stats in stats_list:
        grid_str = f"{stats.rows}x{stats.cols}"
        usable_str = f"{stats.patch_height}x{stats.patch_width}"
        print(
            f"{grid_str:<12}{usable_str:<12}"
            f"{stats.num_images:>8}"
            f"{stats.position_accuracy_mean * 100:>12.2f}"
            f"{stats.neighbor_accuracy_mean * 100:>12.2f}"
            f"{stats.total_cost_mean:>14.2f}"
            f"{stats.runtime_mean:>10.3f}"
        )

    print()
    print("=== Statistics ===")
    all_pos = [s.position_accuracy_mean * 100 for s in stats_list]
    all_nbr = [s.neighbor_accuracy_mean * 100 for s in stats_list]
    print(f"Mean Position Accuracy: {np.mean(all_pos):.2f}% ± {np.std(all_pos):.2f}%")
    print(f"Mean Neighbor Accuracy: {np.mean(all_nbr):.2f}% ± {np.std(all_nbr):.2f}%")


def write_json_report(
    path: Path,
    dataset_path: Path,
    args: argparse.Namespace,
    stats_list: List[GridStats],
) -> None:
    """Write benchmark results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stats in stats_list:
        rows.append(
            {
                "rows": stats.rows,
                "cols": stats.cols,
                "patch_height": stats.patch_height,
                "patch_width": stats.patch_width,
                "num_images": stats.num_images,
                "position_accuracy_mean": stats.position_accuracy_mean,
                "position_accuracy_std": stats.position_accuracy_std,
                "position_accuracy_min": stats.position_accuracy_min,
                "position_accuracy_max": stats.position_accuracy_max,
                "neighbor_accuracy_mean": stats.neighbor_accuracy_mean,
                "neighbor_accuracy_std": stats.neighbor_accuracy_std,
                "neighbor_accuracy_min": stats.neighbor_accuracy_min,
                "neighbor_accuracy_max": stats.neighbor_accuracy_max,
                "total_cost_mean": stats.total_cost_mean,
                "total_cost_std": stats.total_cost_std,
                "runtime_mean": stats.runtime_mean,
                "runtime_std": stats.runtime_std,
            }
        )

    # Per-image results
    all_image_results = []
    for stats in stats_list:
        for img_result in stats.image_results:
            all_image_results.append(
                {
                    "image_index": img_result.image_index,
                    "image_name": img_result.image_name,
                    "rows": img_result.rows,
                    "cols": img_result.cols,
                    "patch_height": img_result.patch_height,
                    "patch_width": img_result.patch_width,
                    "position_accuracy": img_result.position_accuracy,
                    "neighbor_accuracy": img_result.neighbor_accuracy,
                    "total_cost": img_result.total_cost,
                    "runtime_seconds": img_result.runtime_seconds,
                }
            )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "num_images": stats_list[0].num_images if stats_list else 0,
        },
        "config": {
            "grid_configs": args.grid_configs,
            "num_images": args.num_images,
            "skip_images": args.skip_images,
            "image_size": args.image_size,
            "solver": args.solver,
            "local_opt_iters": args.local_opt_iters,
        },
        "summary": {
            "mean_position_accuracy": float(np.mean([s.position_accuracy_mean for s in stats_list])),
            "mean_neighbor_accuracy": float(np.mean([s.neighbor_accuracy_mean for s in stats_list])),
            "mean_runtime": float(np.mean([s.runtime_mean for s in stats_list])),
        },
        "rows": rows,
        "image_results": all_image_results,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nJSON report saved to: {path.resolve()}")


def write_csv_report(
    path: Path,
    stats_list: List[GridStats],
) -> None:
    """Write benchmark results to CSV file (per-image)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_index",
                "image_name",
                "rows",
                "cols",
                "patch_height",
                "patch_width",
                "position_accuracy",
                "neighbor_accuracy",
                "total_cost",
                "runtime_seconds",
            ]
        )

        for stats in stats_list:
            for img_result in stats.image_results:
                writer.writerow(
                    [
                        img_result.image_index,
                        img_result.image_name,
                        img_result.rows,
                        img_result.cols,
                        img_result.patch_height,
                        img_result.patch_width,
                        f"{img_result.position_accuracy:.6f}",
                        f"{img_result.neighbor_accuracy:.6f}",
                        f"{img_result.total_cost:.2f}",
                        f"{img_result.runtime_seconds:.4f}",
                    ]
                )

    print(f"CSV report saved to: {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run jigsaw benchmark on real-world images"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--grid-sizes",
        type=str,
        nargs="+",
        default=["3x5", "5x5", "8x8"],
        help="Grid sizes to benchmark (e.g., '3x5' '5x5' '8x8', default: 3x5 5x5 8x8)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of images to test (default: 50)",
    )
    parser.add_argument(
        "--skip-images",
        type=int,
        default=0,
        help="Skip first N images (default: 0)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=300,
        help="Resize images to this size (default: 300)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="default",
        choices=["default", "gaps"],
        help="Solver to use (default: default)",
    )
    parser.add_argument(
        "--local-opt-iters",
        type=int,
        default=1000,
        help="Local optimization iterations (default: 1000)",
    )
    parser.add_argument(
        "--use-position-prior",
        action="store_true",
        help="Enable position prior",
    )
    parser.add_argument(
        "--auto-position-prior",
        action="store_true",
        help="Enable auto position prior",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save CSV report (per-image)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress",
    )
    args = parser.parse_args()

    # Parse grid sizes
    args.grid_configs = [parse_grid_size(g) for g in args.grid_sizes]

    return args


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Initialize loader
    if args.verbose:
        print(f"Loading images from: {dataset_path}")

    loader = DirectoryImageLoader(
        directory=dataset_path,
        target_size=args.image_size,
        crop_to_square=True,
        recursive=False,
    )

    if args.verbose:
        print(f"Found {len(loader)} images")

    # Validate skip_images
    if args.skip_images < 0:
        print(f"Error: skip_images ({args.skip_images}) must be non-negative")
        sys.exit(1)

    # Validate num_images
    if args.num_images <= 0:
        print(f"Error: num_images ({args.num_images}) must be positive")
        sys.exit(1)

    available_images = len(loader) - args.skip_images
    if available_images <= 0:
        print(
            f"Error: skip_images ({args.skip_images}) >= number of available images ({len(loader)})"
        )
        sys.exit(1)
    if len(loader) < args.num_images + args.skip_images:
        print(
            f"Warning: Only {len(loader)} images available, "
            f"but requested {args.num_images + args.skip_images}. "
            f"Using {available_images} images instead."
        )
        args.num_images = min(args.num_images, available_images)

    # Configure solver (rows/cols will be updated per grid)
    solver_config = SolverConfig(
        rows=3,
        cols=3,
        seed=args.seed,
        local_opt_iters=args.local_opt_iters,
        use_position_prior=args.use_position_prior,
        auto_position_prior=args.auto_position_prior,
    )

    # Progress callback
    def progress(completed: int, total: int) -> None:
        if args.verbose:
            print(f"Progress: {completed}/{total} ({completed * 100 // total}%)")

    # Run benchmark
    print(f"\nRunning benchmark on {args.num_images} images...")
    print(f"Grid sizes: {args.grid_sizes}")

    # Warn about gaps solver not supporting seed
    if args.solver == "gaps":
        print("Warning: gaps solver does not support seed parameter. Results may not be reproducible.")

    stats_list = run_benchmark(
        loader=loader,
        grid_configs=args.grid_configs,
        num_images=args.num_images,
        skip_images=args.skip_images,
        solver_config=solver_config,
        solver_name=args.solver,
        verbose=args.verbose,
        progress_callback=progress if args.verbose else None,
    )

    # Print results
    print_table(stats_list, dataset_path)

    # Write reports
    if args.output_json:
        write_json_report(
            Path(args.output_json),
            dataset_path,
            args,
            stats_list,
        )

    if args.output_csv:
        write_csv_report(Path(args.output_csv), stats_list)


if __name__ == "__main__":
    main()
