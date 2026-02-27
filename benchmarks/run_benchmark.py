"""Benchmark solver quality across puzzle sizes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.gap_splitter import split_with_gap_aware
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import Patch, PuzzleSplitter
from jigsaw.utils import generate_natural_like_image, shuffle_patches


@dataclass
class BenchmarkRow:
    grid: str
    seeds: int
    pos_acc_mean: float
    pos_acc_min: float
    nbr_acc_mean: float
    nbr_acc_min: float
    total_cost_mean: float
    position_accuracy: float
    neighbor_accuracy: float
    total_cost: float
    runtime_mean_sec: float
    runtime_sec: float


def run_case(grid_size: int, seed: int, local_opt_iters: int) -> BenchmarkRow:
    image_size = grid_size * 60
    image = generate_natural_like_image(size=image_size, seed=seed)
    splitter = PuzzleSplitter()
    patches = splitter.split(image, grid_size, grid_size)
    shuffled, _ = shuffle_patches(patches, seed=seed)

    matcher = EdgeMatcher()
    cost = matcher.build_cost_matrix(shuffled)
    solver = JigsawSolver(
        SolverConfig(
            rows=grid_size,
            cols=grid_size,
            seed=seed,
            local_opt_iters=local_opt_iters,
        )
    )

    t0 = time.perf_counter()
    solved = solver.solve(shuffled, cost_matrix=cost)
    runtime_sec = time.perf_counter() - t0

    evaluator = PuzzleEvaluator()
    result = evaluator.evaluate(solved, shuffled, cost)
    return BenchmarkRow(
        grid=f"{grid_size}x{grid_size}",
        seeds=1,
        pos_acc_mean=result.position_accuracy,
        pos_acc_min=result.position_accuracy,
        nbr_acc_mean=result.neighbor_accuracy,
        nbr_acc_min=result.neighbor_accuracy,
        total_cost_mean=result.total_cost,
        position_accuracy=result.position_accuracy,
        neighbor_accuracy=result.neighbor_accuracy,
        total_cost=result.total_cost,
        runtime_mean_sec=runtime_sec,
        runtime_sec=runtime_sec,
    )


def _compose_with_uniform_gap(
    patches: List[Patch], rows: int, cols: int, gap: int, background: int = 0
) -> np.ndarray:
    patch_h, patch_w, ch = patches[0].image.shape
    out_h = rows * patch_h + (rows - 1) * gap
    out_w = cols * patch_w + (cols - 1) * gap
    canvas = np.full((out_h, out_w, ch), background, dtype=patches[0].image.dtype)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r * (patch_h + gap)
            x0 = c * (patch_w + gap)
            canvas[y0 : y0 + patch_h, x0 : x0 + patch_w, :] = patches[idx].image
            idx += 1
    return canvas


def run_case_with_gap(
    grid_size: int, seed: int, local_opt_iters: int, gap: int
) -> BenchmarkRow:
    image_size = grid_size * 60
    image = generate_natural_like_image(size=image_size, seed=seed)
    splitter = PuzzleSplitter()
    patches = splitter.split(image, grid_size, grid_size)
    shuffled, _ = shuffle_patches(patches, seed=seed)

    gapped = _compose_with_uniform_gap(shuffled, grid_size, grid_size, gap=gap, background=10)
    extracted, _, _ = split_with_gap_aware(gapped, rows=grid_size, cols=grid_size)
    # Restore semantic labels for evaluation.
    labeled = [
        Patch.from_image(extracted[i].image, original_index=shuffled[i].original_index)
        for i in range(len(extracted))
    ]

    matcher = EdgeMatcher()
    cost = matcher.build_cost_matrix(labeled)
    solver = JigsawSolver(
        SolverConfig(
            rows=grid_size,
            cols=grid_size,
            seed=seed,
            local_opt_iters=local_opt_iters,
        )
    )

    t0 = time.perf_counter()
    solved = solver.solve(labeled, cost_matrix=cost)
    runtime_sec = time.perf_counter() - t0

    evaluator = PuzzleEvaluator()
    result = evaluator.evaluate(solved, labeled, cost)
    return BenchmarkRow(
        grid=f"{grid_size}x{grid_size}(gap={gap})",
        seeds=1,
        pos_acc_mean=result.position_accuracy,
        pos_acc_min=result.position_accuracy,
        nbr_acc_mean=result.neighbor_accuracy,
        nbr_acc_min=result.neighbor_accuracy,
        total_cost_mean=result.total_cost,
        position_accuracy=result.position_accuracy,
        neighbor_accuracy=result.neighbor_accuracy,
        total_cost=result.total_cost,
        runtime_mean_sec=runtime_sec,
        runtime_sec=runtime_sec,
    )


def run_case_multi_seed(
    grid_size: int, seeds: List[int], local_opt_iters: int, gap: int
) -> BenchmarkRow:
    if gap > 0:
        rows = [
            run_case_with_gap(grid_size, seed=seed, local_opt_iters=local_opt_iters, gap=gap)
            for seed in seeds
        ]
        grid_label = f"{grid_size}x{grid_size}(gap={gap})"
    else:
        rows = [run_case(grid_size, seed=seed, local_opt_iters=local_opt_iters) for seed in seeds]
        grid_label = f"{grid_size}x{grid_size}"

    pos = np.array([r.position_accuracy for r in rows], dtype=np.float64)
    nbr = np.array([r.neighbor_accuracy for r in rows], dtype=np.float64)
    cst = np.array([r.total_cost for r in rows], dtype=np.float64)
    rt = np.array([r.runtime_sec for r in rows], dtype=np.float64)
    return BenchmarkRow(
        grid=grid_label,
        seeds=len(seeds),
        pos_acc_mean=float(np.mean(pos)),
        pos_acc_min=float(np.min(pos)),
        nbr_acc_mean=float(np.mean(nbr)),
        nbr_acc_min=float(np.min(nbr)),
        total_cost_mean=float(np.mean(cst)),
        position_accuracy=float(pos[0]),
        neighbor_accuracy=float(nbr[0]),
        total_cost=float(cst[0]),
        runtime_mean_sec=float(np.mean(rt)),
        runtime_sec=float(rt[0]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run jigsaw benchmark on multiple grid sizes.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 8, 10],
        help="Grid sizes to benchmark (default: 5 8 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to evaluate per grid (default: 1)",
    )
    parser.add_argument(
        "--local-opt-iters",
        type=int,
        default=1000,
        help="Local optimization iterations",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Optional uniform gap width between patches for stress testing (default: 0)",
    )
    return parser.parse_args()


def print_table(rows: List[BenchmarkRow]) -> None:
    header = (
        f"{'Grid':<8}{'Seeds':>7}{'PosMean':>10}{'PosMin':>10}"
        f"{'NbrMean':>10}{'NbrMin':>10}{'CostMean':>12}{'RtMean(s)':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.grid:<8}"
            f"{row.seeds:>7d}"
            f"{row.pos_acc_mean:>10.4f}"
            f"{row.pos_acc_min:>10.4f}"
            f"{row.nbr_acc_mean:>10.4f}"
            f"{row.nbr_acc_min:>10.4f}"
            f"{row.total_cost_mean:>12.2f}"
            f"{row.runtime_mean_sec:>11.4f}"
        )


def main() -> None:
    args = parse_args()
    seeds = [args.seed + i for i in range(args.num_seeds)]
    rows = [
        run_case_multi_seed(
            size,
            seeds=seeds,
            local_opt_iters=args.local_opt_iters,
            gap=args.gap,
        )
        for size in args.sizes
    ]
    print_table(rows)


if __name__ == "__main__":
    main()
