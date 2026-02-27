"""Benchmark solver quality across puzzle sizes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import PuzzleSplitter
from jigsaw.utils import generate_natural_like_image, shuffle_patches


@dataclass
class BenchmarkRow:
    grid: str
    position_accuracy: float
    neighbor_accuracy: float
    total_cost: float
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
        position_accuracy=result.position_accuracy,
        neighbor_accuracy=result.neighbor_accuracy,
        total_cost=result.total_cost,
        runtime_sec=runtime_sec,
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
        "--local-opt-iters",
        type=int,
        default=1000,
        help="Local optimization iterations",
    )
    return parser.parse_args()


def print_table(rows: List[BenchmarkRow]) -> None:
    header = (
        f"{'Grid':<8}{'PosAcc':>10}{'NbrAcc':>10}{'TotalCost':>14}{'Runtime(s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.grid:<8}"
            f"{row.position_accuracy:>10.4f}"
            f"{row.neighbor_accuracy:>10.4f}"
            f"{row.total_cost:>14.2f}"
            f"{row.runtime_sec:>12.4f}"
        )


def main() -> None:
    args = parse_args()
    rows = [run_case(size, seed=args.seed, local_opt_iters=args.local_opt_iters) for size in args.sizes]
    print_table(rows)


if __name__ == "__main__":
    main()
