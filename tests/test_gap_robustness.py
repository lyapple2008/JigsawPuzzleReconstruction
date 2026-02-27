"""End-to-end tests for gap-aware splitting and reconstruction."""

from __future__ import annotations

import numpy as np

from jigsaw.evaluator import PuzzleEvaluator
from jigsaw.gap_splitter import split_with_gap_aware
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import Patch, PuzzleSplitter
from jigsaw.utils import generate_natural_like_image, shuffle_patches


def _compose_with_uniform_gap(
    patches: list[Patch], rows: int, cols: int, gap: int, background: int = 0
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


def _relabel_from_source(
    extracted: list[Patch], source_shuffled: list[Patch]
) -> list[Patch]:
    # split_with_gap_aware returns row-major extracted pieces; align labels with source list.
    return [
        Patch.from_image(extracted[i].image, original_index=source_shuffled[i].original_index)
        for i in range(len(extracted))
    ]


def test_gap_aware_split_and_solve_8x8() -> None:
    """Gap-aware split should preserve recoverability for 8x8 shuffled-with-gap input."""
    rows, cols = 8, 8
    image = generate_natural_like_image(size=rows * 42, seed=42)
    base = PuzzleSplitter().split(image, rows, cols)
    shuffled, _ = shuffle_patches(base, seed=7)
    gapped = _compose_with_uniform_gap(shuffled, rows, cols, gap=3, background=10)

    extracted, _, _ = split_with_gap_aware(gapped, rows=rows, cols=cols)
    labeled = _relabel_from_source(extracted, shuffled)

    matcher = EdgeMatcher()
    cost = matcher.build_cost_matrix(labeled)
    grid = JigsawSolver(
        SolverConfig(rows=rows, cols=cols, seed=42, local_opt_iters=1200)
    ).solve(labeled, cost_matrix=cost)

    evaluator = PuzzleEvaluator()
    acc = evaluator.compute_position_accuracy(grid, labeled)
    assert acc > 0.9
