"""Extract puzzle state from iOS screenshots for RL inference.

Pipeline:
1. Extract puzzle region from screenshot (roi_color or puzzle_roi)
2. Split into grid cells (gap_splitter)
3. Build edge cost matrix (matcher)
4. Output observation matching PuzzleSwapEnv format
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from jigsaw.matcher import EdgeMatcher
from jigsaw.roi_color import ColorBasedPuzzleExtractor
from jigsaw.splitter import Patch


class ScreenStateExtractor:
    """Extract RL-compatible observation from a puzzle screenshot.

    Takes a raw screenshot (e.g., from iOS device) and produces
    the same observation format as PuzzleSwapEnv: {"grid": ..., "edge_costs": ...}.

    The grid is unknown from a single screenshot — we can only extract
    the image pieces and build cost features. The actual grid assignment
    needs to be inferred or tracked from previous actions.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (6, 6),
        cost_fill_multiplier: float = 10.0,
    ) -> None:
        """Initialize the state extractor.

        Args:
            grid_size: (rows, cols) of the puzzle grid.
            cost_fill_multiplier: Multiplier for inf entries in cost matrix.
        """
        self.rows, self.cols = grid_size
        self.n = self.rows * self.cols
        self.cost_fill_multiplier = cost_fill_multiplier
        self.roi_extractor = ColorBasedPuzzleExtractor()
        self.matcher = EdgeMatcher()

    def extract_from_screenshot(
        self,
        screenshot: np.ndarray,
    ) -> Tuple[np.ndarray, List[Patch], np.ndarray]:
        """Extract puzzle region and build cost matrix from screenshot.

        Args:
            screenshot: Full screenshot as HxWx3 uint8 numpy array (RGB).

        Returns:
            (puzzle_image, patches, normalized_cost_matrix)
        """
        from jigsaw.gap_splitter import split_with_gap_aware

        # Step 1: Extract puzzle ROI
        roi_result = self.roi_extractor.extract(screenshot)
        if roi_result is None:
            raise ValueError("Could not extract puzzle region from screenshot")
        puzzle_image = roi_result.image

        # Step 2: Split into patches (gap-aware for real screenshots)
        patches, _, _ = split_with_gap_aware(
            puzzle_image, self.rows, self.cols
        )

        # Step 3: Build cost matrix
        raw_cost = self.matcher.build_cost_matrix(patches)
        normalized_cost = self._normalize_cost(raw_cost)

        return puzzle_image, patches, normalized_cost

    def extract_from_image(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Patch], np.ndarray]:
        """Extract patches and cost matrix from a clean puzzle image.

        For testing with pre-cropped puzzle images (no ROI extraction needed).

        Args:
            image: Clean puzzle image as HxWx3 uint8 numpy array.

        Returns:
            (patches, normalized_cost_matrix)
        """
        from jigsaw.splitter import PuzzleSplitter

        splitter = PuzzleSplitter()
        patches = splitter.split(image, self.rows, self.cols)

        raw_cost = self.matcher.build_cost_matrix(patches)
        normalized_cost = self._normalize_cost(raw_cost)

        return patches, normalized_cost

    def _normalize_cost(self, raw_cost: np.ndarray) -> np.ndarray:
        """Normalize cost matrix: replace inf, scale to [0, 1]."""
        finite_vals = raw_cost[np.isfinite(raw_cost)]
        cost_max = float(finite_vals.max()) if finite_vals.size > 0 else 1.0
        cost_fill = cost_max * self.cost_fill_multiplier
        normalized = np.where(np.isfinite(raw_cost), raw_cost, cost_fill)
        if cost_fill > 0:
            normalized = normalized / cost_fill
        return normalized.astype(np.float32)

    def build_observation(
        self,
        grid: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> dict:
        """Build observation dict matching PuzzleSwapEnv format.

        Args:
            grid: Current grid state as (rows, cols) int32 array.
            cost_matrix: Normalized cost matrix as (n, n, 2) float32.

        Returns:
            Observation dict with "grid" and "edge_costs".
        """
        return {
            "grid": grid.flatten().astype(np.int32),
            "edge_costs": cost_matrix.astype(np.float32),
        }

    def estimate_grid_from_screenshot(
        self,
        current_patches: List[Patch],
        reference_patches: List[Patch],
    ) -> np.ndarray:
        """Estimate current grid by matching screenshot patches to reference.

        Uses L2 distance on flattened patch images to find best matches.
        This is a simple heuristic — for real deployment, track state
        from action history instead.

        Args:
            current_patches: Patches extracted from current screenshot.
            reference_patches: Patches from the original image (known order).

        Returns:
            Estimated grid as (rows, cols) int32 array.
        """
        n = len(current_patches)
        grid = np.zeros(n, dtype=np.int32)

        # Flatten reference patches
        ref_flat = [p.image.flatten().astype(np.float32) for p in reference_patches]

        for i, cp in enumerate(current_patches):
            cp_flat = cp.image.flatten().astype(np.float32)
            best_j = 0
            best_dist = float("inf")
            for j, rf in enumerate(ref_flat):
                dist = float(np.mean((cp_flat - rf) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            grid[i] = best_j

        return grid.reshape(self.rows, self.cols)
