"""Extract puzzle region from a screenshot containing UI and a central grid puzzle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class PuzzleROIResult:
    """Result of puzzle region extraction."""

    image: np.ndarray
    """Cropped image containing only the puzzle grid (HxWx3 RGB)."""

    bbox: Tuple[int, int, int, int]
    """Bounding box (x_min, y_min, x_max, y_max) in original image coordinates."""

    rows: Optional[int] = None
    """Inferred number of rows (pieces along height), if available."""

    cols: Optional[int] = None
    """Inferred number of columns (pieces along width), if available."""


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("puzzle_roi requires opencv-python (cv2)")


def _gray_and_edges(image: np.ndarray, blur_ksize: int = 3, canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
    """Convert to grayscale, blur, and compute Canny edges."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges


def _detect_line_segments(edges: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """Detect line segments and split into horizontal and vertical.
    Returns (horizontal_segments, vertical_segments) as list of (x1,y1,x2,y2).
    """
    h, w = edges.shape
    min_len = min(h, w) // 20
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=4,
    )
    if lines is None:
        return [], []

    horizontal: List[Tuple[float, float, float, float]] = []
    vertical: List[Tuple[float, float, float, float]] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx + dy < 1e-6:
            continue
        if dx >= dy:
            horizontal.append((x1, y1, x2, y2))
        else:
            vertical.append((x1, y1, x2, y2))
    return horizontal, vertical


def _line_positions(
    segments: List[Tuple[float, float, float, float]],
    use_coord: str,
    merge_dist: int,
) -> List[float]:
    """Extract primary coordinate from each segment (y for horizontal, x for vertical) and merge nearby."""
    positions: List[float] = []
    for (x1, y1, x2, y2) in segments:
        if use_coord == "y":
            pos = (y1 + y2) / 2
        else:
            pos = (x1 + x2) / 2
        positions.append(pos)
    positions.sort()
    if not positions:
        return []

    merged: List[float] = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] <= merge_dist:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    return merged


def _dominant_period(positions: List[float], min_period_ratio: float = 0.02, max_period_ratio: float = 0.5) -> Optional[float]:
    """Estimate dominant spacing (period) from sorted positions using gap histogram."""
    if len(positions) < 2:
        return None
    gaps: List[float] = []
    for i in range(1, len(positions)):
        g = positions[i] - positions[i - 1]
        if g > 0:
            gaps.append(g)
    if not gaps:
        return None
    span = max(positions) - min(positions)
    if span <= 0:
        return None
    min_period = span * min_period_ratio
    max_period = span * max_period_ratio
    gaps = [g for g in gaps if min_period <= g <= max_period]
    if not gaps:
        return None
    hist, bin_edges = np.histogram(gaps, bins=min(50, len(gaps) + 1))
    peak_idx = int(np.argmax(hist))
    period = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2)
    return period


def _filter_regular_grid(positions: List[float], period: float, tolerance: float = 0.25) -> List[float]:
    """Keep only positions that lie on a regular grid with given period."""
    if not positions or period <= 0:
        return positions
    base = positions[0]
    tol = period * tolerance
    kept: List[float] = []
    for p in positions:
        k = round((p - base) / period)
        if abs((p - base) - k * period) <= tol:
            kept.append(p)
    return sorted(kept) if kept else positions


def _infer_grid_size(positions: List[float], period: float) -> int:
    """Infer number of cells: (n_lines - 1) for grid boundaries."""
    if not positions or period <= 0:
        return 0
    n_lines = len(positions)
    return max(0, n_lines - 1)


def extract_puzzle_region(
    image: np.ndarray,
    *,
    return_bbox: bool = False,
    merge_dist: Optional[int] = None,
    min_cells: int = 2,
    max_cells: int = 30,
    period_tolerance: float = 0.25,
) -> np.ndarray | Tuple[np.ndarray, Tuple[int, int, int, int]] | PuzzleROIResult:
    """Extract the puzzle grid region from an image (e.g. screenshot with UI).

    Detects regular horizontal and vertical grid lines, estimates cell size,
    and crops to the bounding box of the grid. If no clear grid is found,
    returns the original image (and optional full-image bbox).

    Args:
        image: RGB image (HxWx3) from screenshot.
        return_bbox: If True, return (cropped_image, bbox) instead of just image.
        merge_dist: Max pixel distance to merge nearby lines; default min(h,w)//80.
        min_cells: Minimum number of cells per dimension to consider valid grid.
        max_cells: Maximum number of cells per dimension (filter out noise).
        period_tolerance: Tolerance for aligning positions to grid (fraction of period).

    Returns:
        By default: cropped RGB image (numpy array).
        If return_bbox=True: (cropped_image, (x_min, y_min, x_max, y_max)).
        Can also return PuzzleROIResult with optional rows/cols when inferred.

    Raises:
        RuntimeError: If opencv (cv2) is not available.
    """
    _require_cv2()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    h, w = image.shape[:2]
    if merge_dist is None:
        merge_dist = max(2, min(h, w) // 80)

    edges = _gray_and_edges(image)
    horizontal_seg, vertical_seg = _detect_line_segments(edges)

    y_positions = _line_positions(horizontal_seg, "y", merge_dist)
    x_positions = _line_positions(vertical_seg, "x", merge_dist)

    period_y = _dominant_period(y_positions) if len(y_positions) >= 2 else None
    period_x = _dominant_period(x_positions) if len(x_positions) >= 2 else None

    if period_y is not None:
        y_positions = _filter_regular_grid(y_positions, period_y, period_tolerance)
    if period_x is not None:
        x_positions = _filter_regular_grid(x_positions, period_x, period_tolerance)

    rows_inferred = _infer_grid_size(y_positions, period_y) if period_y else 0
    cols_inferred = _infer_grid_size(x_positions, period_x) if period_x else 0

    if (
        len(y_positions) >= min_cells + 1
        and len(x_positions) >= min_cells + 1
        and rows_inferred <= max_cells
        and cols_inferred <= max_cells
        and rows_inferred >= min_cells
        and cols_inferred >= min_cells
    ):
        y_min = int(round(min(y_positions)))
        y_max = int(round(max(y_positions)))
        x_min = int(round(min(x_positions)))
        x_max = int(round(max(x_positions)))
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        if y_max > y_min and x_max > x_min:
            cropped = image[y_min:y_max, x_min:x_max, :].copy()
            bbox = (x_min, y_min, x_max, y_max)
            if return_bbox:
                return cropped, bbox
            return cropped

    cropped = image
    bbox = (0, 0, w, h)
    if return_bbox:
        return cropped, bbox
    return cropped


def extract_puzzle_region_with_metadata(
    image: np.ndarray,
    **kwargs: object,
) -> PuzzleROIResult:
    """Extract puzzle region and return result with bbox and inferred grid size."""
    _require_cv2()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    h, w = image.shape[:2]
    merge_dist = kwargs.pop("merge_dist", None)
    if merge_dist is None:
        merge_dist = max(2, min(h, w) // 80)
    kwargs["merge_dist"] = merge_dist

    edges = _gray_and_edges(image)
    horizontal_seg, vertical_seg = _detect_line_segments(edges)

    y_positions = _line_positions(horizontal_seg, "y", merge_dist)
    x_positions = _line_positions(vertical_seg, "x", merge_dist)

    period_y = _dominant_period(y_positions) if len(y_positions) >= 2 else None
    period_x = _dominant_period(x_positions) if len(x_positions) >= 2 else None

    min_cells = kwargs.get("min_cells", 2)
    max_cells = kwargs.get("max_cells", 30)
    period_tolerance = kwargs.get("period_tolerance", 0.25)

    if period_y is not None:
        y_positions = _filter_regular_grid(y_positions, period_y, period_tolerance)
    if period_x is not None:
        x_positions = _filter_regular_grid(x_positions, period_x, period_tolerance)

    rows_inferred = _infer_grid_size(y_positions, period_y) if period_y else 0
    cols_inferred = _infer_grid_size(x_positions, period_x) if period_x else 0

    if (
        len(y_positions) >= min_cells + 1
        and len(x_positions) >= min_cells + 1
        and min_cells <= rows_inferred <= max_cells
        and min_cells <= cols_inferred <= max_cells
    ):
        y_min = max(0, int(round(min(y_positions))))
        y_max = min(h, int(round(max(y_positions))))
        x_min = max(0, int(round(min(x_positions))))
        x_max = min(w, int(round(max(x_positions))))
        if y_max > y_min and x_max > x_min:
            cropped = image[y_min:y_max, x_min:x_max, :].copy()
            return PuzzleROIResult(
                image=cropped,
                bbox=(x_min, y_min, x_max, y_max),
                rows=rows_inferred,
                cols=cols_inferred,
            )

    return PuzzleROIResult(
        image=image.copy(),
        bbox=(0, 0, w, h),
        rows=None,
        cols=None,
    )
