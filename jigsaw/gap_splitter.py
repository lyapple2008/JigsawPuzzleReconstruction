"""Gap-aware splitting helpers for shuffled puzzle images with visible seams."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .splitter import Patch

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def _axis_texture_signal(gray: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return gray.std(axis=1).astype(np.float32)
    if axis == 1:
        return gray.std(axis=0).astype(np.float32)
    raise ValueError("axis must be 0 (rows) or 1 (cols)")


def _find_separator_bands(signal: np.ndarray, parts: int) -> List[Tuple[int, int]]:
    if parts <= 1:
        return []
    n = int(signal.shape[0])
    if n < parts:
        raise ValueError("image axis is too small for the requested grid")

    low_q = float(np.quantile(signal, 0.2))
    bands: List[Tuple[int, int]] = []
    for i in range(1, parts):
        target = int(round(i * n / parts))
        radius = max(3, n // (parts * 8))
        lo = max(1, target - radius)
        hi = min(n - 2, target + radius)
        if lo > hi:
            lo = max(1, min(target, n - 2))
            hi = lo

        local = signal[lo : hi + 1]
        idx = int(lo + int(np.argmin(local)))
        local_median = float(np.median(local))
        if local_median > 0 and float(signal[idx]) > 0.85 * local_median:
            bands.append((target, target - 1))
            continue

        threshold = min(low_q, float(signal[idx]) * 1.2)
        left = idx
        right = idx
        max_half = max(1, n // (parts * 4))
        while left > 1 and idx - left < max_half and signal[left - 1] <= threshold:
            left -= 1
        while right < n - 2 and right - idx < max_half and signal[right + 1] <= threshold:
            right += 1
        bands.append((left, right))
    return bands


def _bands_to_ranges(length: int, bands: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    prev_end = -1
    for start, end in bands:
        cell_start = prev_end + 1
        cell_end = max(cell_start, start - 1)
        ranges.append((cell_start, cell_end))
        prev_end = end

    last_start = min(prev_end + 1, length - 1)
    ranges.append((last_start, length - 1))
    return ranges


def _normalize_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    sizes = [end - start + 1 for start, end in ranges]
    target = int(min(sizes))
    if target < 2:
        raise ValueError("detected patch size is too small; please check rows/cols")

    normalized: List[Tuple[int, int]] = []
    for start, end in ranges:
        center = (start + end) // 2
        half = target // 2
        if target % 2 == 0:
            new_start = center - half + 1
            new_end = center + half
        else:
            new_start = center - half
            new_end = center + half
        normalized.append((new_start, new_end))
    return normalized


def split_with_gap_aware(
    image: np.ndarray, rows: int, cols: int
) -> Tuple[List[Patch], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split image into patches while being robust to visible gap seams."""
    gray = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if cv2 is not None
        else np.mean(image.astype(np.float32), axis=2).astype(np.float32)
    )
    row_signal = _axis_texture_signal(gray, axis=0)
    col_signal = _axis_texture_signal(gray, axis=1)

    row_bands = _find_separator_bands(row_signal, rows)
    col_bands = _find_separator_bands(col_signal, cols)
    row_ranges = _normalize_ranges(_bands_to_ranges(image.shape[0], row_bands))
    col_ranges = _normalize_ranges(_bands_to_ranges(image.shape[1], col_bands))

    patches: List[Patch] = []
    index = 0
    for r in range(rows):
        y0, y1 = row_ranges[r]
        for c in range(cols):
            x0, x1 = col_ranges[c]
            patch_img = image[y0 : y1 + 1, x0 : x1 + 1, :].copy()
            patches.append(Patch.from_image(patch_img, original_index=index))
            index += 1
    return patches, row_ranges, col_ranges
