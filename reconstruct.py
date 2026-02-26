"""Reconstruct a shuffled jigsaw image from a grid split configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
from jigsaw.splitter import Patch
from jigsaw.utils import compose_image_from_grid

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


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8 array."""
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to load image from path: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    import matplotlib.image as mpimg

    image = mpimg.imread(path)
    if image is None:
        raise ValueError(f"failed to load image from path: {path}")
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save RGB image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise ValueError(f"failed to write image to path: {path}")
        return

    import matplotlib.pyplot as plt

    plt.imsave(path, image.astype(np.uint8))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reconstruct a shuffled jigsaw image.")
    parser.add_argument("--image", required=True, help="Path to shuffled input image")
    parser.add_argument("--grid", type=parse_grid, required=True, help="Grid format: ROWSxCOLS")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for reconstructed image (default: do not save)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--local-opt-iters",
        type=int,
        default=1000,
        help="Local swap optimization iterations (default: 1000)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display shuffled and reconstructed images",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Overlay detected grid boundaries on the shuffled input image when showing",
    )
    return parser.parse_args()


def _axis_texture_signal(gray: np.ndarray, axis: int) -> np.ndarray:
    """Return per-row or per-column texture signal based on std-dev."""
    if axis == 0:
        return gray.std(axis=1).astype(np.float32)
    if axis == 1:
        return gray.std(axis=0).astype(np.float32)
    raise ValueError("axis must be 0 (rows) or 1 (cols)")


def _find_separator_bands(signal: np.ndarray, parts: int) -> List[Tuple[int, int]]:
    """Locate low-texture separator bands near expected split boundaries."""
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
            # No obvious separator in this neighborhood: keep a clean split at target.
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
    """Convert separator bands into content ranges between separators."""
    ranges: List[Tuple[int, int]] = []
    prev_end = -1
    for start, end in bands:
        cell_start = prev_end + 1
        cell_end = start - 1
        if cell_end < cell_start:
            cell_end = cell_start
        ranges.append((cell_start, cell_end))
        prev_end = end
    last_start = prev_end + 1
    if last_start > length - 1:
        last_start = length - 1
    ranges.append((last_start, length - 1))
    return ranges


def _normalize_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Center-crop each range so all ranges share the same size."""
    sizes = [end - start + 1 for start, end in ranges]
    target = int(min(sizes))
    if target < 2:
        raise ValueError("detected patch size is too small; please check --grid setting")

    normalized: List[Tuple[int, int]] = []
    for start, end in ranges:
        size = end - start + 1
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
    """Split image into grid patches while being robust to visible gaps."""
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


def main() -> None:
    """Run reconstruction pipeline from shuffled image to reconstructed output image."""
    args = parse_args()
    rows, cols = args.grid
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else None

    image = load_image(image_path)
    patches, row_ranges, col_ranges = split_with_gap_aware(image, rows=rows, cols=cols)

    matcher = EdgeMatcher()
    cost_matrix = matcher.build_cost_matrix(patches)
    solver = JigsawSolver(
        SolverConfig(
            rows=rows,
            cols=cols,
            seed=args.seed,
            local_opt_iters=args.local_opt_iters,
        )
    )
    solved_grid = solver.solve(patches, cost_matrix=cost_matrix)

    reconstructed = compose_image_from_grid(solved_grid, patches)
    if output_path is not None:
        save_image(output_path, reconstructed)

    print(f"Input image: {image_path}")
    print(f"Grid: {rows}x{cols}")
    print(f"Patch size used: {patches[0].image.shape[0]}x{patches[0].image.shape[1]}")
    if output_path is not None:
        print(f"Output image: {output_path.resolve()}")
    else:
        print("Output image: not saved (no --output specified)")
    print("Solved grid indices:")
    print(solved_grid)

    if not args.no_show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        display_image = image.copy()
        if args.show_grid:
            for y0, y1 in row_ranges:
                display_image[max(0, y0 - 1) : min(display_image.shape[0], y0 + 1), :, :] = (255, 0, 0)
                display_image[max(0, y1 - 1) : min(display_image.shape[0], y1 + 1), :, :] = (255, 0, 0)
            for x0, x1 in col_ranges:
                display_image[:, max(0, x0 - 1) : min(display_image.shape[1], x0 + 1), :] = (255, 0, 0)
                display_image[:, max(0, x1 - 1) : min(display_image.shape[1], x1 + 1), :] = (255, 0, 0)

        axes[0].imshow(display_image)
        axes[0].set_title("Shuffled Input")
        axes[1].imshow(reconstructed)
        axes[1].set_title("Reconstructed")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
