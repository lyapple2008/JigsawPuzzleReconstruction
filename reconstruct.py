"""Reconstruct a shuffled jigsaw image from a grid split configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from jigsaw.gap_splitter import split_with_gap_aware
from jigsaw.matcher import EdgeMatcher
from jigsaw.solver import JigsawSolver, SolverConfig
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
    parser.add_argument(
        "--use-position-prior",
        action="store_true",
        help="Enable lightweight learned coordinate prior during solving",
    )
    parser.add_argument(
        "--prior-samples",
        type=int,
        default=24,
        help="Synthetic training sample count for position prior (default: 24)",
    )
    parser.add_argument(
        "--auto-position-prior",
        action="store_true",
        help="Use position prior only when matcher ambiguity is high",
    )
    return parser.parse_args()


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
            use_position_prior=args.use_position_prior,
            auto_position_prior=args.auto_position_prior,
            position_prior_samples=args.prior_samples,
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
