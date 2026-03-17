"""Main automation pipeline for iOS jigsaw puzzle solver."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .connector import DeviceConnector
from .gesture import Gesture, Rect
from .planner import MotionPlanner, SwapMove
from .screenshot import Screenshot


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def solve_puzzle(
    image: np.ndarray,
    grid_size: Tuple[int, int] = (8, 8),
    solver_type: str = "default",
    border_width: int = 10,
    robust_method: str = "median",
) -> np.ndarray:
    """Solve the puzzle using reconstruct.py logic.

    Args:
        image: Input puzzle image
        grid_size: Grid dimensions (rows, cols)
        solver_type: Solver algorithm ("default" or "gaps")
        border_width: Number of edge pixels for dissimilarity (gaps solver)
        robust_method: Dissimilarity method ("mse", "median", "percentile", "huber")

    Returns:
        solved_grid: 2D array of piece indices in solved order
    """
    from jigsaw.gap_splitter import split_with_gap_aware
    from jigsaw.solver import SolverFactory

    rows, cols = grid_size

    # Split image into patches
    patches, _, _ = split_with_gap_aware(image, rows=rows, cols=cols)

    # Build solver kwargs
    solver_kwargs = {}
    if solver_type == "gaps":
        patch_h, patch_w = patches[0].image.shape[:2]
        solver_kwargs["piece_size"] = (patch_h, patch_w)
        solver_kwargs["border_width"] = border_width
        solver_kwargs["robust_method"] = robust_method

    # Create solver
    solver = SolverFactory.create(
        solver_type,
        rows=rows,
        cols=cols,
        seed=42,
        **solver_kwargs
    )

    # Solve
    result = solver.solve(patches)
    return result.grid


def get_current_grid(
    screenshot: Screenshot,
    grid_size: Tuple[int, int] = (8, 8),
    solver_type: str = "default",
    puzzle_bbox: Optional[Tuple[int, int, int, int]] = None,
    border_width: int = 10,
    robust_method: str = "median",
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Get current puzzle state from screenshot.

    Args:
        screenshot: Screenshot instance
        grid_size: Grid dimensions
        solver_type: Solver to use
        puzzle_bbox: Pre-defined puzzle bounding box (optional)
        border_width: Number of edge pixels for dissimilarity (gaps solver)
        robust_method: Dissimilarity method ("mse", "median", "percentile", "huber")

    Returns:
        (cropped_image, solved_grid, detected_bbox)
    """
    # Capture screenshot
    image = screenshot.capture()

    # Extract puzzle region
    if puzzle_bbox is None:
        from jigsaw.roi_color import extract_puzzle_region_by_color
        roi_result = extract_puzzle_region_by_color(image)
        cropped = roi_result.image
        detected_bbox = roi_result.bbox
    else:
        x1, y1, x2, y2 = puzzle_bbox
        cropped = image[y1:y2, x1:x2]
        detected_bbox = puzzle_bbox

    # Solve the puzzle
    solved_grid = solve_puzzle(cropped, grid_size, solver_type, border_width, robust_method)

    return cropped, solved_grid, detected_bbox


def run_automation(
    device_url: str = "http://localhost:8100",
    udid: Optional[str] = None,
    grid_size: Tuple[int, int] = (8, 8),
    solver_type: str = "default",
    max_time: float = 210.0,  # 3.5 minutes
    check_interval: float = 5.0,
    output_dir: Optional[Path] = None,
    border_width: int = 10,
    robust_method: str = "median",
) -> None:
    """Run the full automation pipeline.

    Args:
        device_url: WDA server URL
        udid: Device UDID for USB connection
        grid_size: Puzzle grid size (rows, cols)
        solver_type: Solver algorithm
        max_time: Maximum runtime in seconds
        check_interval: Time between puzzle checks
        output_dir: Directory to save debug screenshots
        border_width: Number of edge pixels for dissimilarity (gaps solver)
        robust_method: Dissimilarity method ("mse", "median", "percentile", "huber")
    """
    print("=" * 50)
    print("iOS Jigsaw Puzzle Solver - Automation Started")
    print("=" * 50)

    # Setup output directory
    if output_dir is None:
        output_dir = Path("ios_auto_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to device
    print("\n[1/6] Connecting to iOS device...")
    connector = DeviceConnector(url=device_url, udid=udid)

    if not connector.connect():
        print("Failed to connect to device!")
        return

    try:
        # Initialize components
        screenshot = Screenshot(connector)
        gesture = Gesture(connector, grid_size=grid_size)
        planner = MotionPlanner(grid_size=grid_size)

        # Get initial puzzle state
        print("\n[2/6] Capturing initial puzzle state...")
        start_time = time.time()

        cropped, solved_grid, bbox = get_current_grid(
            screenshot, grid_size, solver_type, border_width=border_width, robust_method=robust_method
        )

        if bbox is None:
            print("ERROR: Could not detect puzzle region!")
            return

        # Set puzzle bounding box
        gesture.set_puzzle_bbox(bbox)
        print(f"    Puzzle bbox: {bbox}")
        print(f"    Solved grid:\n{solved_grid}")

        # Save initial state
        import cv2

        cv2.imwrite(
            str(output_dir / "initial_puzzle.png"),
            cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
        )

        # Main loop
        print("\n[3/6] Starting solve loop...")
        iteration = 0

        while time.time() - start_time < max_time:
            iteration += 1
            elapsed = time.time() - start_time

            print(f"\n--- Iteration {iteration} (elapsed: {elapsed:.1f}s) ---")

            # Capture current state
            cropped, solved_grid, _ = get_current_grid(
                screenshot, grid_size, solver_type, bbox, border_width, robust_method
            )

            # Generate move plan (assuming identity current grid - pieces are shuffled)
            # In reality, we'd need to analyze the current screenshot to determine
            # the actual current positions
            moves = planner.plan_greedy(solved_grid.tolist())
            print(f"    Planned {len(moves)} moves")

            if not moves:
                print("\n[SUCCESS] Puzzle appears to be solved!")
                break

            # Execute moves in reverse order (last move first)
            # This is because later moves don't affect earlier positions
            print(f"    Executing moves...")
            for i, move in enumerate(reversed(moves)):
                if time.time() - start_time >= max_time:
                    print("    Time limit reached!")
                    break

                print(f"    Move {i+1}/{len(moves)}: {move}")
                gesture.swap_pieces(move.from_pos, move.to_pos)
                time.sleep(0.3)  # Wait for animation

            # Save debug screenshot
            cv2.imwrite(
                str(output_dir / f"iteration_{iteration}.png"),
                cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
            )

            # Wait before next iteration
            time.sleep(check_interval)

        # Check final state
        print("\n[4/6] Checking final state...")
        elapsed = time.time() - start_time

        if elapsed >= max_time:
            print(f"    Time limit ({max_time}s) reached!")
        else:
            print(f"    Completed in {elapsed:.1f}s")

        # Capture final state
        final_image = screenshot.capture()
        cv2.imwrite(
            str(output_dir / "final_puzzle.png"),
            cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR),
        )

        print("\n[5/6] Automation completed!")
        print(f"    Total iterations: {iteration}")
        print(f"    Output saved to: {output_dir}")

    finally:
        print("\n[6/6] Disconnecting device...")
        connector.disconnect()

    print("\n" + "=" * 50)
    print("Automation Finished")
    print("=" * 50)


def test_offline(
    image_path: Path,
    grid_size: Tuple[int, int] = (8, 8),
    solver_type: str = "default",
    output_dir: Optional[Path] = None,
    border_width: int = 10,
    robust_method: str = "median",
) -> None:
    """Test the solver offline without device.

    Args:
        image_path: Path to puzzle screenshot
        grid_size: Grid dimensions
        solver_type: Solver algorithm
        output_dir: Output directory
        border_width: Number of edge pixels for dissimilarity (gaps solver)
        robust_method: Dissimilarity method ("mse", "median", "percentile", "huber")
    """
    print("=" * 50)
    print("iOS Jigsaw Puzzle Solver - Offline Test")
    print("=" * 50)

    if output_dir is None:
        output_dir = Path("ios_auto_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    print(f"    Image shape: {image.shape}")

    # Extract puzzle region using color-based method
    from jigsaw.roi_color import extract_puzzle_region_by_color

    roi_result = extract_puzzle_region_by_color(image)
    cropped = roi_result.image
    print(f"    Cropped shape: {cropped.shape}")
    print(f"    BBox: {roi_result.bbox}")

    # Solve - use solver's reconstructed image directly
    print("\nSolving puzzle...")
    print(f"    Solver: {solver_type}")
    print(f"    Border width: {border_width}")
    print(f"    Robust method: {robust_method}")

    from jigsaw.gap_splitter import split_with_gap_aware
    from jigsaw.solver import SolverFactory

    rows, cols = grid_size
    patches, _, _ = split_with_gap_aware(cropped, rows=rows, cols=cols)

    # Build solver kwargs
    solver_kwargs = {}
    if solver_type == "gaps":
        patch_h, patch_w = patches[0].image.shape[:2]
        solver_kwargs["piece_size"] = (patch_h, patch_w)
        solver_kwargs["border_width"] = border_width
        solver_kwargs["robust_method"] = robust_method

    solver = SolverFactory.create(solver_type, rows=rows, cols=cols, seed=42, **solver_kwargs)
    result = solver.solve(patches)
    solved_grid = result.grid
    reconstructed = result.reconstructed_image

    print(f"    Solved grid:\n{solved_grid}")

    # Save result
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cropped)
    axes[0].set_title("Input Puzzle")
    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_result.png")
    print(f"\nResult saved to: {output_dir / 'reconstruction_result.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="iOS Jigsaw Puzzle Solver")
    parser.add_argument("--test-offline", type=Path, help="Test without device")
    parser.add_argument("--grid", type=lambda x: tuple(map(int, x.split("x"))), default=(8, 8))
    parser.add_argument("--solver", default="default", choices=["default", "gaps"])
    parser.add_argument("--border-width", type=int, default=10, help="Border width for gaps solver (default: 10)")
    parser.add_argument("--robust-method", default="median", choices=["mse", "median", "percentile", "huber"],
                        help="Robust method for gaps solver (default: median)")
    parser.add_argument("--output", type=Path, default=Path("ios_auto_output"))
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--udid", default=None)
    parser.add_argument("--max-time", type=float, default=210.0)
    parser.add_argument("--interval", type=float, default=5.0)

    args = parser.parse_args()

    if args.test_offline:
        test_offline(
            args.test_offline,
            args.grid,
            args.solver,
            args.output,
            border_width=args.border_width,
            robust_method=args.robust_method,
        )
    else:
        run_automation(
            device_url=args.url,
            udid=args.udid,
            grid_size=args.grid,
            solver_type=args.solver,
            max_time=args.max_time,
            check_interval=args.interval,
            output_dir=args.output,
            border_width=args.border_width,
            robust_method=args.robust_method,
        )
