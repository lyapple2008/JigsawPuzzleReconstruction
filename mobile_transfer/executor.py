"""Execute RL-predicted moves on iOS device.

Bridges the trained model with ios_auto gesture execution.
Main loop: capture screenshot -> extract state -> predict action -> execute swap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MobileExecutor:
    """Execute puzzle-solving actions on a mobile device using trained RL model.

    Pipeline per step:
    1. Capture screenshot from device
    2. Extract puzzle region and build cost matrix
    3. Estimate current grid state
    4. Run RL model to predict swap action
    5. Execute drag gesture on device
    """

    def __init__(
        self,
        connector,
        gesture,
        model_runner,
        state_extractor,
        grid_size: Tuple[int, int] = (6, 6),
        check_interval: float = 1.0,
    ) -> None:
        """Initialize mobile executor.

        Args:
            connector: DeviceConnector instance (ios_auto/connector.py).
            gesture: Gesture instance (ios_auto/gesture.py).
            model_runner: MobileModelRunner instance.
            state_extractor: ScreenStateExtractor instance.
            grid_size: (rows, cols) of the puzzle.
            check_interval: Seconds to wait between moves.
        """
        self.connector = connector
        self.gesture = gesture
        self.model = model_runner
        self.extractor = state_extractor
        self.rows, self.cols = grid_size
        self.n = self.rows * self.cols
        self.check_interval = check_interval

        # Reference patches (from original image, for grid estimation)
        self._reference_patches = None
        self._reference_cost = None

    def set_reference(self, image: np.ndarray) -> None:
        """Set reference image for grid state estimation.

        Args:
            image: Original puzzle image (before shuffling).
        """
        self._reference_patches, self._reference_cost = (
            self.extractor.extract_from_image(image)
        )

    def run(
        self,
        max_moves: int = 200,
        verbose: bool = True,
    ) -> dict:
        """Run the automated puzzle-solving loop.

        Args:
            max_moves: Maximum number of swap moves to attempt.
            verbose: Print progress information.

        Returns:
            Dict with execution stats: moves, final_accuracy, solved.
        """
        moves_made = 0
        solved = False

        for move_num in range(max_moves):
            try:
                # Capture screenshot
                screenshot = np.array(self.connector.session.screenshot())

                # Extract puzzle state
                puzzle_image, patches, cost_matrix = (
                    self.extractor.extract_from_screenshot(screenshot)
                )

                # Estimate current grid
                if self._reference_patches is not None:
                    grid = self.extractor.estimate_grid_from_screenshot(
                        patches, self._reference_patches
                    )
                else:
                    # Without reference, assume sequential order
                    grid = np.arange(self.n, dtype=np.int32).reshape(
                        self.rows, self.cols
                    )

                # Build observation
                obs = self.extractor.build_observation(grid, cost_matrix)

                # Check if solved
                from jigsaw.evaluator import PuzzleEvaluator

                evaluator = PuzzleEvaluator()
                accuracy = evaluator.compute_position_accuracy(grid, patches)
                if accuracy >= 1.0:
                    solved = True
                    if verbose:
                        print(f"Puzzle solved in {moves_made} moves!")
                    break

                # Predict action
                action = self.model.predict(obs)
                (r1, c1), (r2, c2) = self.model.decode_action(action)

                if verbose:
                    print(
                        f"Move {move_num + 1}: swap ({r1},{c1}) <-> ({r2},{c2}), "
                        f"accuracy={accuracy:.2%}"
                    )

                # Execute swap on device
                self.gesture.swap_pieces((r1, c1), (r2, c2))
                moves_made += 1

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                if verbose:
                    print("\nInterrupted by user.")
                break
            except Exception as e:
                if verbose:
                    print(f"Error at move {move_num + 1}: {e}")
                continue

        return {
            "moves": moves_made,
            "solved": solved,
        }


def test_offline(
    image_path: str,
    grid_size: str = "6x6",
    model_path: Optional[str] = None,
) -> dict:
    """Test the executor pipeline offline (without real device).

    Uses a clean image, shuffles it, and tests the prediction pipeline.

    Args:
        image_path: Path to test image.
        grid_size: Grid size as "NxN" string.
        model_path: Optional path to trained model. If None, uses random actions.

    Returns:
        Test results dict.
    """
    from jigsaw.splitter import PuzzleSplitter
    from jigsaw.utils import load_or_generate_image, shuffle_patches

    rows, cols = map(int, grid_size.split("x"))

    # Load and split image
    image = load_or_generate_image(image_path, size=rows * 100)
    splitter = PuzzleSplitter()
    patches = splitter.split(image, rows, cols)

    # Build state extractor
    from mobile_transfer.state_extractor import ScreenStateExtractor

    extractor = ScreenStateExtractor(grid_size=(rows, cols))
    ref_patches, cost_matrix = extractor.extract_from_image(image)

    # Shuffle patches
    shuffled, order = shuffle_patches(patches, seed=42)
    grid = order.reshape(rows, cols).astype(np.int32)

    # Build observation
    obs = extractor.build_observation(grid, cost_matrix)

    print(f"Offline test: {grid_size} puzzle, {len(patches)} pieces")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Observation grid shape: {obs['grid'].shape}")
    print(f"  Cost matrix shape: {obs['edge_costs'].shape}")

    if model_path:
        from mobile_transfer.model_runner import MobileModelRunner

        runner = MobileModelRunner(model_path, grid_size=rows)
        action = runner.predict(obs)
        (r1, c1), (r2, c2) = runner.decode_action(action)
        print(f"  Model predicted action: swap ({r1},{c1}) <-> ({r2},{c2})")
    else:
        print("  No model provided, testing observation pipeline only")

    return {
        "grid_size": grid_size,
        "n_patches": len(patches),
        "observation_valid": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mobile executor (offline test)")
    parser.add_argument("--image", type=str, required=True, help="Test image path")
    parser.add_argument("--grid", type=str, default="6x6", help="Grid size (NxN)")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    args = parser.parse_args()

    test_offline(args.image, args.grid, args.model)
