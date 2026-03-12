"""Gaps solver wrapper around GeneticAlgorithm."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

# Try to import imageio for GIF creation
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add gaps to path
_gaps_path = Path(__file__).parent.parent.parent / "thirdparty" / "gaps"
if str(_gaps_path) not in sys.path:
    sys.path.insert(0, str(_gaps_path))

from gaps.genetic_algorithm import GeneticAlgorithm
from gaps.individual import Individual

from jigsaw.solver.base import BaseSolver, SolveResult
from jigsaw.splitter import Patch
from jigsaw.utils import compose_image_from_grid


class GapsSolver(BaseSolver):
    """Wrapper around the gaps GeneticAlgorithm solver.

    This solver uses a genetic algorithm with crossover operations
    to solve jigsaw puzzles.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        piece_size: Optional[Union[int, Tuple[int, int]]] = None,
        population_size: int = 200,
        generations: int = 20,
        elite_size: int = 2,
        visualize: bool = False,
        save_gif: Optional[str] = None,
        **kwargs
    ):
        """Initialize the gaps solver.

        Args:
            rows: Number of puzzle rows
            cols: Number of puzzle columns
            piece_size: Size of each puzzle piece in pixels. Can be:
                       - int: square piece (piece_size x piece_size)
                       - tuple (height, width): rectangular piece
                       - None: auto-calculate from image dimensions
            population_size: Number of individuals in genetic algorithm
            generations: Number of evolution generations
            elite_size: Number of elite individuals to preserve
            visualize: Whether to show real-time visualization of solving process
            save_gif: Path to save GIF animation of solving process (requires imageio)
            **kwargs: Additional parameters (ignored)
        """
        self.rows = rows
        self.cols = cols
        self._piece_size = piece_size
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.visualize = visualize
        self.save_gif = save_gif

    def solve(
        self,
        patches: List[Patch],
        original_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> SolveResult:
        """Solve puzzle using the gaps genetic algorithm.

        Args:
            patches: List of shuffled patches
            original_image: Original image (will be reconstructed from patches if not provided)
            **kwargs: Additional parameters (ignored)

        Returns:
            SolveResult with grid and reconstructed image
        """
        # Reconstruct original image from patches if not provided
        if original_image is None:
            sorted_patches = sorted(patches, key=lambda p: p.original_index)
            original_image = compose_image_from_grid(
                np.arange(len(patches)).reshape(self.rows, self.cols),
                sorted_patches
            )

        # Calculate piece_size - supports both square (int) and rectangular (tuple)
        if self._piece_size is not None:
            piece_size = self._piece_size
        else:
            # Auto-calculate piece_size from image dimensions to match grid
            height, width = original_image.shape[:2]
            # Use integer division to ensure pieces fit exactly
            piece_h = height // self.rows
            piece_w = width // self.cols
            # Use tuple for rectangular pieces, or int for square
            if piece_h == piece_w:
                piece_size = piece_h
            else:
                piece_size = (piece_h, piece_w)

        # Prepare for visualization/GIF if needed
        frames: List[np.ndarray] = []
        fitness_history: List[float] = []
        fig = None
        ax = None

        if self.visualize or self.save_gif:
            if self.visualize and not HAS_MATPLOTLIB:
                print("Warning: matplotlib not available, disabling visualization")
                self.visualize = False
            if self.save_gif and not HAS_IMAGEIO:
                print("Warning: imageio not available, disabling GIF saving")
                self.save_gif = None

            if self.visualize:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_axis_off()
                plt.ion()
                plt.show()

        def generation_callback(generation: int, fitness: float, image: np.ndarray):
            """Callback called after each generation."""
            fitness_history.append(fitness)

            if self.save_gif:
                frames.append(image.copy())

            if self.visualize and fig is not None and ax is not None:
                ax.clear()
                ax.set_axis_off()
                ax.set_title(f"Generation: {generation + 1}/{self.generations}\nFitness: {fitness:.2f}")
                ax.imshow(image)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)

        # Run genetic algorithm
        ga = GeneticAlgorithm(
            image=original_image,
            piece_size=piece_size,
            population_size=self.population_size,
            generations=self.generations,
            elite_size=self.elite_size
        )
        result_individual: Individual = ga.start_evolution(
            verbose=False,
            generation_callback=generation_callback if (self.visualize or self.save_gif) else None
        )

        # Save GIF if requested
        if self.save_gif and frames:
            print(f"Saving GIF to {self.save_gif}...")
            # Duplicate first frame for better viewing and add final result
            full_frames = [frames[0]] * 5 + frames
            if result_individual.to_image() is not None:
                full_frames.append(result_individual.to_image())
            imageio.mimsave(self.save_gif, full_frames, duration=0.2)
            print(f"GIF saved: {self.save_gif}")

        # Close visualization window
        if self.visualize and fig is not None:
            plt.ioff()
            plt.close(fig)

        # Convert Individual to grid
        grid = self._individual_to_grid(result_individual, patches)

        return SolveResult(
            grid=grid,
            reconstructed_image=result_individual.to_image(),
            solver_name="gaps",
            metadata={
                "fitness": result_individual.fitness,
                "fitness_history": fitness_history if (self.visualize or self.save_gif) else None
            }
        )

    def _individual_to_grid(
        self,
        individual: Individual,
        original_patches: List[Patch]
    ) -> np.ndarray:
        """Convert Individual to grid of patch indices.

        Args:
            individual: Gaps Individual solution
            original_patches: Original shuffled patches

        Returns:
            Grid array where each position contains the index of the patch
        """
        # Get actual dimensions from individual
        rows = individual.rows
        cols = individual.columns
        total_pieces = rows * cols
        target_pieces = self.rows * self.cols

        # Check if piece count matches
        if total_pieces != target_pieces:
            raise ValueError(
                f"Piece count mismatch: gaps returned {total_pieces} pieces "
                f"({rows}x{cols}), but we have {target_pieces} patches "
                f"(requested {self.rows}x{self.cols}). "
                f"Try using a different piece_size or use the 'default' solver instead."
            )

        grid = np.zeros((rows, cols), dtype=int)

        # Create mapping from piece id to original patch index
        # piece id in gaps corresponds to original_index of patches
        id_to_index = {p.original_index: i for i, p in enumerate(original_patches)}

        for row in range(rows):
            for col in range(cols):
                piece = individual[row][col]
                if piece.id not in id_to_index:
                    raise KeyError(
                        f"Piece id {piece.id} not found in original patches. "
                        f"Available ids: {list(id_to_index.keys())[:10]}..."
                    )
                grid[row, col] = id_to_index[piece.id]

        return grid
