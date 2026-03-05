"""Default solver wrapper around JigsawSolver."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import numpy as np

from jigsaw.solver.base import BaseSolver, SolveResult
from jigsaw.splitter import Patch
from jigsaw.matcher import EdgeMatcher
from jigsaw.utils import compose_image_from_grid

# Use lazy import to avoid circular imports
if TYPE_CHECKING:
    from jigsaw.solver import JigsawSolver, SolverConfig


class DefaultSolver(BaseSolver):
    """Wrapper around the existing JigsawSolver algorithm.

    This solver uses edge compatibility heuristics with beam search
    and simulated annealing optimization.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        seed: int = 42,
        local_opt_iters: int = 1000,
        **kwargs
    ):
        """Initialize the default solver.

        Args:
            rows: Number of puzzle rows
            cols: Number of puzzle columns
            seed: Random seed for reproducibility
            local_opt_iters: Number of local optimization iterations
            **kwargs: Additional SolverConfig parameters
        """
        # Import here to avoid circular imports
        from jigsaw.solver import JigsawSolver, SolverConfig

        self.config = SolverConfig(
            rows=rows,
            cols=cols,
            seed=seed,
            local_opt_iters=local_opt_iters,
            **kwargs
        )
        self._solver = JigsawSolver(self.config)
        self._matcher = EdgeMatcher()

    def solve(
        self,
        patches: List[Patch],
        original_image: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> SolveResult:
        """Solve puzzle using the default algorithm.

        Args:
            patches: List of shuffled patches
            original_image: Original image (unused for default solver)
            cost_matrix: Pre-computed cost matrix (optional)
            **kwargs: Additional parameters

        Returns:
            SolveResult with grid and reconstructed image
        """
        # Build cost matrix if not provided
        if cost_matrix is None:
            cost_matrix = self._matcher.build_cost_matrix(patches)

        # Solve using the original JigsawSolver
        grid = self._solver.solve(patches, cost_matrix)

        # Build reconstructed image
        reconstructed = compose_image_from_grid(grid, patches)

        return SolveResult(
            grid=grid,
            reconstructed_image=reconstructed,
            solver_name="default",
            metadata={"cost_matrix": cost_matrix}
        )


# Import JigsawSolver here to avoid circular imports
