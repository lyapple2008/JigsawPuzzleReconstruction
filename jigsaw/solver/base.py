"""Abstract base classes for puzzle solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional
import numpy as np

from jigsaw.splitter import Patch


@dataclass
class SolveResult:
    """Unified result structure for all solvers."""

    grid: np.ndarray  # (rows, cols) grid of patch indices
    reconstructed_image: np.ndarray  # RGB image
    solver_name: str
    metadata: dict  # solver-specific info (time, fitness, etc.)


class BaseSolver(ABC):
    """Abstract base class for puzzle solvers."""

    @abstractmethod
    def solve(
        self,
        patches: List[Patch],
        original_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> SolveResult:
        """Solve puzzle from shuffled patches.

        Args:
            patches: List of shuffled patches
            original_image: Original image (optional, needed for some solvers)
            **kwargs: Additional solver-specific parameters

        Returns:
            SolveResult containing grid and reconstructed image
        """
        pass
