"""Factory for creating solver instances."""

from __future__ import annotations

from typing import Dict, Type, List, Any

from .base import BaseSolver


class SolverFactory:
    """Factory for creating solver instances by name."""

    _solvers: Dict[str, Type[BaseSolver]] = {}

    @classmethod
    def register(cls, name: str, solver_class: Type[BaseSolver]) -> None:
        """Register a solver by name.

        Args:
            name: Solver identifier (e.g., "default", "gaps")
            solver_class: Solver class (must inherit from BaseSolver)
        """
        cls._solvers[name] = solver_class

    @classmethod
    def create(cls, name: str, **config) -> BaseSolver:
        """Create a solver instance by name.

        Args:
            name: Solver identifier
            **config: Configuration parameters for the solver

        Returns:
            Solver instance

        Raises:
            ValueError: If solver name is not registered
        """
        if name not in cls._solvers:
            available = list(cls._solvers.keys())
            raise ValueError(f"Unknown solver: {name}. Available: {available}")
        return cls._solvers[name](**config)

    @classmethod
    def list_solvers(cls) -> List[str]:
        """List all registered solver names.

        Returns:
            List of available solver names
        """
        return list(cls._solvers.keys())
