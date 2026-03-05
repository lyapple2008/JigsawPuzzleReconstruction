"""Modular solver framework with pluggable algorithms."""

from .base import BaseSolver, SolveResult
from .factory import SolverFactory

# Import original solver classes
from .jigsaw_solver import JigsawSolver, SolverConfig

# Register solvers
from .default_solver import DefaultSolver
from .gaps_solver import GapsSolver

SolverFactory.register("default", DefaultSolver)
SolverFactory.register("gaps", GapsSolver)

__all__ = [
    "BaseSolver",
    "SolveResult",
    "SolverFactory",
    "DefaultSolver",
    "GapsSolver",
    "JigsawSolver",
    "SolverConfig",
]
