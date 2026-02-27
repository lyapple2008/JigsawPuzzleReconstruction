"""Square jigsaw puzzle reconstruction package."""

from .evaluator import EvaluationResult, PuzzleEvaluator
from .gap_splitter import split_with_gap_aware
from .matcher import Direction, EdgeMatcher
from .solver import JigsawSolver, SolverConfig
from .splitter import Patch, PuzzleSplitter

__all__ = [
    "Patch",
    "PuzzleSplitter",
    "Direction",
    "EdgeMatcher",
    "split_with_gap_aware",
    "SolverConfig",
    "JigsawSolver",
    "EvaluationResult",
    "PuzzleEvaluator",
]
