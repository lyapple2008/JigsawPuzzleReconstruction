"""Square jigsaw puzzle reconstruction package."""

from .evaluator import EvaluationResult, PuzzleEvaluator
from .gap_splitter import split_with_gap_aware
from .matcher import Direction, EdgeMatcher
from .position_prior import build_position_penalty, train_position_prior
from .puzzle_roi import PuzzleROIResult, extract_puzzle_region, extract_puzzle_region_with_metadata
from .solver import JigsawSolver, SolverConfig
from .splitter import Patch, PuzzleSplitter

__all__ = [
    "Patch",
    "PuzzleSplitter",
    "Direction",
    "EdgeMatcher",
    "split_with_gap_aware",
    "train_position_prior",
    "build_position_penalty",
    "SolverConfig",
    "JigsawSolver",
    "EvaluationResult",
    "PuzzleEvaluator",
    "PuzzleROIResult",
    "extract_puzzle_region",
    "extract_puzzle_region_with_metadata",
]
