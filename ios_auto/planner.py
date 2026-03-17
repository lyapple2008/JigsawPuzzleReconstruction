"""Motion planning for puzzle piece swaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SwapMove:
    """A single swap move operation."""

    from_pos: Tuple[int, int]  # (row, col)
    to_pos: Tuple[int, int]  # (row, col)

    def __repr__(self) -> str:
        return f"SwapMove({self.from_pos} -> {self.to_pos})"


class MotionPlanner:
    """Plans motion sequences to solve the puzzle."""

    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        """Initialize motion planner.

        Args:
            grid_size: Puzzle grid size (rows, cols)
        """
        self.rows, self.cols = grid_size
        self.grid_size = grid_size

    def plan_from_solved_grid(
        self,
        current_grid: List[List[int]],
        solved_grid: List[List[int]],
    ) -> List[SwapMove]:
        """Plan swap moves to transform current arrangement to solved state.

        Args:
            current_grid: Current piece arrangement (rows x cols 2D list)
                         value at (i,j) is the piece index at that position
            solved_grid: Target solved arrangement (same format)

        Returns:
            List of SwapMove operations to solve the puzzle
        """
        moves: List[SwapMove] = []

        # Create a working copy
        grid = [row[:] for row in current_grid]

        # Strategy: Process from last position to first
        # For each position, if the piece is wrong, swap it with the correct position
        for row in range(self.rows - 1, -1, -1):
            for col in range(self.cols - 1, -1, -1):
                target_piece = solved_grid[row][col]
                current_piece = grid[row][col]

                if current_piece != target_piece:
                    # Find where the target piece currently is
                    for r in range(self.rows):
                        for c in range(self.cols):
                            if grid[r][c] == target_piece:
                                # Swap: move target_piece to (row, col)
                                # and current_piece to (r, c)
                                if (r, c) != (row, col):
                                    moves.append(SwapMove(
                                        from_pos=(row, col),
                                        to_pos=(r, c),
                                    ))
                                    # Perform the swap in our working grid
                                    grid[row][col], grid[r][c] = grid[r][c], grid[row][col]
                                break

        return moves

    def plan_greedy(self, solved_grid: List[List[int]]) -> List[SwapMove]:
        """Plan moves assuming current grid is identity (pieces in order).

        This is a simpler version that generates moves to place pieces
        in the correct positions from top-left to bottom-right.

        Args:
            solved_grid: Target solved arrangement

        Returns:
            List of SwapMove operations
        """
        moves: List[SwapMove] = []

        # Start from the last position and work backwards
        for target_row in range(self.rows - 1, -1, -1):
            for target_col in range(self.cols - 1, -1, -1):
                target_piece = solved_grid[target_row][target_col]

                # The piece should be at position (target_piece // cols, target_piece % cols)
                # in a solved grid
                expected_row = target_piece // self.cols
                expected_col = target_piece % self.cols

                # Skip if already in correct position
                if (target_row, target_col) == (expected_row, expected_col):
                    continue

                # Move the correct piece to this position
                moves.append(SwapMove(
                    from_pos=(target_row, target_col),
                    to_pos=(expected_row, expected_col),
                ))

        return moves

    def execute_moves(
        self,
        moves: List[SwapMove],
        gesture,
    ) -> None:
        """Execute a list of moves using gesture controller.

        Args:
            moves: List of SwapMove operations
            gesture: Gesture controller instance
        """
        for move in moves:
            gesture.swap_pieces(move.from_pos, move.to_pos)
            # Small delay between moves
            import time
            time.sleep(0.1)
