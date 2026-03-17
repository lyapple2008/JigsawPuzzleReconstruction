"""Gesture control for iOS device - drag and swipe operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .connector import DeviceConnector


@dataclass
class Point:
    """2D point with pixel coordinates."""

    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Rect:
    """Rectangle defined by top-left and bottom-right corners."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Point:
        """Get center point of rectangle."""
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Gesture:
    """Gesture control for iOS device."""

    def __init__(
        self,
        connector: DeviceConnector,
        puzzle_bbox: Optional[Rect] = None,
        grid_size: Tuple[int, int] = (8, 8),
    ):
        """Initialize gesture control.

        Args:
            connector: Device connector instance
            puzzle_bbox: Bounding box of puzzle area (optional, can be set later)
            grid_size: Puzzle grid size (rows, cols)
        """
        self.connector = connector
        self.puzzle_bbox = puzzle_bbox
        self.grid_size = grid_size  # (rows, cols)

    def set_puzzle_bbox(self, bbox: Tuple[int, int, int, int]) -> None:
        """Set puzzle bounding box.

        Args:
            bbox: (x_min, y_min, x_max, y_max)
        """
        self.puzzle_bbox = Rect(bbox[0], bbox[1], bbox[2], bbox[3])

    def _get_puzzle_rect(self) -> Rect:
        """Get puzzle rectangle, raising if not set."""
        if self.puzzle_bbox is None:
            raise ValueError("puzzle_bbox not set. Call set_puzzle_bbox() first.")
        return self.puzzle_bbox

    def grid_to_pixel(
        self,
        row: int,
        col: int,
        padding: float = 0.5,
    ) -> Point:
        """Convert grid position to pixel coordinates.

        Args:
            row: Row index (0 to rows-1)
            col: Column index (0 to cols-1)
            padding: Padding ratio within each cell (0 to 1)

        Returns:
            Point with pixel coordinates
        """
        rect = self._get_puzzle_rect()
        rows, cols = self.grid_size

        cell_w = rect.width / cols
        cell_h = rect.height / rows

        # Calculate cell center with padding
        offset_x = cell_w * padding
        offset_y = cell_h * padding

        x = rect.x1 + col * cell_w + offset_x
        y = rect.y1 + row * cell_h + offset_y

        return Point(x, y)

    def pixel_to_grid(
        self,
        x: float,
        y: float,
    ) -> Tuple[int, int]:
        """Convert pixel coordinates to grid position.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate

        Returns:
            (row, col) grid indices
        """
        rect = self._get_puzzle_rect()
        rows, cols = self.grid_size

        if x < rect.x1 or x > rect.x2 or y < rect.y1 or y > rect.y2:
            return (-1, -1)  # Outside puzzle area

        col = int((x - rect.x1) / (rect.width / cols))
        row = int((y - rect.y1) / (rect.height / rows))

        # Clamp to valid range
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))

        return (row, col)

    def drag(
        self,
        from_pos: Union[Point, Tuple[int, int]],
        to_pos: Union[Point, Tuple[int, int]],
        duration: float = 0.3,
    ) -> None:
        """Drag from one position to another.

        Args:
            from_pos: Start position (Point or (row, col) for grid coordinates)
            to_pos: End position (Point or (row, col) for grid coordinates)
            duration: Drag duration in seconds
        """
        # Convert grid coordinates to pixel if needed
        if isinstance(from_pos, tuple):
            from_pos = self.grid_to_pixel(from_pos[0], from_pos[1])
        if isinstance(to_pos, tuple):
            to_pos = self.grid_to_pixel(to_pos[0], to_pos[1])

        session = self.connector.session

        # Perform drag gesture using swipe
        session.swipe(
            from_pos.x,
            from_pos.y,
            to_pos.x,
            to_pos.y,
            duration,
        )

    def swap_pieces(
        self,
        from_pos: Union[Point, Tuple[int, int]],
        to_pos: Union[Point, Tuple[int, int]],
        duration: float = 0.3,
    ) -> None:
        """Swap two puzzle pieces by dragging.

        This performs a drag operation which in the puzzle game
        means swapping the pieces at the two positions.

        Args:
            from_pos: Source position (grid tuple or Point)
            to_pos: Target position (grid tuple or Point)
            duration: Drag duration in seconds
        """
        self.drag(from_pos, to_pos, duration)

    def tap(
        self,
        pos: Union[Point, Tuple[int, int]],
    ) -> None:
        """Tap at a position.

        Args:
            pos: Position (grid tuple or Point)
        """
        if isinstance(pos, tuple):
            pos = self.grid_to_pixel(pos[0], pos[1])

        session = self.connector.session
        session.tap(pos.x, pos.y)

    def tap_at_pixel(self, x: float, y: float) -> None:
        """Tap at pixel coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.connector.session.tap(x, y)
