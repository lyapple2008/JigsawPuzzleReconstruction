"""Interactive puzzle piece editor with drag-and-drop swapping."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

import numpy as np


class PuzzlePieceEditor:
    """Interactive puzzle editor with drag-and-drop piece swapping."""

    def __init__(
        self,
        parent: Optional[tk.Tk] = None,
        patches: List[np.ndarray] = None,
        grid: np.ndarray = None,
        grid_size: Tuple[int, int] = (8, 8),
        piece_size: Tuple[int, int] = (50, 50),
        title: str = "Puzzle Piece Editor",
    ):
        """Initialize the puzzle piece editor.

        Args:
            parent: Parent tkinter window (creates new if None)
            patches: List of puzzle piece images (numpy arrays)
            grid: 2D array of piece indices representing current arrangement
            grid_size: Grid dimensions (rows, cols)
            piece_size: Size of each piece cell in pixels (width, height)
            title: Window title
        """
        self.patches = patches or []
        self.grid = grid.copy() if grid is not None else None
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.piece_width, self.piece_height = piece_size

        # Drag state
        self.drag_piece_pos: Optional[Tuple[int, int]] = None
        self.drag_item_id: Optional[int] = None
        self.drag_start_x: float = 0
        self.drag_start_y: float = 0

        # Swap highlighting
        self.first_swap_pos: Optional[Tuple[int, int]] = None
        self.highlight_ids: List[int] = []

        # Callback for when grid is modified
        self.on_grid_changed: Optional[Callable[[np.ndarray], None]] = None

        # Create or use parent window
        if parent is None:
            self.root = tk.Tk()
            self.own_window = True
        else:
            self.root = parent
            self.own_window = False

        self.root.title(title)

        # Create canvas
        canvas_width = self.cols * self.piece_width + 20
        canvas_height = self.rows * self.piece_height + 20
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_width,
            height=canvas_height,
            bg="gray",
            cursor="hand1",
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Create piece images on canvas
        self.piece_ids: List[List[int]] = []  # 2D list of canvas item IDs
        self._create_pieces()

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Info label
        self.info_label = tk.Label(
            self.root,
            text="Click two pieces to swap them\nor drag a piece onto another",
            justify=tk.LEFT,
        )
        self.info_label.pack(side=tk.TOP, padx=10)

        # Buttons frame
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, pady=10)

        # Buttons
        self.reset_btn = ttk.Button(
            self.button_frame,
            text="Reset to Solved",
            command=self._reset_grid,
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.confirm_btn = ttk.Button(
            self.button_frame,
            text="Confirm & Continue",
            command=self._confirm,
        )
        self.confirm_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = ttk.Button(
            self.button_frame,
            text="Cancel",
            command=self._cancel,
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        # Photo images cache for pieces
        self.photo_images: List[tk.PhotoImage] = []

    def _create_pieces(self) -> None:
        """Create puzzle pieces on the canvas."""
        self.piece_ids = []
        self.photo_images = []  # Keep references to avoid garbage collection

        offset_x = 10
        offset_y = 10

        for row in range(self.rows):
            row_ids = []
            for col in range(self.cols):
                piece_idx = self.grid[row][col]
                patch = self.patches[piece_idx]

                # Resize patch to piece size
                from PIL import Image, ImageTk

                pil_image = Image.fromarray(patch)
                pil_image = pil_image.resize(
                    (self.piece_width, self.piece_height),
                    Image.Resampling.LANCZOS,
                )

                photo = ImageTk.PhotoImage(pil_image)
                self.photo_images.append(photo)

                x1 = offset_x + col * self.piece_width
                y1 = offset_y + row * self.piece_height
                x2 = x1 + self.piece_width
                y2 = y1 + self.piece_height

                item_id = self.canvas.create_image(
                    x1, y1, image=photo, anchor=tk.NW
                )
                self.canvas.coords(item_id, x1, y1)

                # Store row, col info with the item
                self.canvas.itemconfig(item_id, tags=f"piece_{row}_{col}")

                row_ids.append(item_id)
            self.piece_ids.append(row_ids)

        # Draw grid lines
        for row in range(self.rows + 1):
            y = offset_y + row * self.piece_height
            self.canvas.create_line(
                offset_x, y,
                offset_x + self.cols * self.piece_width, y,
                fill="white", width=2
            )
        for col in range(self.cols + 1):
            x = offset_x + col * self.piece_width
            self.canvas.create_line(
                x, offset_y,
                x, offset_y + self.rows * self.piece_height,
                fill="white", width=2
            )

    def _get_piece_pos(self, item_id: int) -> Optional[Tuple[int, int]]:
        """Get the (row, col) position of a piece by its canvas item ID."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.piece_ids[row][col] == item_id:
                    return (row, col)
        return None

    def _swap_pieces(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> None:
        """Swap two pieces in the grid and update display."""
        if pos1 == pos2:
            return

        r1, c1 = pos1
        r2, c2 = pos2

        # Swap in grid
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]

        # Swap piece images on canvas
        self._update_piece_image(r1, c1)
        self._update_piece_image(r2, c2)

        # Clear highlight
        self._clear_highlight()

        # Update info
        self.info_label.config(
            text=f"Swapped ({r1},{c1}) with ({r2},{c2})\n"
            f"Click two pieces to swap them\nor drag a piece onto another"
        )

    def _update_piece_image(self, row: int, col: int) -> None:
        """Update the image for a specific grid position."""
        piece_idx = self.grid[row][col]
        patch = self.patches[piece_idx]

        from PIL import Image, ImageTk

        pil_image = Image.fromarray(patch)
        pil_image = pil_image.resize(
            (self.piece_width, self.piece_height),
            Image.Resampling.LANCZOS,
        )

        photo = ImageTk.PhotoImage(pil_image)
        self.photo_images.append(photo)

        item_id = self.piece_ids[row][col]
        self.canvas.itemconfig(item_id, image=photo)
        self.photo_images.append(photo)

    def _highlight_piece(self, row: int, col: int) -> None:
        """Highlight a piece with a colored border."""
        x1 = 10 + col * self.piece_width
        y1 = 10 + row * self.piece_height
        x2 = x1 + self.piece_width
        y2 = y1 + self.piece_height

        highlight_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="yellow", width=3, dash=(5, 3)
        )
        self.highlight_ids.append(highlight_id)
        self.canvas.tag_raise(self.piece_ids[row][col])

    def _clear_highlight(self) -> None:
        """Clear all highlights."""
        for hid in self.highlight_ids:
            self.canvas.delete(hid)
        self.highlight_ids = []
        self.first_swap_pos = None

    def _on_click(self, event: tk.Event) -> None:
        """Handle mouse click."""
        item_id = self.canvas.find_withtag(tk.CURRENT)
        if not item_id:
            return

        item_id = item_id[0]
        pos = self._get_piece_pos(item_id)
        if pos is None:
            return

        # If dragging, store the start position
        self.drag_item_id = item_id
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        # If shift+click or ctrl+click, highlight for swapping
        if event.state & 0x4 or event.state & 0x1:  # Shift or Control
            if self.first_swap_pos is None:
                self.first_swap_pos = pos
                self._highlight_piece(pos[0], pos[1])
                self.info_label.config(
                    text=f"First piece: ({pos[0]},{pos[1]})\nClick another piece to swap"
                )
            else:
                self._swap_pieces(self.first_swap_pos, pos)

    def _on_drag(self, event: tk.Event) -> None:
        """Handle mouse drag - move piece with cursor."""
        if self.drag_item_id is None:
            return

        # Don't do real-time drag, just update cursor position info
        pass

    def _on_release(self, event: tk.Event) -> None:
        """Handle mouse release."""
        if self.drag_item_id is None:
            return

        item_id = self.canvas.find_withtag(tk.CURRENT)
        if item_id:
            item_id = item_id[0]
            target_pos = self._get_piece_pos(item_id)

            if target_pos and self.first_swap_pos:
                # Drag release on a different piece - swap them
                self._swap_pieces(self.first_swap_pos, target_pos)
            elif target_pos and self.drag_item_id != item_id:
                # Regular release - swap with clicked piece
                from_pos = self._get_piece_pos(self.drag_item_id)
                if from_pos:
                    self._swap_pieces(from_pos, target_pos)

        self.drag_item_id = None
        self.first_swap_pos = None

    def _reset_grid(self) -> None:
        """Reset grid to solved state (sequential order)."""
        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col] = row * self.cols + col
                self._update_piece_image(row, col)

        self._clear_highlight()
        self.info_label.config(
            text="Grid reset to solved state\nClick two pieces to swap them"
        )

    def _confirm(self) -> None:
        """Confirm changes and close."""
        if self.on_grid_changed:
            self.on_grid_changed(self.grid)

        if self.own_window:
            self.root.quit()
            self.root.destroy()

    def _cancel(self) -> None:
        """Cancel and close without changes."""
        if self.own_window:
            self.root.quit()
            self.root.destroy()

    def show(self) -> np.ndarray:
        """Show the editor and return the modified grid.

        Returns:
            The modified grid array
        """
        if self.own_window:
            self.root.mainloop()

        return self.grid


def edit_puzzle_pieces(
    patches: List[np.ndarray],
    grid: np.ndarray,
    grid_size: Tuple[int, int] = (8, 8),
    piece_size: Tuple[int, int] = (60, 60),
    title: str = "Puzzle Piece Editor - Drag to swap pieces",
) -> Tuple[np.ndarray, bool]:
    """Show interactive puzzle editor and return modified grid.

    Args:
        patches: List of puzzle piece images
        grid: Initial grid arrangement
        grid_size: Grid dimensions
        piece_size: Size of each piece cell
        title: Window title

    Returns:
        (modified_grid, confirmed): Modified grid and whether user confirmed
    """
    result = {"grid": grid.copy(), "confirmed": False}

    def on_confirm(modified_grid: np.ndarray):
        result["grid"] = modified_grid
        result["confirmed"] = True

    editor = PuzzlePieceEditor(
        patches=patches,
        grid=grid,
        grid_size=grid_size,
        piece_size=piece_size,
        title=title,
    )
    editor.on_grid_changed = on_confirm

    editor.show()

    return result["grid"], result["confirmed"]


if __name__ == "__main__":
    # Test with sample data
    import cv2
    import numpy as np
    from pathlib import Path

    # Create dummy patches
    patches = []
    for i in range(16):
        patch = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        patches.append(patch)

    grid = np.array([[i * 4 + j for j in range(4)] for i in range(4)])

    # Shuffle for demo
    grid[0, 0], grid[0, 1] = grid[0, 1], grid[0, 0]

    modified_grid, confirmed = edit_puzzle_pieces(
        patches=patches,
        grid=grid,
        grid_size=(4, 4),
        piece_size=(80, 80),
        title="Test Puzzle Editor",
    )

    print(f"Confirmed: {confirmed}")
    print(f"Modified grid:\n{modified_grid}")
