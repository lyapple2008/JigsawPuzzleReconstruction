import numpy as np


def dissimilarity_measure(first_piece, second_piece, orientation="LR", border_width=1):
    """Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces
    in the original image tend to share similar colors along their abutting
    edges, i.e., the sum (over all neighboring pixels) of squared color
    differences (over all three color bands) should be minimal. Let pieces pi ,
    pj be represented in normalized L*a*b* space by corresponding W x W x 3
    matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params orientation:  How input pieces are oriented.

                          LR => 'Left - Right'
                          TD => 'Top - Down'
    :params border_width: Number of pixel rows/cols from edge to use (default: 1).
                         If > 1, uses N pixels inward from the edge.

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")
        >>> dissimilarity_measure(p1, p2, orientation="LR", border_width=3)

    """
    rows, columns, _ = first_piece.shape()

    # Clamp border_width to valid range
    border_width = max(1, min(border_width, min(rows, columns)))

    color_difference = None

    # | L | - | R |
    if orientation == "LR":
        # Use right edge of first_piece and left edge of second_piece
        # border_width=1: use columns-1 and 0 (outermost)
        # border_width>1: use columns-border_width and 0:border_width (inner)
        left_edge = first_piece[:, columns - border_width:, :]
        right_edge = second_piece[:, :border_width, :]
        color_difference = left_edge - right_edge

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        # Use bottom edge of first_piece and top edge of second_piece
        bottom_edge = first_piece[rows - border_width:, :, :]
        top_edge = second_piece[:border_width, :, :]
        color_difference = bottom_edge - top_edge

    squared_color_difference = np.power(color_difference / 255.0, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=(1, 2))
    total_difference = np.sum(color_difference_per_row, axis=0)

    value = np.sqrt(total_difference)

    return value
