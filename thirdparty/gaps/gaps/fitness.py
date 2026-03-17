import numpy as np


def _compute_robust_difference(color_difference: np.ndarray, method: str = "mse", percentile: float = 25.0) -> float:
    """Compute robust dissimilarity from color difference array.

    Args:
        color_difference: Array of shape (H, W, C) containing color differences.
        method: Method to compute dissimilarity. Options:
            - "mse": Mean Squared Error (original)
            - "median": Median of squared differences (robust to outliers)
            - "percentile": Percentile-based (ignores extreme values)
            - "huber": Huber loss (robust M-estimator)
        percentile: Percentile to use when method="percentile" (default: 25.0)

    Returns:
        Dissimilarity value (lower is better)
    """
    # Normalize to [0, 1] range
    normalized = (color_difference / 255.0) ** 2

    if method == "mse":
        # Original: MSE
        return float(np.mean(normalized))

    elif method == "median":
        # Median: robust to outliers
        return float(np.median(normalized))

    elif method == "percentile":
        # Percentile: ignore extreme values
        # Use percentile-based range to ignore outliers
        p_low = percentile
        p_high = 100.0 - percentile
        return float(np.percentile(normalized, [p_low, p_high]).mean())

    elif method == "huber":
        # Huber loss: quadratic for small errors, linear for large errors
        # This balances MSE robustness with outlier handling
        delta = 0.5  # Threshold for switching from quadratic to linear
        flat = normalized.flatten()
        abs_diff = np.sqrt(flat)
        quadratic = np.where(abs_diff <= delta, 0.5 * flat ** 2, delta * abs_diff - 0.5 * delta ** 2)
        return float(np.mean(quadratic))

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: mse, median, percentile, huber")


def dissimilarity_measure(
    first_piece,
    second_piece,
    orientation="LR",
    border_width=1,
    robust_method: str = "mse",
    percentile: float = 25.0,
):
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
    :params robust_method: Method for robust dissimilarity calculation.
                          Options: "mse" (original), "median", "percentile", "huber"
    :params percentile: Percentile to use when robust_method="percentile" (default: 25.0)

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")
        >>> dissimilarity_measure(p1, p2, orientation="LR", border_width=3)
        >>> dissimilarity_measure(p1, p2, robust_method="median")
        >>> dissimilarity_measure(p1, p2, robust_method="percentile", percentile=20.0)

    """
    rows, columns, _ = first_piece.shape()

    # Clamp border_width to valid range
    border_width = max(1, min(border_width, min(rows, columns)))

    color_difference = None

    # | L | - | R |
    if orientation == "LR":
        # Compare right edge of first_piece (left in arrangement) with left edge of second_piece (right in arrangement)
        # border_width=1: use columns-1 and 0 (outermost)
        # border_width>1: use columns-border_width and 0:border_width (inner)
        right_edge_of_first = first_piece[:, columns - border_width:, :]
        left_edge_of_second = second_piece[:, :border_width, :]
        color_difference = right_edge_of_first - left_edge_of_second

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        # Compare bottom edge of first_piece (top in arrangement) with top edge of second_piece (bottom in arrangement)
        bottom_edge_of_first = first_piece[rows - border_width:, :, :]
        top_edge_of_second = second_piece[:border_width, :, :]
        color_difference = bottom_edge_of_first - top_edge_of_second

    # Compute dissimilarity using robust method
    diff_value = _compute_robust_difference(color_difference, method=robust_method, percentile=percentile)

    # Return sqrt to maintain scale similar to original
    value = np.sqrt(diff_value)

    return value
