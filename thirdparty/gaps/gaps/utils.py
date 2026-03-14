import numpy as np

from gaps.piece import Piece


def parse_piece_size(piece_size):
    """Parse piece_size from integer, tuple, or string format.

    Supports:
    - Integer: 32 -> (32, 32)
    - Tuple: (64, 32) -> (64, 32)
    - String: "64x32" -> (64, 32)

    Returns: (width, height) tuple
    """
    if isinstance(piece_size, tuple):
        return piece_size
    if isinstance(piece_size, str):
        if 'x' in piece_size.lower():
            parts = piece_size.lower().split('x')
            return (int(parts[0]), int(parts[1]))
        return (int(piece_size), int(piece_size))
    # Integer
    return (piece_size, piece_size)


def flatten_image(image, piece_size, indexed=False):
    """Converts image into list of rectangular pieces.

    Input image is divided into pieces of specified size and then
    flattened into list. Each list element is HEIGHT x WIDTH x 3

    :params image:      Input image.
    :params piece_size: Size of single piece as integer, (width, height) tuple,
                        or "WxH" string format.
    :params indexed: If True list of Pieces with IDs will be returned,
        otherwise list of ndarray pieces

    Usage::

        >>> from gaps.image_helpers import flatten_image
        >>> flat_image = flatten_image(image, 32)
        >>> flat_image = flatten_image(image, (64, 32))
        >>> flat_image = flatten_image(image, "64x32")

    """
    width, height = parse_piece_size(piece_size)

    rows = image.shape[0] // height
    columns = image.shape[1] // width
    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = (
                x * width,
                y * height,
                (x + 1) * width,
                (y + 1) * height,
            )
            piece = np.empty((height, width, image.shape[2]))
            piece[:height, :width, :] = image[top:h, left:w, :]
            pieces.append(piece)

    if indexed:
        pieces = [Piece(value, index) for index, value in enumerate(pieces)]

    return pieces, rows, columns


def assemble_image(pieces, rows, columns):
    """Assembles image from pieces.

    Given an array of pieces and desired image dimensions, function assembles
    image by stacking pieces.

    :params pieces:  Image pieces as an array.
    :params rows:    Number of rows in resulting image.
    :params columns: Number of columns in resulting image.

    Usage::

        >>> from gaps.image_helpers import assemble_image
        >>> from gaps.image_helpers import flatten_image
        >>> pieces, rows, cols = flatten_image(...)
        >>> original_img = assemble_image(pieces, rows, cols)

    """
    vertical_stack = []
    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            horizontal_stack.append(pieces[i * columns + j])
        vertical_stack.append(np.hstack(horizontal_stack))
    return np.vstack(vertical_stack).astype(np.uint8)
