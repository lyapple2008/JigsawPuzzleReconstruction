import numpy as np

from gaps.piece import Piece


def flatten_image(image, piece_size, indexed=False):
    """Converts image into list of pieces (square or rectangular).

    Input image is divided into pieces of specified size and then
    flattened into list. Each list element is PIECE_HEIGHT x PIECE_WIDTH x 3

    :params image:      Input image.
    :params piece_size: Size of single piece. Can be:
                       - int: square piece (piece_size x piece_size)
                       - tuple (height, width): rectangular piece
    :params indexed: If True list of Pieces with IDs will be returned,
        otherwise list of ndarray pieces

    Usage::

        >>> from gaps.image_helpers import flatten_image
        >>> flat_image = flatten_image(image, 32)  # square pieces
        >>> flat_image = flatten_image(image, (32, 48))  # rectangular pieces

    """
    # Support both square (int) and rectangular (tuple) piece sizes
    if isinstance(piece_size, tuple):
        piece_h, piece_w = piece_size
    else:
        piece_h = piece_w = piece_size

    rows, columns = image.shape[0] // piece_h, image.shape[1] // piece_w
    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = (
                x * piece_w,
                y * piece_h,
                (x + 1) * piece_w,
                (y + 1) * piece_h,
            )
            piece = np.empty((piece_h, piece_w, image.shape[2]))
            piece[:piece_h, :piece_w, :] = image[top:h, left:w, :]
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
