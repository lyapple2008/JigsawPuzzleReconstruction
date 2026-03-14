import click
import cv2 as cv
import numpy as np

from gaps import utils
from gaps.genetic_algorithm import GeneticAlgorithm
from gaps.size_detector import SizeDetector

DEFAULT_GENERATIONS: int = 20
DEFAULT_POPULATION: int = 200

MIN_PIECE_SIZE: int = 32
MAX_PIECE_SIZE: int = 128


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "ignore_unknown_options": True,
    }
)
def cli() -> None:
    """Solve or create puzzles with square pieces."""


def _parse_piece_size(value):
    """Parse piece_size from string format like '32' or '64x32'.

    Returns (width, height) tuple.
    """
    if value is None:
        return None

    if isinstance(value, tuple):
        return value

    # Handle string format
    if isinstance(value, str):
        if 'x' in value.lower():
            parts = value.lower().split('x')
            if len(parts) != 2:
                raise ValueError("Invalid piece size format. Use 'WxH' (e.g., '64x32')")
            return (int(parts[0]), int(parts[1]))
        else:
            # Single value means square piece
            return (int(value), int(value))

    # Integer means square piece
    return (value, value)


def _validate_piece_size(_context: click.Context, _param: str, value):
    """Validate piece_size and return (width, height) tuple."""
    if value is None:
        return value

    parsed = _parse_piece_size(value)

    # Validate each dimension
    for dim in parsed:
        if dim < MIN_PIECE_SIZE:
            raise click.BadParameter(f"Minimum piece size is {MIN_PIECE_SIZE} pixels")
        if dim > MAX_PIECE_SIZE:
            raise click.BadParameter(f"Maximum piece size is {MAX_PIECE_SIZE} pixels")

    return parsed


def _validate_positive_integer(_context: click.Context, _param: str, value: int) -> int:
    if value <= 0:
        raise click.BadParameter("Should be a positive integer.")

    return value


@click.command()
@click.argument("puzzle", type=click.Path(exists=True, readable=True))
@click.argument("solution", type=click.Path(dir_okay=False, writable=True))
@click.option(
    "-s",
    "--size",
    type=str,
    help="Size of puzzle piece(s). Can be integer (square), 'WxH' (rectangular), or autodetected.",
)
@click.option(
    "-g",
    "--generations",
    type=int,
    show_default=True,
    default=DEFAULT_GENERATIONS,
    callback=_validate_positive_integer,
    help="The number of generations for genetic algorithm.",
)
@click.option(
    "-p",
    "--population",
    type=int,
    show_default=True,
    default=DEFAULT_POPULATION,
    callback=_validate_positive_integer,
    help="The size of the initial population for genetic algorithm.",
)
@click.option(
    "-d",
    "--debug",
    type=bool,
    is_flag=True,
    default=False,
    help="If enabled, shows the best individual after each generation.",
)
def run(
    puzzle: str,
    solution: str,
    size,
    generations: int,
    population: int,
    debug: bool,
) -> None:
    """Run puzzle solver.

    \b
    PUZZLE is the input puzzle image with puzzle pieces.
    SOLUTION is the output image file for solved puzzle.

    Examples:

    $ gaps run puzzle.jpg solution.jpg --size=32 --generations=100 --population=1000
    $ gaps run puzzle.jpg solution.jpg --size=64x32

    """

    input_puzzle = cv.imread(puzzle)

    if size is None:
        detector = SizeDetector(input_puzzle)
        detected = detector.detect()
        size = (detected, detected)

    click.echo(f"Population: {population}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Piece size: {size[0]}x{size[1]}")

    ga = GeneticAlgorithm(
        image=input_puzzle,
        piece_size=size,
        population_size=population,
        generations=generations,
    )
    result = ga.start_evolution(debug)
    output_image = result.to_image()

    cv.imwrite(solution, output_image)

    click.echo("Puzzle solved")


@click.command()
@click.argument("image", type=click.Path(exists=True, readable=True))
@click.argument("puzzle", type=click.Path(writable=True))
@click.option(
    "-s",
    "--size",
    type=str,
    show_default=True,
    default=str(MAX_PIECE_SIZE),
    callback=_validate_piece_size,
    help="Size of puzzle piece(s). Can be integer (square) or 'WxH' (rectangular).",
)
def create(image: str, puzzle: str, size) -> None:
    """Create jigsaw puzzle with puzzle pieces.

    \b
    IMAGE is the input image file to create puzzle.
    PUZZLE is the output puzzle image with puzzle pieces.

    Examples:

    $ gaps create image.jpg puzzle.jpg --size=32
    $ gaps create image.jpg puzzle.jpg --size=64x32

    """

    input_image = cv.imread(image)
    pieces, rows, columns = utils.flatten_image(input_image, size)

    # Randomize pieces in order to make puzzle
    np.random.shuffle(pieces)

    # Create puzzle by stacking pieces
    output_image = utils.assemble_image(pieces, rows, columns)

    cv.imwrite(puzzle, output_image)

    click.echo(f"\nCreated puzzle with {len(pieces)} pieces")


cli.add_command(run, name="run")
cli.add_command(create, name="create")

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
