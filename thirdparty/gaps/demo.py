#!/usr/bin/env python
"""
Demo script for GAPS image puzzle solver.
This script demonstrates using genetic algorithm to solve image puzzles.
"""
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from gaps import utils
from gaps.genetic_algorithm import GeneticAlgorithm


def create_puzzle(image, piece_size):
    """Create a shuffled puzzle from the image."""
    pieces, rows, columns = utils.flatten_image(image, piece_size)
    # Shuffle pieces to create puzzle
    shuffled_pieces = pieces.copy()
    np.random.shuffle(shuffled_pieces)
    return shuffled_pieces, rows, columns


def solve_puzzle(puzzle_pieces, rows, columns, piece_size, population=200, generations=50):
    """Solve the puzzle using genetic algorithm."""
    # Reconstruct the shuffled image for the algorithm
    shuffled_image = utils.assemble_image(puzzle_pieces, rows, columns)

    # Run genetic algorithm
    ga = GeneticAlgorithm(
        image=shuffled_image,
        piece_size=piece_size,
        population_size=population,
        generations=generations,
    )
    result = ga.start_evolution(verbose=False)
    return result.to_image()


def main():
    parser = argparse.ArgumentParser(description="GAPS puzzle solver demo")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-r", "--rows", type=int, required=True, help="Number of rows (pieces)")
    parser.add_argument("-c", "--cols", type=int, required=True, help="Number of columns (pieces)")
    parser.add_argument("-p", "--population", type=int, default=200, help="Population size")
    parser.add_argument("-g", "--generations", type=int, default=30, help="Number of generations")

    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)

    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    # Calculate piece size based on rows and columns
    rows, cols = args.rows, args.cols

    # Calculate piece dimensions (may not be square)
    piece_width = image.shape[1] // cols
    piece_height = image.shape[0] // rows
    piece_size = (piece_width, piece_height)
    print(f"Piece size: {piece_size[0]}x{piece_size[1]} (rows: {rows}, cols: {cols})")

    # Create puzzle
    print(f"Creating puzzle...")
    puzzle_pieces, puzzle_rows, puzzle_cols = create_puzzle(image, piece_size)
    print(f"Actual pieces: rows={puzzle_rows}, cols={puzzle_cols}, total={puzzle_rows * puzzle_cols}")

    # Assemble shuffled puzzle image
    puzzle_image = utils.assemble_image(puzzle_pieces, puzzle_rows, puzzle_cols)
    puzzle_image_rgb = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2RGB)

    # Solve puzzle
    print(f"Solving puzzle with {puzzle_rows * puzzle_cols} pieces...")
    print(f"Population: {args.population}, Generations: {args.generations}")
    solved_image = solve_puzzle(
        puzzle_pieces, puzzle_rows, puzzle_cols, piece_size,
        population=args.population, generations=args.generations
    )
    solved_image_rgb = cv2.cvtColor(solved_image, cv2.COLOR_BGR2RGB)

    # Convert original image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display results
    print("Displaying results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(puzzle_image_rgb)
    axes[1].set_title("Shuffled Puzzle")
    axes[1].axis("off")

    axes[2].imshow(solved_image_rgb)
    axes[2].set_title("Solved Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
