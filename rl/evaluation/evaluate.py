"""Evaluate a trained puzzle RL model.

Usage:
    conda activate jigsaw
    python3 -m rl.evaluation.evaluate --model rl_models/puzzle_ppo_final --grid-size 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate(
    model_path: str,
    grid_size: int = 6,
    n_episodes: int = 50,
    max_steps: int = 500,
    seed: int = 42,
    image_path: str = None,
) -> dict:
    """Evaluate trained model on puzzle environment.

    Args:
        model_path: Path to saved SB3 model.
        grid_size: Puzzle grid size.
        n_episodes: Number of evaluation episodes.
        max_steps: Max steps per episode.
        seed: Random seed.
        image_path: Optional puzzle image path.

    Returns:
        Dict with evaluation metrics.
    """
    from stable_baselines3 import PPO

    from jigsaw.utils import load_or_generate_image
    from rl.envs.puzzle_env import PuzzleSwapEnv

    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create env
    image_size = grid_size * 100
    image = load_or_generate_image(image_path, size=image_size, seed=seed)
    env = PuzzleSwapEnv(
        image=image,
        rows=grid_size,
        cols=grid_size,
        max_steps=max_steps,
        seed=seed,
    )

    # Evaluate
    accuracies = []
    costs = []
    steps_list = []
    solved_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1

        acc = info.get("position_accuracy", 0.0)
        cost = info.get("total_cost", 0.0)
        accuracies.append(acc)
        costs.append(cost)
        steps_list.append(steps)
        if acc >= 1.0:
            solved_count += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: acc={acc:.3f}, steps={steps}")

    # Summary
    results = {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_cost": float(np.mean(costs)),
        "mean_steps": float(np.mean(steps_list)),
        "solve_rate": solved_count / n_episodes,
        "n_episodes": n_episodes,
        "grid_size": grid_size,
    }

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({grid_size}x{grid_size}, {n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean Accuracy:  {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"  Mean Cost:      {results['mean_cost']:.1f}")
    print(f"  Mean Steps:     {results['mean_steps']:.0f}")
    print(f"  Solve Rate:     {results['solve_rate']:.2%}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained puzzle RL model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--grid-size", type=int, default=6, help="Puzzle grid size")
    parser.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image", type=str, default=None, help="Puzzle image path")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        grid_size=args.grid_size,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
