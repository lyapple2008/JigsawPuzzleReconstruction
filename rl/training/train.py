"""PPO training script for puzzle environment.

Usage:
    conda activate jigsaw
    python3 -m rl.training.train --grid-size 6 --timesteps 1000000
    python3 -m rl.training.train --grid-size 6 --image examples/IMG_0970.PNG
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.training.config import TrainConfig


def make_env(config: TrainConfig):
    """Create puzzle environment from config."""
    from jigsaw.utils import generate_natural_like_image, load_or_generate_image
    from rl.envs.puzzle_env import PuzzleSwapEnv

    image_size = config.image_size or config.grid_size * 100
    image = load_or_generate_image(config.image_path, size=image_size, seed=config.seed)

    env = PuzzleSwapEnv(
        image=image,
        rows=config.grid_size,
        cols=config.grid_size,
        max_steps=config.max_episode_steps,
        seed=config.seed,
        cost_improvement_weight=config.cost_improvement_weight,
        accuracy_bonus_weight=config.accuracy_bonus_weight,
        solve_bonus=config.solve_bonus,
        step_penalty=config.step_penalty,
    )
    return env


def train(config: TrainConfig) -> None:
    """Run PPO training."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList

    from rl.training.callbacks import CheckpointCallback, PuzzleEvalCallback

    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Create environment
    env = make_env(config)
    print(f"Environment: {config.grid_size}x{config.grid_size} puzzle")
    print(f"  Patches: {env.n}")
    print(f"  Action space: {env.n_swaps} possible swaps")
    print(f"  Observation: grid({env.n},) + edge_costs({env.n},{env.n},2)")

    # Create model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        tensorboard_log=config.log_dir,
        seed=config.seed,
    )

    # Callbacks
    eval_cb = PuzzleEvalCallback(
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=config.model_dir,
        name_prefix=config.model_name,
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    # Train
    print(f"\nStarting training for {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(config.model_dir, f"{config.model_name}_final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on puzzle environment")
    parser.add_argument("--grid-size", type=int, default=6, help="Puzzle grid size (NxN)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--image", type=str, default=None, help="Path to puzzle image")
    parser.add_argument("--image-size", type=int, default=None, help="Resize image to this size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model-dir", type=str, default="rl_models", help="Model save directory")
    parser.add_argument("--log-dir", type=str, default="rl_logs", help="TensorBoard log directory")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    config = TrainConfig(
        grid_size=args.grid_size,
        total_timesteps=args.timesteps,
        image_path=args.image,
        image_size=args.image_size,
        seed=args.seed,
        learning_rate=args.lr,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        max_episode_steps=args.max_steps,
    )

    train(config)


if __name__ == "__main__":
    main()
