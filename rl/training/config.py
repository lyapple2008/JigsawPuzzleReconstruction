"""Training hyperparameter configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for PPO training on puzzle environment."""

    # Environment
    grid_size: int = 6
    max_episode_steps: int = 500
    seed: int = 42

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training schedule
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    checkpoint_freq: int = 50_000
    n_eval_episodes: int = 20

    # Reward shaping
    cost_improvement_weight: float = 1.0
    accuracy_bonus_weight: float = 10.0
    solve_bonus: float = 100.0
    step_penalty: float = 0.01

    # Image
    image_size: Optional[int] = None  # None = auto (grid_size * 100)
    image_path: Optional[str] = None  # None = generate synthetic

    # Output
    model_dir: str = "rl_models"
    log_dir: str = "rl_logs"
    model_name: str = "puzzle_ppo"
