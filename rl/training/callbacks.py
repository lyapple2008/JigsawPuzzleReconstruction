"""Custom callbacks for puzzle RL training."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PuzzleEvalCallback(BaseCallback):
    """Logs puzzle-specific metrics during training.

    Tracks: mean accuracy, mean steps to solve, solve rate.
    """

    def __init__(
        self,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Run evaluation episodes
        env = self.model.get_env()
        accuracies = []
        costs = []
        steps_list = []
        solved_count = 0

        for _ in range(self.n_eval_episodes):
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                steps += 1
                if done:
                    break

            info_dict = info[0] if isinstance(info, list) else info
            acc = info_dict.get("position_accuracy", 0.0)
            cost = info_dict.get("total_cost", 0.0)
            accuracies.append(acc)
            costs.append(cost)
            steps_list.append(steps)
            if acc >= 1.0:
                solved_count += 1

        mean_acc = np.mean(accuracies)
        mean_cost = np.mean(costs)
        mean_steps = np.mean(steps_list)
        solve_rate = solved_count / self.n_eval_episodes

        if self.verbose > 0:
            print(
                f"\n[PuzzleEval] Step {self.n_calls}: "
                f"accuracy={mean_acc:.3f}, cost={mean_cost:.1f}, "
                f"steps={mean_steps:.0f}, solve_rate={solve_rate:.2f}"
            )

        # Log to tensorboard
        self.logger.record("eval/mean_accuracy", mean_acc)
        self.logger.record("eval/mean_cost", mean_cost)
        self.logger.record("eval/mean_steps", mean_steps)
        self.logger.record("eval/solve_rate", solve_rate)

        return True


class CheckpointCallback(BaseCallback):
    """Save model checkpoints at regular intervals."""

    def __init__(
        self,
        save_freq: int = 50_000,
        save_path: str = "rl_models",
        name_prefix: str = "puzzle",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"\n[Checkpoint] Saved model to {path}")
        return True
