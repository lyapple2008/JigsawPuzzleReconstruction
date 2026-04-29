"""Run trained RL model for puzzle action prediction.

Supports both SB3 model (direct loading) and ONNX inference.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


class MobileModelRunner:
    """Run trained puzzle RL model to predict swap actions.

    Supports:
    - SB3 PPO model (requires torch + stable_baselines3)
    - ONNX model (requires onnxruntime)
    """

    def __init__(
        self,
        model_path: str,
        grid_size: int = 6,
        model_type: str = "auto",
    ) -> None:
        """Initialize model runner.

        Args:
            model_path: Path to saved model file.
            grid_size: Puzzle grid size (N for NxN puzzle).
            model_type: "sb3", "onnx", or "auto" (detect from file extension).
        """
        self.model_path = model_path
        self.grid_size = grid_size
        self.n = grid_size * grid_size
        self.model_type = model_type
        self._model = None
        self._session = None

        if model_type == "auto":
            if model_path.endswith(".onnx"):
                model_type = "onnx"
            elif model_path.endswith(".zip"):
                model_type = "sb3"
            else:
                model_type = "sb3"

        if model_type == "sb3":
            self._load_sb3(model_path)
        elif model_type == "onnx":
            self._load_onnx(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_sb3(self, path: str) -> None:
        """Load Stable Baselines3 PPO model."""
        from stable_baselines3 import PPO

        self._model = PPO.load(path)
        self._model_type = "sb3"

    def _load_onnx(self, path: str) -> None:
        """Load ONNX model."""
        import onnxruntime as ort

        self._session = ort.InferenceSession(path)
        self._model_type = "onnx"

    def predict(self, observation: dict) -> int:
        """Predict swap action from observation.

        Args:
            observation: Dict with "grid" (int32 array) and "edge_costs" (float32 array).

        Returns:
            Integer action index (into swap pair list).
        """
        if self._model_type == "sb3":
            return self._predict_sb3(observation)
        elif self._model_type == "onnx":
            return self._predict_onnx(observation)
        else:
            raise RuntimeError(f"Unknown model type: {self._model_type}")

    def _predict_sb3(self, observation: dict) -> int:
        """Predict using SB3 model."""
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def _predict_onnx(self, observation: dict) -> int:
        """Predict using ONNX model."""
        inputs = {
            "grid": observation["grid"].reshape(1, -1),
            "edge_costs": observation["edge_costs"].reshape(1, self.n, self.n, 2),
        }
        outputs = self._session.run(None, inputs)
        # outputs[0] is action logits, take argmax
        logits = outputs[0]
        return int(np.argmax(logits, axis=-1)[0])

    def decode_action(self, action: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Decode action index to grid position pairs.

        Args:
            action: Integer action index.

        Returns:
            ((row1, col1), (row2, col2))
        """
        # Reconstruct swap pairs (same logic as PuzzleSwapEnv)
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j))
                if len(pairs) > action:
                    pos_a, pos_b = pairs[-1]
                    r1, c1 = divmod(pos_a, self.cols if hasattr(self, 'cols') else self.grid_size)
                    r2, c2 = divmod(pos_b, self.cols if hasattr(self, 'cols') else self.grid_size)
                    return (r1, c1), (r2, c2)
        raise ValueError(f"Invalid action index: {action}")

    @property
    def cols(self) -> int:
        return self.grid_size
