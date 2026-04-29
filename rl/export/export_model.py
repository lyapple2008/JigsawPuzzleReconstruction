"""Export trained puzzle RL model to ONNX format.

Usage:
    conda activate jigsaw
    python3 -m rl.export.export_model --model rl_models/puzzle_ppo_final --grid-size 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def export_to_onnx(
    model_path: str,
    grid_size: int = 6,
    output_path: str = None,
) -> str:
    """Export SB3 PPO model to ONNX format.

    Args:
        model_path: Path to saved SB3 model.
        grid_size: Puzzle grid size (to build dummy input).
        output_path: Output ONNX file path. Defaults to model_path.onnx.

    Returns:
        Path to exported ONNX file.
    """
    import torch
    from stable_baselines3 import PPO

    from rl.envs.puzzle_env import PuzzleSwapEnv

    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    if output_path is None:
        output_path = model_path + ".onnx"

    # Get policy
    policy = model.policy
    policy.eval()

    # Build dummy input matching observation space
    n = grid_size * grid_size
    dummy_grid = np.zeros((1, n), dtype=np.int32)
    dummy_costs = np.zeros((1, n, n, 2), dtype=np.float32)

    # SB3 MultiInputPolicy expects dict, but for ONNX we trace the forward pass
    # We need to extract the underlying network
    obs_tensor = {
        "grid": torch.tensor(dummy_grid, dtype=torch.long),
        "edge_costs": torch.tensor(dummy_costs, dtype=torch.float32),
    }

    # Export using torch.onnx
    # Note: SB3's policy forward is complex; we export the feature extractor + action head
    try:
        # Get the action distribution (logits)
        features = policy.features_extractor(obs_tensor)
        latent_pi = policy.mlp_extractor.policy_net(features)
        latent_vf = policy.mlp_extractor.value_net(features)

        # Create wrapper for export
        class PolicyWrapper(torch.nn.Module):
            def __init__(self, pi_net, action_net):
                super().__init__()
                self.pi_net = pi_net
                self.action_net = action_net

            def forward(self, grid, edge_costs):
                obs = {"grid": grid, "edge_costs": edge_costs}
                feat = policy.features_extractor(obs)
                latent = self.pi_net(feat)
                return self.action_net(latent)

        wrapper = PolicyWrapper(policy.mlp_extractor.policy_net, policy.action_net)
        wrapper.eval()

        torch.onnx.export(
            wrapper,
            (obs_tensor["grid"], obs_tensor["edge_costs"]),
            output_path,
            input_names=["grid", "edge_costs"],
            output_names=["action_logits"],
            dynamic_axes={
                "grid": {0: "batch"},
                "edge_costs": {0: "batch"},
                "action_logits": {0: "batch"},
            },
            opset_version=17,
        )

        print(f"Exported ONNX model to {output_path}")

        # Verify
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(output_path)
            inputs = {
                "grid": dummy_grid,
                "edge_costs": dummy_costs,
            }
            outputs = session.run(None, inputs)
            print(f"ONNX verification OK. Output shape: {outputs[0].shape}")
        except Exception as e:
            print(f"ONNX verification warning: {e}")

        return output_path

    except Exception as e:
        print(f"Export failed: {e}")
        print("Falling back to torch.jit.trace export...")

        # Fallback: save as TorchScript
        jit_path = output_path.replace(".onnx", ".pt")
        traced = torch.jit.trace(
            wrapper,
            (obs_tensor["grid"], obs_tensor["edge_costs"]),
        )
        traced.save(jit_path)
        print(f"Exported TorchScript model to {jit_path}")
        return jit_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export puzzle RL model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to saved SB3 model")
    parser.add_argument("--grid-size", type=int, default=6, help="Puzzle grid size")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        grid_size=args.grid_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
