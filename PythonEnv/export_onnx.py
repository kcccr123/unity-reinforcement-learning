"""Export a PPO MlpPolicy checkpoint for RlModelRunner."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO


class DeterministicPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        actions, _, _ = self.policy(obs, deterministic=True)
        # Match SB3's Box-action clipping before Unity decodes the action.
        return torch.clamp(actions, -1.0, 1.0)


def export_onnx(checkpoint: Path, output: Path) -> None:
    model = PPO.load(checkpoint, device="cpu")
    policy = DeterministicPolicy(model.policy).eval()
    observation = torch.zeros((1, *model.observation_space.shape), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            policy,
            observation,
            str(output),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=15,
            dynamo=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)
    print(f"Exported {args.output}")
