"""Use case: train one Unity agent. Inputs: YAML settings and bridge states. Outputs: checkpoints and a final model."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from unity_env import UnityGymEnv


ALGORITHMS = {"ppo": PPO, "a2c": A2C}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping.")

    model_name = config.get("model", "ppo")
    if model_name not in ALGORITHMS:
        raise ValueError("model must be ppo or a2c.")

    positive_ints = ("port", "total_timesteps", "n_steps")
    for name in positive_ints:
        value = config.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if config["port"] > 65535:
        raise ValueError("port must be at most 65535.")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] <= 2_147_483_647:
        raise ValueError("seed must be a nonnegative 32-bit integer.")
    if type(config.get("checkpoint_freq")) is not int or config["checkpoint_freq"] < 0:
        raise ValueError("checkpoint_freq must be a nonnegative integer.")
    if (type(config.get("learning_rate")) not in (int, float) or
            not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("learning_rate must be positive.")
    if (type(config.get("gamma")) not in (int, float) or
            not math.isfinite(config["gamma"]) or not 0 < config["gamma"] <= 1):
        raise ValueError("gamma must be between 0 and 1.")
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"]:
        raise ValueError("output_dir must be a path.")
    if not isinstance(config.get("device"), str) or not config["device"]:
        raise ValueError("device must be a nonempty string.")
    timeout = config.get("socket_timeout_seconds", 300)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("socket_timeout_seconds must be positive.")
    if config.get("resume_from") is not None and not isinstance(config["resume_from"], str):
        raise ValueError("resume_from must be a path or null.")
    if model_name == "ppo":
        ppo = config.get("ppo")
        batch_size = ppo.get("batch_size") if isinstance(ppo, dict) else None
        if type(batch_size) is not int or not 1 < batch_size <= config["n_steps"]:
            raise ValueError("ppo.batch_size must be between 2 and n_steps.")

    return config


def train(config_path: Path) -> Path:
    config = load_config(config_path)
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = config_path.parent / output_root
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

    environment = UnityGymEnv(
        port=config["port"],
        seed=config["seed"],
        timeout_seconds=config.get("socket_timeout_seconds", 300),
    )
    started = time.monotonic()
    vector_environment = None
    try:
        vector_environment = DummyVecEnv([lambda: Monitor(environment)])
        run_dir.mkdir(parents=True, exist_ok=False)
        algorithm = ALGORITHMS[config.get("model", "ppo")]
        resume_path = config.get("resume_from")
        if resume_path:
            resume_path = Path(resume_path)
            if not resume_path.is_absolute():
                resume_path = config_path.parent / resume_path
            model = algorithm.load(str(resume_path), env=vector_environment, device=config["device"])
        else:
            parameters = {
                "learning_rate": config["learning_rate"],
                "gamma": config["gamma"],
                "n_steps": config["n_steps"],
                "device": config["device"],
                "seed": config["seed"],
                "verbose": 1,
            }
            if config.get("model", "ppo") == "ppo":
                parameters["batch_size"] = config["ppo"]["batch_size"]
            model = algorithm("MlpPolicy", vector_environment, **parameters)

        callback = None
        if config["checkpoint_freq"]:
            callback = CheckpointCallback(
                save_freq=config["checkpoint_freq"],
                save_path=str(run_dir),
                name_prefix="checkpoint",
            )
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            reset_num_timesteps=not bool(resume_path),
        )
        final_path = run_dir / "final_model.zip"
        model.save(str(final_path))
        print(f"Training complete: {final_path} ({time.monotonic() - started:.1f}s)")
        return final_path
    finally:
        if vector_environment is not None:
            vector_environment.close()
        else:
            environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one Unity RL environment.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        train(args.config.resolve())
        return 0
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
