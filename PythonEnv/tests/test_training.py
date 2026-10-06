import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import load_config, main, train
from unity_env import BridgeError, UnityGymEnv


class CounterServer:
    def __init__(self, error_on_action=False, disconnect_on_reset=False, malformed_state=False):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.error_on_action = error_on_action
        self.disconnect_on_reset = disconnect_on_reset
        self.malformed_state = malformed_state
        self.messages = []
        self.connection = None
        self.closed = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        try:
            self.connection, _ = self.listener.accept()
            with self.connection:
                self.connection.sendall(b"CONFIG:OBS=1;")
                time.sleep(0.01)
                self.connection.sendall(b"ACT=1;ENV_TYPE=SINGLE\n")
                reader = self.connection.makefile("rb")
                steps = 0
                while True:
                    line = reader.readline()
                    if not line:
                        return
                    message = line.decode("utf-8").strip()
                    self.messages.append(message)
                    if message.startswith("ACT=RESET;SEED="):
                        if self.disconnect_on_reset:
                            return
                        steps = 0
                        self.connection.sendall(b"OBS=0;REW=0;")
                        time.sleep(0.001)
                        self.connection.sendall(b"DONE=0\n")
                    elif self.error_on_action:
                        self.connection.sendall(b"ERROR=ENVIRONMENT\n")
                        return
                    elif self.malformed_state:
                        self.connection.sendall(b"OBS=not-a-number;REW=0;DONE=0\n")
                        return
                    else:
                        steps += 1
                        done = int(steps == 2)
                        reward = done
                        self.connection.sendall(
                            f"OBS={steps};REW={reward};DONE={done}\n".encode("utf-8")
                        )
        except OSError:
            if not self.closed:
                raise

    def close(self):
        self.closed = True
        if self.connection is not None:
            self.connection.close()
        self.listener.close()
        self.thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_reset_step_and_episode_end():
    with CounterServer() as server:
        environment = UnityGymEnv(port=server.port, seed=5, timeout_seconds=2)
        try:
            observation, info = environment.reset()
            assert observation.tolist() == [0.0]
            assert info == {}
            observation, reward, terminated, truncated, info = environment.step([0.25])
            assert (observation.tolist(), reward, terminated, truncated) == ([1.0], 0.0, False, False)
            observation, reward, terminated, truncated, info = environment.step([-0.5])
            assert (observation.tolist(), reward, terminated, truncated) == ([2.0], 1.0, True, False)
            with pytest.raises(RuntimeError, match="Reset"):
                environment.step([0.0])
        finally:
            environment.close()
        assert server.messages == ["ACT=RESET;SEED=5", "ACT=0.25", "ACT=-0.5"]


def test_invalid_action_stays_local():
    with CounterServer() as server:
        environment = UnityGymEnv(port=server.port, timeout_seconds=2)
        try:
            environment.reset()
            with pytest.raises(ValueError, match="between -1 and 1"):
                environment.step([2.0])
            environment.step(np.array([0.0], dtype=np.float32))
        finally:
            environment.close()
        assert server.messages == ["ACT=RESET;SEED=0", "ACT=0"]


def test_bridge_error_is_not_a_training_transition():
    with CounterServer(error_on_action=True) as server:
        environment = UnityGymEnv(port=server.port, timeout_seconds=2)
        try:
            environment.reset()
            with pytest.raises(BridgeError, match="ENVIRONMENT"):
                environment.step([0.0])
        finally:
            environment.close()


def test_disconnect_is_not_a_training_transition():
    with CounterServer(disconnect_on_reset=True) as server:
        environment = UnityGymEnv(port=server.port, timeout_seconds=2)
        try:
            with pytest.raises(BridgeError, match="disconnected"):
                environment.reset()
        finally:
            environment.close()


def test_malformed_state_is_rejected():
    with CounterServer(malformed_state=True) as server:
        environment = UnityGymEnv(port=server.port, timeout_seconds=2)
        try:
            environment.reset()
            with pytest.raises(BridgeError, match="Invalid Unity state"):
                environment.step([0.0])
        finally:
            environment.close()


def test_gymnasium_contract():
    with CounterServer() as server:
        environment = UnityGymEnv(port=server.port, timeout_seconds=2)
        try:
            check_env(environment, warn=False)
        finally:
            environment.close()


@pytest.mark.parametrize("model_name,model_class", [("ppo", PPO), ("a2c", A2C)])
def test_training_saves_model(tmp_path, capsys, model_name, model_class):
    with CounterServer() as server:
        config_path = tmp_path / "training.yaml"
        config = {
            "model": model_name,
            "port": server.port,
            "seed": 11,
            "total_timesteps": 16,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "n_steps": 8,
            "device": "cpu",
            "checkpoint_freq": 8,
            "output_dir": "runs",
            "socket_timeout_seconds": 5,
        }
        if model_name == "ppo":
            config["ppo"] = {"batch_size": 4}
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        final_path = train(config_path)
        assert final_path.is_file()
        assert list(final_path.parent.glob("checkpoint*.zip"))
        model_class.load(str(final_path))
        assert "Training complete:" in capsys.readouterr().out


def test_training_resumes_from_saved_model(tmp_path):
    config_path = tmp_path / "training.yaml"
    config = {
        "model": "ppo",
        "seed": 3,
        "total_timesteps": 16,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "n_steps": 8,
        "device": "cpu",
        "checkpoint_freq": 0,
        "output_dir": "runs",
        "ppo": {"batch_size": 4},
    }
    with CounterServer() as server:
        config["port"] = server.port
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        first_model = train(config_path)

    with CounterServer() as server:
        config["port"] = server.port
        config["resume_from"] = str(first_model)
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        resumed_model = train(config_path)

    assert resumed_model.is_file()
    assert PPO.load(str(resumed_model)).num_timesteps > PPO.load(str(first_model)).num_timesteps


def test_config_rejects_unknown_model(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("model: unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ppo or a2c"):
        load_config(config_path)


def test_cli_reports_failure(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("model: unknown\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    assert main() == 1
    assert "Training failed:" in capsys.readouterr().err
