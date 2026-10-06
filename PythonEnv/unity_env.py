"""Use case: expose one Unity environment to a trainer. Inputs: actions and reset seeds. Outputs: observations, rewards, and episode endings."""

from __future__ import annotations

import math
import socket

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BridgeError(RuntimeError):
    pass


class UnityGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port: int = 7777, seed: int = 0, timeout_seconds: float = 300):
        super().__init__()
        self._socket = socket.create_connection(("127.0.0.1", port), timeout=timeout_seconds)
        self._socket.settimeout(timeout_seconds)
        self._stream = self._socket.makefile("rwb")
        self._next_seed = seed
        self._active = False
        try:
            self.observation_size, self.action_size = self._parse_handshake(self._read_line())
        except Exception:
            self.close()
            raise

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = self._next_seed if seed is None else seed
        if not isinstance(episode_seed, int) or not 0 <= episode_seed <= 2_147_483_647:
            raise ValueError("Reset seed must be a nonnegative 32-bit integer.")
        self._next_seed = (episode_seed + 1) % 2_147_483_648
        self._write_line(f"ACT=RESET;SEED={episode_seed}")
        observation, reward, done = self._parse_state(self._read_line())
        if reward != 0.0 or done:
            raise BridgeError("Unity returned an invalid reset state.")
        self._active = True
        return observation, {}

    def step(self, action):
        if not self._active:
            raise RuntimeError("Reset the environment before stepping.")

        values = np.asarray(action, dtype=np.float32)
        if values.shape != (self.action_size,) or not np.all(np.isfinite(values)):
            raise ValueError("Action must be a finite vector with the configured size.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("Action values must be between -1 and 1.")

        encoded = ",".join(format(float(value), ".9g") for value in values)
        self._write_line(f"ACT={encoded}")
        observation, reward, done = self._parse_state(self._read_line())
        self._active = not done
        return observation, reward, done, False, {}

    def close(self):
        self._active = False
        if getattr(self, "_stream", None) is not None:
            self._stream.close()
            self._stream = None
        if getattr(self, "_socket", None) is not None:
            self._socket.close()
            self._socket = None

    def _write_line(self, message: str):
        self._stream.write((message + "\n").encode("utf-8"))
        self._stream.flush()

    def _read_line(self) -> str:
        line = self._stream.readline()
        if not line:
            self._active = False
            raise BridgeError("Unity disconnected.")
        message = line.decode("utf-8").rstrip("\r\n")
        if message.startswith("ERROR="):
            self._active = False
            raise BridgeError(f"Unity bridge error: {message[6:]}")
        return message

    @staticmethod
    def _parse_handshake(message: str) -> tuple[int, int]:
        if not message.startswith("CONFIG:"):
            raise BridgeError("Expected a Unity configuration handshake.")
        try:
            fields = dict(part.split("=", 1) for part in message[7:].split(";"))
            observation_size = int(fields["OBS"])
            action_size = int(fields["ACT"])
            if fields["ENV_TYPE"] != "SINGLE" or observation_size <= 0 or action_size <= 0:
                raise ValueError
            return observation_size, action_size
        except (KeyError, ValueError) as error:
            raise BridgeError("Invalid Unity configuration handshake.") from error

    def _parse_state(self, message: str) -> tuple[np.ndarray, float, bool]:
        try:
            fields = dict(part.split("=", 1) for part in message.split(";"))
            values = [float(part) for part in fields["OBS"].split(",")]
            observation = np.asarray(values, dtype=np.float32)
            reward = float(fields["REW"])
            done_text = fields["DONE"]
            if (
                observation.shape != (self.observation_size,)
                or not np.all(np.isfinite(observation))
                or not math.isfinite(reward)
                or done_text not in ("0", "1")
            ):
                raise ValueError
            return observation, reward, done_text == "1"
        except (KeyError, ValueError) as error:
            raise BridgeError("Invalid Unity state response.") from error
