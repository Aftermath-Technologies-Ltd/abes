# Author: Bradley R. Kinnard
"""
TrajectoryBuffer - stores (state, action, reward, next_state, done) transitions
for policy gradient training. Supports FIFO eviction, batched sampling,
JSON persistence, and configurable capacity.
"""

import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryBuffer:
    """Fixed-capacity experience replay buffer with FIFO eviction.

    Stores transitions as (state, action, reward, next_state, done) tuples.
    When capacity is reached, the oldest entry is evicted automatically.
    States and actions are stored as plain Python lists for JSON portability.

    Args:
        max_size: Maximum number of transitions. Oldest entries are dropped
            when the buffer is full. Defaults to 10_000.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self._buffer: deque[dict] = deque(maxlen=max_size)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append a single transition. Oldest entry evicted when full.

        Args:
            state: Observation vector before the step.
            action: Action vector applied at this step.
            reward: Scalar reward received.
            next_state: Observation vector after the step.
            done: Whether the episode terminated after this step.
        """
        self._buffer.append({
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": float(reward),
            "next_state": next_state.tolist(),
            "done": bool(done),
        })

    def sample(self, batch_size: int) -> list[dict]:
        """Return `batch_size` transitions sampled uniformly without replacement.

        Args:
            batch_size: Number of transitions to return.

        Returns:
            List of transition dicts with keys state, action, reward,
            next_state, done. Each state/action value is a list of floats.

        Raises:
            ValueError: If batch_size exceeds the current buffer length.
        """
        n = len(self._buffer)
        if batch_size > n:
            raise ValueError(
                f"batch_size={batch_size} exceeds buffer length={n}; "
                "add more transitions before sampling"
            )
        return random.sample(list(self._buffer), batch_size)

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """Maximum capacity of the buffer."""
        return self._max_size

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize all transitions to a JSON file.

        Args:
            path: Destination file path. Parent directories are created
                if they do not exist.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_size": self._max_size,
            "transitions": list(self._buffer),
        }
        dest.write_text(json.dumps(payload, indent=2))
        logger.info("TrajectoryBuffer: saved %d transitions to %s", len(self._buffer), dest)

    def load(self, path: str | Path) -> None:
        """Replace buffer contents with transitions loaded from a JSON file.

        The file must have been written by :meth:`save`. Existing contents
        are cleared before loading.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If the file format is unrecognised.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"trajectory file not found: {src}")
        data = json.loads(src.read_text())
        if "transitions" not in data:
            raise ValueError(f"invalid trajectory file format: missing 'transitions' key in {src}")
        self.clear()
        for t in data["transitions"]:
            self._buffer.append(t)
        logger.info("TrajectoryBuffer: loaded %d transitions from %s", len(self._buffer), src)

    # ------------------------------------------------------------------
    # Helpers for training
    # ------------------------------------------------------------------

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all transitions as stacked NumPy arrays.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones), each a
            float32 array. dones is float32 (1.0 = terminal, 0.0 = not).
        """
        if not self._buffer:
            empty = np.empty((0,), dtype=np.float32)
            return empty, empty, empty, empty, empty
        transitions = list(self._buffer)
        states = np.array([t["state"] for t in transitions], dtype=np.float32)
        actions = np.array([t["action"] for t in transitions], dtype=np.float32)
        rewards = np.array([t["reward"] for t in transitions], dtype=np.float32)
        next_states = np.array([t["next_state"] for t in transitions], dtype=np.float32)
        dones = np.array([float(t["done"]) for t in transitions], dtype=np.float32)
        return states, actions, rewards, next_states, dones


__all__ = ["TrajectoryBuffer"]
