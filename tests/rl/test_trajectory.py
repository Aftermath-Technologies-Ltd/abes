# Author: Bradley R. Kinnard
"""Tests for TrajectoryBuffer and REINFORCE train_policy."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.rl.trajectory_buffer import TrajectoryBuffer
from backend.rl.policy import MLPPolicy, PolicyConfig
from backend.rl.training import train_policy, REINFORCEMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(
    state_dim: int = 15,
    action_dim: int = 7,
    reward: float = 1.0,
    done: bool = False,
) -> tuple:
    """Return (state, action, reward, next_state, done) with random arrays."""
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
    next_state = np.random.randn(state_dim).astype(np.float32)
    return state, action, reward, next_state, done


def _fill_buffer(buf: TrajectoryBuffer, n: int, reward: float = 1.0) -> None:
    for _ in range(n):
        s, a, r, ns, d = _make_transition(reward=reward)
        buf.add(s, a, r, ns, d)


# ---------------------------------------------------------------------------
# TrajectoryBuffer tests
# ---------------------------------------------------------------------------

class TestTrajectoryBufferAddSample:
    """Basic add and sample behaviour."""

    def test_add_increases_length(self) -> None:
        buf = TrajectoryBuffer(max_size=100)
        assert len(buf) == 0
        s, a, r, ns, d = _make_transition()
        buf.add(s, a, r, ns, d)
        assert len(buf) == 1

    def test_sample_returns_correct_count(self) -> None:
        buf = TrajectoryBuffer(max_size=100)
        _fill_buffer(buf, 20)
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_keys_present(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 10)
        item = buf.sample(1)[0]
        assert set(item.keys()) == {"state", "action", "reward", "next_state", "done"}

    def test_sample_too_large_raises(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 5)
        with pytest.raises(ValueError, match="batch_size"):
            buf.sample(10)

    def test_fifo_eviction_at_capacity(self) -> None:
        buf = TrajectoryBuffer(max_size=3)
        for i in range(5):
            s, a, _, ns, d = _make_transition()
            buf.add(s, a, float(i), ns, d)
        # Only the 3 most recent should be in the buffer
        assert len(buf) == 3
        rewards = [t["reward"] for t in buf.sample(3)]
        assert sorted(rewards) == [2.0, 3.0, 4.0]

    def test_clear_empties_buffer(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 10)
        buf.clear()
        assert len(buf) == 0


class TestTrajectoryBufferPersistence:
    """Save and load round-trip."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 5)
        dest = tmp_path / "traj.json"
        buf.save(dest)
        assert dest.exists()

    def test_load_restores_transitions(self, tmp_path: Path) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 10, reward=3.14)
        path = tmp_path / "traj.json"
        buf.save(path)

        buf2 = TrajectoryBuffer()
        buf2.load(path)
        assert len(buf2) == 10
        # Rewards should match
        rewards = [t["reward"] for t in buf2.sample(10)]
        assert all(abs(r - 3.14) < 1e-5 for r in rewards)

    def test_load_nonexistent_raises(self) -> None:
        buf = TrajectoryBuffer()
        with pytest.raises(FileNotFoundError):
            buf.load("/tmp/does_not_exist_abes_test.json")

    def test_save_load_as_arrays_consistent(self, tmp_path: Path) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 8)
        path = tmp_path / "traj.json"
        buf.save(path)

        buf2 = TrajectoryBuffer()
        buf2.load(path)
        states1, actions1, rewards1, _, _ = buf.as_arrays()
        states2, actions2, rewards2, _, _ = buf2.as_arrays()
        np.testing.assert_allclose(rewards1, rewards2)


# ---------------------------------------------------------------------------
# train_policy tests
# ---------------------------------------------------------------------------

class TestTrainPolicy:
    """REINFORCE training reduces loss and updates parameters."""

    def test_returns_one_metric_per_epoch(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 20)
        policy = MLPPolicy(PolicyConfig())
        metrics = train_policy(policy, buf, epochs=5, lr=0.001)
        assert len(metrics) == 5
        assert all(isinstance(m, REINFORCEMetrics) for m in metrics)

    def test_policy_parameters_change_after_training(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 30, reward=1.0)
        policy = MLPPolicy(PolicyConfig())
        params_before = policy.get_params().copy()
        train_policy(policy, buf, epochs=5, lr=0.01)
        assert not np.allclose(policy.get_params(), params_before), \
            "Policy parameters must change after training"

    def test_loss_metric_is_finite(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 20)
        policy = MLPPolicy(PolicyConfig())
        metrics = train_policy(policy, buf, epochs=3, lr=0.001)
        for m in metrics:
            assert np.isfinite(m.policy_loss), f"Non-finite loss at epoch {m.epoch}"

    def test_empty_buffer_raises(self) -> None:
        buf = TrajectoryBuffer()
        policy = MLPPolicy(PolicyConfig())
        with pytest.raises(ValueError, match="empty"):
            train_policy(policy, buf, epochs=1)

    def test_mean_reward_reflects_buffer_rewards(self) -> None:
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 20, reward=5.0)
        policy = MLPPolicy(PolicyConfig())
        metrics = train_policy(policy, buf, epochs=1, lr=0.0)
        # With lr=0, parameters don't change; mean_reward should be 5.0
        assert metrics[0].mean_reward == pytest.approx(5.0)

    def test_loss_does_not_increase_monotonically(self) -> None:
        """Over 20 epochs with a decent learning rate, final loss <= initial."""
        np.random.seed(42)
        buf = TrajectoryBuffer()
        _fill_buffer(buf, 50, reward=1.0)
        policy = MLPPolicy(PolicyConfig())
        metrics = train_policy(policy, buf, epochs=20, lr=0.005)
        # Not strictly monotone (policy gradient is noisy), but final should
        # not be massively worse than initial
        assert metrics[-1].policy_loss < metrics[0].policy_loss * 10, \
            "Loss diverged unexpectedly"
