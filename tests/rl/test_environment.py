# Author: Bradley R. Kinnard
"""Tests for RL Environment."""

import pytest
import numpy as np

from backend.rl.environment import BeliefEcologyEnv, EnvConfig, StepResult
from backend.agents.rl_policy import PolicyAction
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "test",
    confidence: float = 0.8,
    status: BeliefStatus = BeliefStatus.Active,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        status=status,
        origin=OriginMetadata(source="test"),
    )


class TestEnvConfig:
    def test_default_values(self):
        config = EnvConfig()
        assert config.max_steps_per_episode == 100
        assert config.target_belief_count == 500

    def test_custom_values(self):
        config = EnvConfig(max_steps_per_episode=50, target_belief_count=200)
        assert config.max_steps_per_episode == 50
        assert config.target_belief_count == 200


class TestEnvReset:
    def test_reset_returns_observation(self):
        env = BeliefEcologyEnv()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (BeliefEcologyEnv.STATE_DIM,)
        assert "episode" in info

    def test_reset_increments_episode(self):
        env = BeliefEcologyEnv()
        env.reset()
        _, info1 = env.reset()
        _, info2 = env.reset()

        assert info2["episode"] == info1["episode"] + 1

    def test_reset_with_callback(self):
        env = BeliefEcologyEnv()
        beliefs = [_make_belief() for _ in range(10)]

        env.set_reset_callback(lambda: beliefs)
        _, info = env.reset()

        assert info["initial_belief_count"] == 10

    def test_reset_clears_step_count(self):
        env = BeliefEcologyEnv()
        env._step_count = 50
        env.reset()
        assert env._step_count == 0


class TestEnvStep:
    def test_step_returns_result(self):
        env = BeliefEcologyEnv()
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, np.ndarray)
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)

    def test_step_increments_count(self):
        env = BeliefEcologyEnv()
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        assert result.info["step"] == 1

    def test_step_with_callback(self):
        env = BeliefEcologyEnv()
        beliefs = [_make_belief() for _ in range(5)]

        def step_callback(policy_action):
            return {
                "beliefs": beliefs,
                "task_success": 0.8,
                "consistency_score": 0.9,
            }

        env.set_step_callback(step_callback)
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        assert result.info["belief_count"] == 5
        assert result.reward > 0

    def test_truncation_at_max_steps(self):
        config = EnvConfig(max_steps_per_episode=3)
        env = BeliefEcologyEnv(config)
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)

        r1 = env.step(action)
        r2 = env.step(action)
        r3 = env.step(action)

        assert not r1.truncated
        assert not r2.truncated
        assert r3.truncated


class TestActionDecoding:
    def test_zero_action_minimal_change(self):
        env = BeliefEcologyEnv()
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        # decay rate should be close to default
        assert abs(result.info["action"]["decay_rate"] - 0.995) < 0.01

    def test_positive_action_increases(self):
        env = BeliefEcologyEnv()
        env.reset()

        # all ones should push values up
        action = np.ones(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        # mutation threshold should increase or reach upper bound
        # default is 0.5, delta is +0.1, so result is 0.6
        assert result.info["action"]["mutation_threshold"] >= 0.6

    def test_negative_action_decreases(self):
        env = BeliefEcologyEnv()
        env.reset()

        # all negative should push values down
        action = -np.ones(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        # threshold should decrease (but stay in bounds)
        assert result.info["action"]["mutation_threshold"] < 0.6

    def test_action_clipping(self):
        env = BeliefEcologyEnv()
        env.reset()

        # extreme values should be clipped
        action = np.ones(BeliefEcologyEnv.ACTION_DIM) * 100
        result = env.step(action)

        # should not exceed max bounds
        assert result.info["action"]["decay_rate"] <= 0.999


class TestObservation:
    def test_observation_shape(self):
        env = BeliefEcologyEnv()
        obs, _ = env.reset()

        assert obs.shape == (BeliefEcologyEnv.STATE_DIM,)

    def test_observation_normalized(self):
        env = BeliefEcologyEnv()
        beliefs = [_make_belief(confidence=0.5) for _ in range(100)]
        env.set_reset_callback(lambda: beliefs)
        obs, _ = env.reset()

        # all values should be roughly in [0, 1]
        assert np.all(obs >= 0)
        assert np.all(obs <= 1.5)  # some slack for edge cases

    def test_observation_dtype(self):
        env = BeliefEcologyEnv()
        obs, _ = env.reset()

        assert obs.dtype == np.float32


class TestEpisodeStats:
    def test_episode_return(self):
        env = BeliefEcologyEnv()

        def step_callback(action):
            return {"beliefs": [], "task_success": 0.5}

        env.set_step_callback(step_callback)
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        env.step(action)
        env.step(action)

        total_return = env.get_episode_return()
        assert total_return > 0

    def test_episode_stats(self):
        env = BeliefEcologyEnv()
        env.set_step_callback(lambda a: {"beliefs": [], "task_success": 1.0})
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        env.step(action)

        stats = env.get_episode_stats()

        assert "episode" in stats
        assert "steps" in stats
        assert "total_return" in stats
        assert "avg_reward" in stats


class TestRewardInfo:
    def test_reward_components_in_info(self):
        env = BeliefEcologyEnv()
        env.set_step_callback(lambda a: {"beliefs": [], "task_success": 1.0})
        env.reset()

        action = np.zeros(BeliefEcologyEnv.ACTION_DIM)
        result = env.step(action)

        components = result.info["reward_components"]
        assert "task" in components
        assert "consistency" in components
        assert "efficiency" in components
        assert "contradiction_penalty" in components
