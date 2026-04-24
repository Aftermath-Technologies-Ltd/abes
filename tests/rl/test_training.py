# Author: Bradley R. Kinnard
"""Tests for ES-based RL Training."""

import pytest
import numpy as np

from backend.rl.training import (
    ESTrainer,
    RLTrainer,
    TrainingConfig,
    TrainingMetrics,
    RolloutBuffer,
)
from backend.rl.environment import BeliefEcologyEnv, EnvConfig
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(confidence: float = 0.8) -> Belief:
    return Belief(
        content="test",
        confidence=confidence,
        status=BeliefStatus.Active,
        origin=OriginMetadata(source="test"),
    )


class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.total_generations == 100
        assert cfg.population_size == 20
        assert cfg.learning_rate == 0.03

    def test_custom_values(self):
        cfg = TrainingConfig(total_generations=50, population_size=10)
        assert cfg.total_generations == 50
        assert cfg.population_size == 10


class TestTrainingMetrics:
    def test_to_dict(self):
        metrics = TrainingMetrics(generation=5, avg_return=10.5)
        d = metrics.to_dict()

        assert d["generation"] == 5
        assert d["avg_return"] == 10.5
        assert "best_return" in d


class TestRolloutBuffer:
    def test_add_and_len(self):
        buffer = RolloutBuffer()
        buffer.add(
            state=np.zeros(5),
            action=np.zeros(3),
            reward=1.0,
            value=0.5,
            log_prob=-0.1,
            done=False,
        )
        assert len(buffer) == 1

    def test_clear(self):
        buffer = RolloutBuffer()
        buffer.add(np.zeros(5), np.zeros(3), 1.0, 0.5, -0.1, False)
        buffer.clear()
        assert len(buffer) == 0

    def test_is_full(self):
        buffer = RolloutBuffer()
        for _ in range(10):
            buffer.add(np.zeros(5), np.zeros(3), 1.0, 0.5, -0.1, False)

        assert buffer.is_full(10)
        assert not buffer.is_full(20)

    def test_compute_returns_empty(self):
        buffer = RolloutBuffer()
        returns, advantages = buffer.compute_returns_and_advantages(0.99, 0.95)
        assert len(returns) == 0
        assert len(advantages) == 0

    def test_compute_returns_and_advantages(self):
        buffer = RolloutBuffer()
        for i in range(5):
            buffer.add(
                state=np.zeros(5),
                action=np.zeros(3),
                reward=1.0,
                value=0.5,
                log_prob=-0.1,
                done=i == 4,
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=0.99, gae_lambda=0.95
        )

        assert len(returns) == 5
        assert len(advantages) == 5
        assert returns.dtype == np.float32


class TestESTrainer:
    def test_init(self):
        env = BeliefEcologyEnv()
        trainer = ESTrainer(env)

        assert trainer.env is env
        assert trainer.policy is not None
        assert trainer.es is not None

    def test_evaluate_candidate(self):
        env_config = EnvConfig(max_steps_per_episode=5)
        env = BeliefEcologyEnv(env_config)
        trainer = ESTrainer(env)

        mean_return = trainer._evaluate_candidate(trainer.policy, num_episodes=2)

        assert isinstance(mean_return, float)
        assert trainer._metrics.total_episodes == 2

    def test_evaluate_policy(self):
        env_config = EnvConfig(max_steps_per_episode=5)
        env = BeliefEcologyEnv(env_config)
        config = TrainingConfig(eval_episodes=3)
        trainer = ESTrainer(env, config)

        mean_return = trainer._evaluate_policy()

        assert isinstance(mean_return, float)

    def test_train_short(self):
        """Test training loop runs to completion."""
        env_config = EnvConfig(max_steps_per_episode=5)
        env = BeliefEcologyEnv(env_config)

        config = TrainingConfig(
            total_generations=3,
            episodes_per_candidate=1,
            population_size=4,
            eval_frequency=2,
            eval_episodes=1,
            patience=100,  # disable early stopping
        )
        trainer = ESTrainer(env, config)

        metrics = trainer.train()

        assert metrics.generation == 3
        assert metrics.total_episodes == 3 * 4 * 1  # gens * pop * eps

    def test_save_load_policy(self, tmp_path):
        env = BeliefEcologyEnv()
        trainer = ESTrainer(env)

        # set some params
        trainer.policy.set_params(np.random.randn(trainer.policy.param_count) * 0.1)
        original = trainer.policy.get_params().copy()

        path = str(tmp_path / "policy.npz")
        trainer.save_policy(path)

        # create new trainer and load
        trainer2 = ESTrainer(env)
        trainer2.load_policy(path)

        np.testing.assert_array_almost_equal(
            trainer2.policy.get_params(), original
        )

    def test_callbacks_called(self):
        env_config = EnvConfig(max_steps_per_episode=3)
        env = BeliefEcologyEnv(env_config)

        config = TrainingConfig(
            total_generations=2,
            episodes_per_candidate=1,
            population_size=2,
            eval_frequency=1,
            eval_episodes=1,
        )
        trainer = ESTrainer(env, config)

        gen_calls = []
        eval_calls = []

        trainer.set_on_generation_end(lambda m: gen_calls.append(m.generation))
        trainer.set_on_eval(lambda m: eval_calls.append(m.eval_return))

        trainer.train()

        assert len(gen_calls) == 2
        assert len(eval_calls) == 2


class TestRLTrainerAlias:
    def test_alias_works(self):
        # RLTrainer should be alias to ESTrainer
        assert RLTrainer is ESTrainer
