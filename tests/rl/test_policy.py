# Author: Bradley R. Kinnard
"""Tests for MLP Policy and Evolution Strategy."""

import pytest
import numpy as np

from backend.rl.policy import (
    MLPPolicy,
    PolicyConfig,
    EvolutionStrategy,
)


class TestPolicyConfig:
    def test_default_values(self):
        cfg = PolicyConfig()
        assert cfg.state_dim == 15
        assert cfg.action_dim == 7
        assert cfg.hidden_sizes == (64, 64)

    def test_custom_values(self):
        cfg = PolicyConfig(state_dim=10, action_dim=5, hidden_sizes=(32,))
        assert cfg.state_dim == 10
        assert cfg.action_dim == 5
        assert cfg.hidden_sizes == (32,)


class TestMLPPolicy:
    def test_init_creates_weights(self):
        policy = MLPPolicy()
        assert len(policy.weights) == 3  # input->h1, h1->h2, h2->output
        assert len(policy.biases) == 3

    def test_param_count(self):
        cfg = PolicyConfig(state_dim=10, action_dim=5, hidden_sizes=(32, 32))
        policy = MLPPolicy(cfg)
        # 10*32 + 32 + 32*32 + 32 + 32*5 + 5 = 320+32+1024+32+160+5 = 1573
        expected = 10 * 32 + 32 + 32 * 32 + 32 + 32 * 5 + 5
        assert policy.param_count == expected

    def test_forward_shape(self):
        policy = MLPPolicy()
        state = np.zeros(15, dtype=np.float32)
        action = policy.forward(state)
        assert action.shape == (7,)

    def test_forward_bounded(self):
        policy = MLPPolicy()
        for _ in range(100):
            state = np.random.randn(15).astype(np.float32)
            action = policy.forward(state)
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_sample_action_stochastic(self):
        policy = MLPPolicy()
        state = np.zeros(15, dtype=np.float32)

        actions = [policy.sample_action(state, deterministic=False) for _ in range(10)]
        # should have variance
        actions = np.array(actions)
        assert np.std(actions) > 0.01

    def test_sample_action_deterministic(self):
        policy = MLPPolicy()
        state = np.zeros(15, dtype=np.float32)

        a1 = policy.sample_action(state, deterministic=True)
        a2 = policy.sample_action(state, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_get_set_params(self):
        policy = MLPPolicy()
        params = policy.get_params()

        assert isinstance(params, np.ndarray)
        assert len(params) == policy.param_count

        # modify and set back
        new_params = params * 2
        policy.set_params(new_params)
        np.testing.assert_array_almost_equal(policy.get_params(), new_params)

    def test_copy(self):
        policy = MLPPolicy()
        policy.set_params(np.random.randn(policy.param_count))

        copy = policy.copy()

        np.testing.assert_array_almost_equal(
            policy.get_params(), copy.get_params()
        )

    def test_save_load(self, tmp_path):
        policy = MLPPolicy()
        policy.set_params(np.random.randn(policy.param_count) * 0.1)
        original_params = policy.get_params().copy()

        path = str(tmp_path / "policy.npz")
        policy.save(path)

        # create new policy and load
        policy2 = MLPPolicy()
        policy2.load(path)

        np.testing.assert_array_almost_equal(
            policy2.get_params(), original_params
        )


class TestEvolutionStrategy:
    def test_ask_returns_population(self):
        policy = MLPPolicy()
        es = EvolutionStrategy(policy, population_size=10)

        perturbations = es.ask()

        assert len(perturbations) == 10
        assert all(len(p) == policy.param_count for p in perturbations)

    def test_tell_updates_params(self):
        policy = MLPPolicy()
        original_params = policy.get_params().copy()

        es = EvolutionStrategy(policy, population_size=5, learning_rate=0.1)
        perturbations = es.ask()

        # give positive reward to first, negative to rest
        rewards = [1.0] + [0.0] * 4

        es.tell(perturbations, rewards)

        # params should have changed
        new_params = policy.get_params()
        assert not np.allclose(original_params, new_params)

    def test_get_candidate(self):
        policy = MLPPolicy()
        es = EvolutionStrategy(policy, population_size=5)

        perturbations = es.ask()
        candidate = es.get_candidate(perturbations[0])

        # should be different from base policy
        assert not np.allclose(
            policy.get_params(),
            candidate.get_params(),
        )

    def test_training_improves(self):
        """Simple sanity check that ES can learn."""
        # target: action should be all 0.5
        target = np.ones(7) * 0.5

        cfg = PolicyConfig(state_dim=3, action_dim=7, hidden_sizes=(16,))
        policy = MLPPolicy(cfg)

        es = EvolutionStrategy(
            policy,
            learning_rate=0.05,
            noise_std=0.1,
            population_size=20,
        )

        state = np.zeros(3, dtype=np.float32)

        # initial error
        initial_action = policy.forward(state)
        initial_error = np.mean((initial_action - target) ** 2)

        # train for a few generations
        for _ in range(20):
            perturbations = es.ask()
            rewards = []

            for pert in perturbations:
                candidate = es.get_candidate(pert)
                action = candidate.forward(state)
                # reward = -MSE (higher is better)
                reward = -np.mean((action - target) ** 2)
                rewards.append(reward)

            es.tell(perturbations, rewards)

        # final error should be lower
        final_action = policy.forward(state)
        final_error = np.mean((final_action - target) ** 2)

        assert final_error < initial_error
