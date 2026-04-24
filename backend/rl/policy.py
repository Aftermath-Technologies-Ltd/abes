# Author: Bradley R. Kinnard
"""
Neural network policy implemented in pure NumPy.
Uses a simple MLP with tanh activations for continuous action output.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for the policy network."""

    state_dim: int = 15
    action_dim: int = 7
    hidden_sizes: tuple[int, ...] = (64, 64)
    init_std: float = 0.1
    action_std: float = 0.3  # exploration noise


def _init_weights(shape: tuple[int, ...], std: float = 0.1) -> np.ndarray:
    """Xavier-like initialization scaled by std."""
    fan_in = shape[0] if len(shape) > 1 else shape[0]
    scale = std * np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape).astype(np.float32) * scale


def _init_bias(size: int) -> np.ndarray:
    """Zero-initialize biases."""
    return np.zeros(size, dtype=np.float32)


class MLPPolicy:
    """
    Simple MLP policy for continuous actions.
    Forward pass: state -> hidden layers with tanh -> action mean.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self._build_network()

    def _build_network(self) -> None:
        """Initialize weight matrices and biases."""
        cfg = self.config
        sizes = [cfg.state_dim] + list(cfg.hidden_sizes) + [cfg.action_dim]

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(sizes) - 1):
            w = _init_weights((sizes[i], sizes[i + 1]), cfg.init_std)
            b = _init_bias(sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        self._param_count = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        logger.debug(f"policy network: {self._param_count} parameters")

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Compute action mean from state.
        Returns action in [-1, 1] range via tanh.
        """
        x = state.astype(np.float32)

        # hidden layers with tanh
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.tanh(x)

        # output layer with tanh for bounded actions
        x = x @ self.weights[-1] + self.biases[-1]
        action_mean = np.tanh(x)

        return action_mean

    def sample_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sample action from policy.
        Adds Gaussian noise for exploration unless deterministic.
        """
        mean = self.forward(state)

        if deterministic:
            return mean

        noise = np.random.randn(*mean.shape) * self.config.action_std
        action = mean + noise
        return np.clip(action, -1.0, 1.0)

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.flatten())
            params.append(b.flatten())
        return np.concatenate(params)

    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from a flattened vector."""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size

            self.weights[i] = params[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size

            self.biases[i] = params[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size

    @property
    def param_count(self) -> int:
        return self._param_count

    def save(self, path: str) -> None:
        """Save policy weights to file."""
        data = {
            "config_state_dim": self.config.state_dim,
            "config_action_dim": self.config.action_dim,
            "config_hidden_sizes": np.array(self.config.hidden_sizes),
        }
        # save each weight/bias separately to avoid inhomogeneous array issue
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"weight_{i}"] = w
            data[f"bias_{i}"] = b
        data["num_layers"] = len(self.weights)

        np.savez(path, **data)
        logger.info(f"saved policy to {path}")

    def load(self, path: str) -> None:
        """Load policy weights from file."""
        data = np.load(path, allow_pickle=True)
        num_layers = int(data["num_layers"])

        self.weights = []
        self.biases = []
        for i in range(num_layers):
            self.weights.append(data[f"weight_{i}"])
            self.biases.append(data[f"bias_{i}"])

        self._param_count = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        logger.info(f"loaded policy from {path}")

    def copy(self) -> "MLPPolicy":
        """Create a copy with same weights."""
        new_policy = MLPPolicy(self.config)
        new_policy.set_params(self.get_params().copy())
        return new_policy


class EvolutionStrategy:
    """
    Natural Evolution Strategy (NES) for policy optimization.
    Gradient-free optimization using population-based search.
    """

    def __init__(
        self,
        policy: MLPPolicy,
        learning_rate: float = 0.01,
        noise_std: float = 0.02,
        population_size: int = 20,
    ):
        self.policy = policy
        self.lr = learning_rate
        self.noise_std = noise_std
        self.pop_size = population_size

        self._base_params = policy.get_params().copy()

    def ask(self) -> list[np.ndarray]:
        """Generate population of parameter perturbations."""
        perturbations = []
        for _ in range(self.pop_size):
            noise = np.random.randn(self.policy.param_count) * self.noise_std
            perturbations.append(noise)
        return perturbations

    def tell(self, perturbations: list[np.ndarray], rewards: list[float]) -> None:
        """Update policy based on rewards for each perturbation."""
        rewards = np.array(rewards)

        # normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        # compute weighted update
        gradient = np.zeros(self.policy.param_count)
        for noise, r in zip(perturbations, rewards):
            gradient += r * noise
        gradient /= (self.pop_size * self.noise_std)

        # update base parameters
        self._base_params += self.lr * gradient
        self.policy.set_params(self._base_params)

    def get_candidate(self, perturbation: np.ndarray) -> MLPPolicy:
        """Create policy with perturbed parameters."""
        candidate = self.policy.copy()
        candidate.set_params(self._base_params + perturbation)
        return candidate


__all__ = [
    "PolicyConfig",
    "MLPPolicy",
    "EvolutionStrategy",
]
