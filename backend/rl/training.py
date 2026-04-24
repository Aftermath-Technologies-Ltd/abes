# Author: Bradley R. Kinnard
"""
RL Training Loop - trains policies over belief ecology episodes.
Uses evolution strategy (ES) for gradient-free policy optimization.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .environment import BeliefEcologyEnv, EnvConfig, StepResult
from .policy import MLPPolicy, PolicyConfig, EvolutionStrategy
from ..agents.rl_policy import EcologyState, PolicyAction

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for ES-based RL training."""

    # training loop
    total_generations: int = 100
    episodes_per_candidate: int = 3
    max_steps_per_episode: int = 100
    eval_frequency: int = 10
    eval_episodes: int = 5

    # ES parameters
    population_size: int = 20
    learning_rate: float = 0.03
    noise_std: float = 0.05

    # policy network
    hidden_sizes: tuple[int, ...] = (64, 64)
    action_std: float = 0.2

    # early stopping
    target_return: Optional[float] = None
    patience: int = 20

    # checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_frequency: int = 25


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    generation: int = 0
    total_episodes: int = 0
    best_return: float = float("-inf")
    avg_return: float = 0.0
    std_return: float = 0.0
    eval_return: float = 0.0
    generations_without_improvement: int = 0

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "total_episodes": self.total_episodes,
            "best_return": self.best_return,
            "avg_return": self.avg_return,
            "std_return": self.std_return,
            "eval_return": self.eval_return,
        }


class ESTrainer:
    """
    Evolution Strategy trainer for belief ecology policies.
    Uses population-based search to optimize the policy network.
    """

    def __init__(
        self,
        env: BeliefEcologyEnv,
        config: Optional[TrainingConfig] = None,
    ):
        self.env = env
        self.config = config or TrainingConfig()

        # build policy
        policy_cfg = PolicyConfig(
            state_dim=BeliefEcologyEnv.STATE_DIM,
            action_dim=BeliefEcologyEnv.ACTION_DIM,
            hidden_sizes=self.config.hidden_sizes,
            action_std=self.config.action_std,
        )
        self.policy = MLPPolicy(policy_cfg)

        # ES optimizer
        self.es = EvolutionStrategy(
            policy=self.policy,
            learning_rate=self.config.learning_rate,
            noise_std=self.config.noise_std,
            population_size=self.config.population_size,
        )

        self._metrics = TrainingMetrics()
        self._best_params: Optional[np.ndarray] = None
        self._return_history: list[float] = []

        # callbacks
        self._on_generation_end: Optional[Callable[[TrainingMetrics], None]] = None
        self._on_eval: Optional[Callable[[TrainingMetrics], None]] = None

    def set_on_generation_end(self, callback: Callable[[TrainingMetrics], None]) -> None:
        self._on_generation_end = callback

    def set_on_eval(self, callback: Callable[[TrainingMetrics], None]) -> None:
        self._on_eval = callback

    def _evaluate_candidate(self, policy: MLPPolicy, num_episodes: int) -> float:
        """Run episodes with a candidate policy and return mean reward."""
        returns = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_return = 0.0

            while not done:
                action = policy.sample_action(obs, deterministic=False)
                result = self.env.step(action)
                obs = result.observation
                ep_return += result.reward
                done = result.terminated or result.truncated

            returns.append(ep_return)
            self._metrics.total_episodes += 1

        return float(np.mean(returns))

    def _evaluate_policy(self) -> float:
        """Evaluate current policy deterministically."""
        returns = []

        for _ in range(self.config.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_return = 0.0

            while not done:
                action = self.policy.sample_action(obs, deterministic=True)
                result = self.env.step(action)
                obs = result.observation
                ep_return += result.reward
                done = result.terminated or result.truncated

            returns.append(ep_return)

        return float(np.mean(returns))

    def train(self) -> TrainingMetrics:
        """
        Run ES training loop.
        Returns final metrics.
        """
        logger.info(
            f"starting ES training: {self.config.total_generations} generations, "
            f"pop_size={self.config.population_size}"
        )

        for gen in range(1, self.config.total_generations + 1):
            self._metrics.generation = gen

            # sample perturbations
            perturbations = self.es.ask()

            # evaluate each candidate
            rewards = []
            for pert in perturbations:
                candidate = self.es.get_candidate(pert)
                reward = self._evaluate_candidate(
                    candidate, self.config.episodes_per_candidate
                )
                rewards.append(reward)

            # update policy
            self.es.tell(perturbations, rewards)

            # track metrics
            self._metrics.avg_return = float(np.mean(rewards))
            self._metrics.std_return = float(np.std(rewards))
            self._return_history.append(self._metrics.avg_return)

            # check for improvement
            if self._metrics.avg_return > self._metrics.best_return:
                self._metrics.best_return = self._metrics.avg_return
                self._best_params = self.policy.get_params().copy()
                self._metrics.generations_without_improvement = 0
            else:
                self._metrics.generations_without_improvement += 1

            # periodic evaluation
            if gen % self.config.eval_frequency == 0:
                self._metrics.eval_return = self._evaluate_policy()
                logger.info(
                    f"gen {gen}: avg={self._metrics.avg_return:.3f}, "
                    f"eval={self._metrics.eval_return:.3f}, "
                    f"best={self._metrics.best_return:.3f}"
                )
                if self._on_eval:
                    self._on_eval(self._metrics)
            else:
                logger.debug(f"gen {gen}: avg={self._metrics.avg_return:.3f}")

            # checkpoint
            if self.config.checkpoint_dir and gen % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(gen)

            # callback
            if self._on_generation_end:
                self._on_generation_end(self._metrics)

            # early stopping
            if self.config.target_return and self._metrics.eval_return >= self.config.target_return:
                logger.info(f"reached target return {self.config.target_return}")
                break

            if self._metrics.generations_without_improvement >= self.config.patience:
                logger.info(f"no improvement for {self.config.patience} generations, stopping")
                break

        # restore best params
        if self._best_params is not None:
            self.policy.set_params(self._best_params)
            logger.info("restored best parameters")

        logger.info(
            f"training complete: {self._metrics.total_episodes} episodes, "
            f"best_return={self._metrics.best_return:.3f}"
        )
        return self._metrics

    def _save_checkpoint(self, generation: int) -> None:
        """Save checkpoint to disk."""
        if self.config.checkpoint_dir is None:
            return

        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.policy.save(str(path / f"policy_gen{generation}.npz"))

        # save metrics
        metrics_path = path / f"metrics_gen{generation}.npz"
        np.savez(
            metrics_path,
            return_history=np.array(self._return_history),
            **self._metrics.to_dict(),
        )

    def save_policy(self, path: str) -> None:
        """Save current policy."""
        self.policy.save(path)

    def load_policy(self, path: str) -> None:
        """Load policy from file."""
        self.policy.load(path)


# Legacy compatibility alias
RLTrainer = ESTrainer


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data (kept for compatibility)."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def is_full(self, buffer_size: int) -> bool:
        return len(self) >= buffer_size

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n = len(self)
        if n == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        return returns, advantages


__all__ = [
    "ESTrainer",
    "RLTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "RolloutBuffer",
]
