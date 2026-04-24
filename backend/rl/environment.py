# Author: Bradley R. Kinnard
"""
RL Environment - Gymnasium-compatible environment for belief ecology optimization.
Defines state, action, and reward spaces per spec 5.1.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, SupportsFloat

import numpy as np

from ..agents.rl_policy import EcologyState, PolicyAction
from ..agents.reward_shaper import RewardShaperAgent
from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the RL environment."""

    # episode settings
    max_steps_per_episode: int = 100
    target_belief_count: int = 500

    # action bounds
    decay_rate_min: float = 0.98
    decay_rate_max: float = 0.999
    threshold_min: float = 0.3
    threshold_max: float = 0.9
    weight_min: float = 0.1
    weight_max: float = 0.6

    # reward shaping
    task_reward_weight: float = 1.0
    consistency_weight: float = 0.3
    efficiency_weight: float = 0.2
    stability_weight: float = 0.2


@dataclass
class StepResult:
    """Result of a single environment step."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class BeliefEcologyEnv:
    """
    RL Environment for Belief Ecology optimization (spec 5.1).

    State space (continuous, 15 dimensions):
    - confidence_mean, confidence_std, confidence_min, confidence_max
    - tension_mean, tension_max, high_tension_count (normalized)
    - total_beliefs, active_count, decaying_count (normalized)
    - cluster_count (normalized)
    - recent_mutations, recent_deprecations, recent_reinforcements (normalized)
    - episode_progress

    Action space (continuous, 7 dimensions):
    - global_decay_rate adjustment (-0.01 to +0.01)
    - mutation_threshold adjustment (-0.1 to +0.1)
    - resolution_threshold adjustment (-0.1 to +0.1)
    - deprecation_threshold adjustment (-0.1 to +0.1)
    - ranking_weight_relevance (-0.1 to +0.1)
    - ranking_weight_confidence (-0.1 to +0.1)
    - beliefs_to_surface (1 to 20)
    """

    # state and action dimensions
    STATE_DIM = 15
    ACTION_DIM = 7

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.reward_shaper = RewardShaperAgent(
            task_weight=self.config.task_reward_weight,
            consistency_weight=self.config.consistency_weight,
            efficiency_weight=self.config.efficiency_weight,
            stability_weight=self.config.stability_weight,
            target_belief_count=self.config.target_belief_count,
        )

        # state tracking
        self._beliefs: list[Belief] = []
        self._step_count = 0
        self._prev_belief_count = 0
        self._episode_count = 0

        # current policy action (applied from last step)
        self._current_action = PolicyAction()

        # external callbacks for integration
        self._step_callback: Optional[callable] = None
        self._reset_callback: Optional[callable] = None

    def set_step_callback(self, callback: callable) -> None:
        """Set callback to run BEL iteration: (action) -> (beliefs, task_success)"""
        self._step_callback = callback

    def set_reset_callback(self, callback: callable) -> None:
        """Set callback to reset scenario: () -> beliefs"""
        self._reset_callback = callback

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.
        Returns initial observation and info dict.
        """
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0
        self._episode_count += 1
        self.reward_shaper.reset()
        self._current_action = PolicyAction()

        # get initial beliefs from callback or use empty
        if self._reset_callback:
            self._beliefs = self._reset_callback()
        else:
            self._beliefs = []

        self._prev_belief_count = len(self._beliefs)

        obs = self._get_observation()
        info = {
            "episode": self._episode_count,
            "initial_belief_count": len(self._beliefs),
        }

        return obs, info

    def step(self, action: np.ndarray) -> StepResult:
        """
        Take an action and advance the environment.

        Args:
            action: Array of shape (ACTION_DIM,) with normalized values.

        Returns:
            StepResult with observation, reward, termination flags, and info.
        """
        self._step_count += 1

        # decode action into PolicyAction
        policy_action = self._decode_action(action)
        self._current_action = policy_action

        # execute step via callback (runs BEL iteration with action params)
        task_success = 0.0
        consistency_score = 1.0
        contradiction_errors = 0
        core_beliefs_lost = 0

        if self._step_callback:
            result = self._step_callback(policy_action)
            self._beliefs = result.get("beliefs", self._beliefs)
            task_success = result.get("task_success", 0.0)
            consistency_score = result.get("consistency_score", 1.0)
            contradiction_errors = result.get("contradiction_errors", 0)
            core_beliefs_lost = result.get("core_beliefs_lost", 0)

        # compute reward
        reward_signal = self.reward_shaper.compute_reward(
            task_success=task_success,
            beliefs=self._beliefs,
            consistency_score=consistency_score,
            contradiction_errors=contradiction_errors,
            core_beliefs_lost=core_beliefs_lost,
            previous_belief_count=self._prev_belief_count,
        )

        self._prev_belief_count = len(self._beliefs)

        # check termination
        terminated = False
        truncated = self._step_count >= self.config.max_steps_per_episode

        # observation
        obs = self._get_observation()

        info = {
            "step": self._step_count,
            "reward_components": {
                "task": reward_signal.components.task_success,
                "consistency": reward_signal.components.consistency,
                "efficiency": reward_signal.components.memory_efficiency,
                "stability": reward_signal.components.stability,
                "contradiction_penalty": reward_signal.components.contradiction_penalty,
                "forgetting_penalty": reward_signal.components.forgetting_penalty,
            },
            "belief_count": len(self._beliefs),
            "action": {
                "decay_rate": policy_action.global_decay_rate,
                "mutation_threshold": policy_action.mutation_threshold,
                "resolution_threshold": policy_action.resolution_threshold,
            },
        }

        return StepResult(
            observation=obs,
            reward=reward_signal.total_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current state."""
        state = EcologyState.from_beliefs(
            self._beliefs,
            episode_step=self._step_count,
        )

        # normalize to roughly [0, 1] range
        max_beliefs = self.config.target_belief_count * 2

        obs = np.array([
            state.confidence_mean,
            min(1.0, state.confidence_std),
            state.confidence_min,
            state.confidence_max,
            min(1.0, state.tension_mean),
            min(1.0, state.tension_max),
            min(1.0, state.high_tension_count / 20.0),
            min(1.0, state.total_beliefs / max_beliefs),
            min(1.0, state.active_count / max_beliefs),
            min(1.0, state.decaying_count / max_beliefs),
            min(1.0, state.cluster_count / 100.0),
            min(1.0, state.recent_mutations / 10.0),
            min(1.0, state.recent_deprecations / 10.0),
            min(1.0, state.recent_reinforcements / 10.0),
            self._step_count / self.config.max_steps_per_episode,
        ], dtype=np.float32)

        return obs

    def _decode_action(self, action: np.ndarray) -> PolicyAction:
        """Convert action array to PolicyAction."""
        # action is expected to be in [-1, 1] range (from policy output)
        a = np.clip(action, -1.0, 1.0)

        # decode each dimension
        decay_delta = a[0] * 0.005  # ±0.005
        mutation_delta = a[1] * 0.1  # ±0.1
        resolution_delta = a[2] * 0.1
        deprecation_delta = a[3] * 0.1
        relevance_delta = a[4] * 0.1
        confidence_delta = a[5] * 0.1
        surface_count = int(5 + a[6] * 5)  # 0 to 10

        # apply deltas to current action
        return PolicyAction(
            global_decay_rate=np.clip(
                self._current_action.global_decay_rate + decay_delta,
                self.config.decay_rate_min,
                self.config.decay_rate_max,
            ),
            mutation_threshold=np.clip(
                self._current_action.mutation_threshold + mutation_delta,
                self.config.threshold_min,
                self.config.threshold_max,
            ),
            resolution_threshold=np.clip(
                self._current_action.resolution_threshold + resolution_delta,
                self.config.threshold_min,
                self.config.threshold_max,
            ),
            deprecation_threshold=np.clip(
                self._current_action.deprecation_threshold + deprecation_delta,
                0.05,
                0.3,
            ),
            ranking_weights={
                "relevance": np.clip(0.4 + relevance_delta, self.config.weight_min, self.config.weight_max),
                "confidence": np.clip(0.3 + confidence_delta, self.config.weight_min, self.config.weight_max),
            },
            beliefs_to_surface=max(1, min(20, surface_count)),
        )

    def get_episode_return(self) -> float:
        """Get cumulative return for current episode."""
        return self.reward_shaper.get_episode_return()

    def get_episode_stats(self) -> dict:
        """Get statistics for current episode."""
        return {
            "episode": self._episode_count,
            "steps": self._step_count,
            "total_return": self.get_episode_return(),
            "avg_reward": self.reward_shaper.get_average_reward(),
            "component_averages": self.reward_shaper.get_component_averages(),
            "final_belief_count": len(self._beliefs),
        }


__all__ = [
    "BeliefEcologyEnv",
    "EnvConfig",
    "StepResult",
]
