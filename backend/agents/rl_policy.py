# Author: Bradley R. Kinnard
"""
RLPolicyAgent - represents the current RL policy over belief ecology decisions.
Outputs parameter and control decisions for a given state.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class EcologyState:
    """
    State features for RL policy input (spec 5.1 State).
    Aggregated features of the belief ecology.
    """

    # distribution stats
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0

    # tension stats
    tension_mean: float = 0.0
    tension_max: float = 0.0
    high_tension_count: int = 0

    # counts
    total_beliefs: int = 0
    active_count: int = 0
    decaying_count: int = 0
    deprecated_count: int = 0
    cluster_count: int = 0

    # recent activity
    recent_mutations: int = 0
    recent_deprecations: int = 0
    recent_reinforcements: int = 0

    # task context
    episode_step: int = 0
    task_type: str = ""

    @classmethod
    def from_beliefs(
        cls,
        beliefs: list[Belief],
        episode_step: int = 0,
        task_type: str = "",
        recent_mutations: int = 0,
        recent_deprecations: int = 0,
        recent_reinforcements: int = 0,
    ) -> "EcologyState":
        """Compute state features from belief list."""
        if not beliefs:
            return cls(episode_step=episode_step, task_type=task_type)

        confidences = [b.confidence for b in beliefs]
        tensions = [b.tension for b in beliefs]
        clusters = {b.cluster_id for b in beliefs if b.cluster_id}

        import statistics

        return cls(
            confidence_mean=statistics.mean(confidences),
            confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            confidence_min=min(confidences),
            confidence_max=max(confidences),
            tension_mean=statistics.mean(tensions),
            tension_max=max(tensions),
            high_tension_count=sum(1 for t in tensions if t >= 0.6),
            total_beliefs=len(beliefs),
            active_count=sum(1 for b in beliefs if b.status == BeliefStatus.Active),
            decaying_count=sum(1 for b in beliefs if b.status == BeliefStatus.Decaying),
            deprecated_count=sum(1 for b in beliefs if b.status == BeliefStatus.Deprecated),
            cluster_count=len(clusters),
            recent_mutations=recent_mutations,
            recent_deprecations=recent_deprecations,
            recent_reinforcements=recent_reinforcements,
            episode_step=episode_step,
            task_type=task_type,
        )


@dataclass
class PolicyAction:
    """
    Action output from RL policy (spec 5.1 Action Space).
    """

    # decay rate adjustments
    global_decay_rate: float = settings.decay_rate
    cluster_decay_overrides: dict[str, float] = field(default_factory=dict)

    # threshold adjustments
    mutation_threshold: float = settings.tension_threshold_mutation
    resolution_threshold: float = settings.tension_threshold_resolution
    deprecation_threshold: float = settings.confidence_threshold_deprecated

    # ranking weights
    ranking_weights: dict[str, float] = field(default_factory=dict)

    # selection control
    beliefs_to_surface: int = 5

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RLPolicyAgent:
    """
    RL Policy Agent (spec 4.1 agent #10).
    Outputs control parameters for the belief ecology based on current state.

    This is a placeholder implementation using heuristics.
    Replace with trained policy when RL layer is implemented.
    """

    def __init__(self):
        self._default_action = PolicyAction()
        self._action_history: list[tuple[EcologyState, PolicyAction]] = []
        self._policy_weights: Optional[dict] = None  # for future trained policy

    def set_policy_weights(self, weights: dict) -> None:
        """Load trained policy weights."""
        self._policy_weights = weights
        logger.info("loaded policy weights")

    def get_action(self, state: EcologyState) -> PolicyAction:
        """
        Compute action given current state.
        Uses heuristics until trained policy is available.
        """
        if self._policy_weights is not None:
            return self._get_learned_action(state)

        return self._get_heuristic_action(state)

    def _get_heuristic_action(self, state: EcologyState) -> PolicyAction:
        """Rule-based fallback policy."""
        action = PolicyAction()

        # if high tension, lower mutation threshold to encourage more mutations
        if state.tension_max > 0.8 or state.high_tension_count > 5:
            action.mutation_threshold = 0.5
            action.resolution_threshold = 0.6

        # if too many decaying beliefs, slow decay
        if state.decaying_count > state.active_count * 0.3:
            action.global_decay_rate = min(0.998, settings.decay_rate + 0.002)

        # if belief count high, accelerate decay
        if state.total_beliefs > settings.max_active_beliefs * 0.8:
            action.global_decay_rate = max(0.99, settings.decay_rate - 0.003)
            action.deprecation_threshold = 0.15

        # adjust beliefs to surface based on available active beliefs
        action.beliefs_to_surface = min(10, max(3, state.active_count // 10))

        # log action
        self._action_history.append((state, action))

        return action

    def _get_learned_action(self, state: EcologyState) -> PolicyAction:
        """
        Placeholder for learned policy inference.
        Would use self._policy_weights to compute action.
        """
        # TODO: implement neural network forward pass
        logger.debug("using learned policy")
        return self._get_heuristic_action(state)

    def get_action_history(self, limit: int = 100) -> list[tuple[EcologyState, PolicyAction]]:
        """Get recent state-action pairs."""
        return self._action_history[-limit:]

    def reset(self) -> None:
        """Reset for new episode."""
        self._action_history.clear()


__all__ = [
    "RLPolicyAgent",
    "EcologyState",
    "PolicyAction",
]
