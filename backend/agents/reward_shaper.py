# Author: Bradley R. Kinnard
"""
RewardShaperAgent - computes RL rewards based on task performance and ecology health.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Breakdown of reward signal."""

    task_success: float = 0.0
    consistency: float = 0.0
    memory_efficiency: float = 0.0
    stability: float = 0.0
    contradiction_penalty: float = 0.0
    forgetting_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.task_success
            + self.consistency
            + self.memory_efficiency
            + self.stability
            - self.contradiction_penalty
            - self.forgetting_penalty
        )


@dataclass
class RewardSignal:
    """Complete reward signal for one step."""

    components: RewardComponents
    total_reward: float
    step: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RewardShaperAgent:
    """
    Reward Shaper Agent (spec 4.1 agent #11, spec 5.1 Reward).
    Computes shaped reward signals for RL training.
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        consistency_weight: float = 0.3,
        efficiency_weight: float = 0.2,
        stability_weight: float = 0.2,
        contradiction_penalty_weight: float = 0.5,
        forgetting_penalty_weight: float = 0.3,
        target_belief_count: int = 500,
    ):
        self._w_task = task_weight
        self._w_consistency = consistency_weight
        self._w_efficiency = efficiency_weight
        self._w_stability = stability_weight
        self._w_contradiction = contradiction_penalty_weight
        self._w_forgetting = forgetting_penalty_weight
        self._target_count = target_belief_count

        self._reward_history: list[RewardSignal] = []
        self._step = 0

    def compute_reward(
        self,
        task_success: float,  # 0-1 task performance
        beliefs: list[Belief],
        consistency_score: float = 1.0,  # from consistency checker
        contradiction_errors: int = 0,
        core_beliefs_lost: int = 0,
        previous_belief_count: Optional[int] = None,
    ) -> RewardSignal:
        """
        Compute shaped reward for current step.

        Args:
            task_success: Task performance metric (0-1)
            beliefs: Current belief list
            consistency_score: Score from consistency checker (0-1)
            contradiction_errors: Count of contradiction-related errors
            core_beliefs_lost: Count of core beliefs that were deprecated
            previous_belief_count: Belief count at previous step (for stability)
        """
        self._step += 1

        # task success (direct performance)
        r_task = self._w_task * task_success

        # consistency bonus
        r_consistency = self._w_consistency * consistency_score

        # memory efficiency (penalize excessive beliefs)
        active_count = sum(1 for b in beliefs if b.status == BeliefStatus.Active)
        efficiency = 1.0 - min(1.0, abs(active_count - self._target_count) / self._target_count)
        r_efficiency = self._w_efficiency * efficiency

        # stability (penalize wild swings in belief count)
        r_stability = 0.0
        if previous_belief_count is not None:
            change_ratio = abs(len(beliefs) - previous_belief_count) / max(1, previous_belief_count)
            stability = max(0.0, 1.0 - change_ratio)
            r_stability = self._w_stability * stability

        # contradiction penalty
        p_contradiction = self._w_contradiction * min(1.0, contradiction_errors * 0.2)

        # forgetting penalty (losing core beliefs is bad)
        p_forgetting = self._w_forgetting * min(1.0, core_beliefs_lost * 0.5)

        components = RewardComponents(
            task_success=r_task,
            consistency=r_consistency,
            memory_efficiency=r_efficiency,
            stability=r_stability,
            contradiction_penalty=p_contradiction,
            forgetting_penalty=p_forgetting,
        )

        signal = RewardSignal(
            components=components,
            total_reward=components.total,
            step=self._step,
        )

        self._reward_history.append(signal)

        return signal

    def get_episode_return(self) -> float:
        """Get cumulative return for current episode."""
        return sum(r.total_reward for r in self._reward_history)

    def get_average_reward(self) -> float:
        """Get average reward per step."""
        if not self._reward_history:
            return 0.0
        return self.get_episode_return() / len(self._reward_history)

    def get_reward_history(self, limit: int = 100) -> list[RewardSignal]:
        """Get recent reward signals."""
        return self._reward_history[-limit:]

    def get_component_averages(self) -> dict[str, float]:
        """Get average of each reward component."""
        if not self._reward_history:
            return {}

        n = len(self._reward_history)
        return {
            "task_success": sum(r.components.task_success for r in self._reward_history) / n,
            "consistency": sum(r.components.consistency for r in self._reward_history) / n,
            "memory_efficiency": sum(r.components.memory_efficiency for r in self._reward_history) / n,
            "stability": sum(r.components.stability for r in self._reward_history) / n,
            "contradiction_penalty": sum(r.components.contradiction_penalty for r in self._reward_history) / n,
            "forgetting_penalty": sum(r.components.forgetting_penalty for r in self._reward_history) / n,
        }

    def reset(self) -> None:
        """Reset for new episode."""
        self._reward_history.clear()
        self._step = 0


__all__ = [
    "RewardShaperAgent",
    "RewardSignal",
    "RewardComponents",
]
