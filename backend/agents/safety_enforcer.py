# Author: Bradley R. Kinnard
"""
Safety limit enforcer that wraps storage operations.
Ensures all hard caps from spec 3.8 are enforced.
"""

import logging
from typing import Optional
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus
from .safety_sanity import SafetySanityAgent, ActionType

logger = logging.getLogger(__name__)


class SafetyLimitError(Exception):
    """Raised when a safety limit is exceeded and action is blocked."""

    def __init__(self, limit_type: str, message: str, current: float, max_val: float):
        self.limit_type = limit_type
        self.current = current
        self.max_value = max_val
        super().__init__(message)


class SafetyLimitEnforcer:
    """
    Enforces hard safety limits per spec 3.8.
    Used as middleware between API/BEL and storage.
    """

    def __init__(self, safety_agent: Optional[SafetySanityAgent] = None):
        self._agent = safety_agent or SafetySanityAgent()
        self._max_content_length = settings.max_belief_content_length
        self._max_active_beliefs = settings.max_active_beliefs
        self._max_beliefs_per_cluster = settings.max_beliefs_per_cluster
        self._max_mutation_depth = settings.max_mutation_depth
        self._max_contradiction_pairs = settings.max_contradiction_pairs_per_iteration
        self._max_snapshot_size_mb = settings.max_snapshot_size_mb

    def validate_belief_content(self, content: str) -> str:
        """
        Validate and potentially truncate belief content.
        Raises SafetyLimitError if content is empty.
        """
        if not content or not content.strip():
            raise SafetyLimitError(
                "content_empty",
                "Belief content cannot be empty",
                0,
                self._max_content_length,
            )

        if len(content) > self._max_content_length:
            logger.warning(
                f"truncating belief content from {len(content)} to {self._max_content_length}"
            )
            return self._agent.truncate_content(content)

        return content

    def validate_belief_creation(
        self, belief: Belief, current_active_count: int
    ) -> None:
        """
        Validate that creating a belief doesn't exceed limits.
        """
        # content length
        if len(belief.content) > self._max_content_length:
            belief.content = self._agent.truncate_content(belief.content)

        # active count
        if current_active_count >= self._max_active_beliefs:
            raise SafetyLimitError(
                "max_active_beliefs",
                f"Cannot create belief: active count {current_active_count} at limit {self._max_active_beliefs}",
                current_active_count,
                self._max_active_beliefs,
            )

    def validate_mutation(
        self, parent: Belief, all_beliefs: list[Belief]
    ) -> None:
        """
        Validate that a mutation doesn't exceed depth limits.
        """
        v = self._agent.check_mutation_depth(parent, all_beliefs)
        if v and v.action_taken == ActionType.Block:
            raise SafetyLimitError(
                "max_mutation_depth",
                f"Cannot mutate: depth exceeds {self._max_mutation_depth}",
                v.metadata.get("depth", 0),
                self._max_mutation_depth,
            )

    def validate_cluster_assignment(
        self, cluster_id: UUID, current_count: int
    ) -> None:
        """
        Validate cluster size before assignment.
        """
        if current_count >= self._max_beliefs_per_cluster:
            raise SafetyLimitError(
                "max_beliefs_per_cluster",
                f"Cluster {cluster_id} at limit {self._max_beliefs_per_cluster}",
                current_count,
                self._max_beliefs_per_cluster,
            )

    def limit_contradiction_pairs(self, pairs: list) -> list:
        """
        Limit contradiction pairs to prevent O(n²) explosion.
        """
        if len(pairs) <= self._max_contradiction_pairs:
            return pairs

        logger.warning(
            f"limiting contradiction pairs from {len(pairs)} to {self._max_contradiction_pairs}"
        )
        return pairs[: self._max_contradiction_pairs]

    def validate_snapshot_size(self, compressed_bytes: int) -> None:
        """
        Validate snapshot size doesn't exceed limit.
        """
        max_bytes = self._max_snapshot_size_mb * 1024 * 1024
        if compressed_bytes > max_bytes:
            raise SafetyLimitError(
                "max_snapshot_size",
                f"Snapshot size {compressed_bytes / 1024 / 1024:.1f}MB exceeds {self._max_snapshot_size_mb}MB",
                compressed_bytes,
                max_bytes,
            )

    def get_safety_agent(self) -> SafetySanityAgent:
        """Get the underlying safety agent."""
        return self._agent


# singleton instance
_enforcer: Optional[SafetyLimitEnforcer] = None


def get_safety_enforcer() -> SafetyLimitEnforcer:
    """Get the global safety limit enforcer."""
    global _enforcer
    if _enforcer is None:
        _enforcer = SafetyLimitEnforcer()
    return _enforcer


def reset_safety_enforcer() -> None:
    """Reset the global enforcer (for testing)."""
    global _enforcer
    _enforcer = None


__all__ = [
    "SafetyLimitEnforcer",
    "SafetyLimitError",
    "get_safety_enforcer",
    "reset_safety_enforcer",
]
