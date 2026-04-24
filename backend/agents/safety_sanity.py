# Author: Bradley R. Kinnard
"""
SafetySanityAgent - enforces guardrails and can veto dangerous actions.
Prevents runaway mutation, catastrophic forgetting, and unsafe belief usage.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    LowConfidenceUsage = "low_confidence_usage"
    RunawayMutation = "runaway_mutation"
    BeliefProliferation = "belief_proliferation"
    CoreBeliefForgotten = "core_belief_forgotten"
    ClusterOverflow = "cluster_overflow"
    ContentTooLong = "content_too_long"
    DeprecationSpike = "deprecation_spike"
    SnapshotTooLarge = "snapshot_too_large"


class ActionType(str, Enum):
    Warn = "warn"
    Block = "block"
    Override = "override"


@dataclass
class SafetyViolation:
    """Record of a safety violation."""

    violation_type: ViolationType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    affected_beliefs: list[UUID] = field(default_factory=list)
    action_taken: ActionType = ActionType.Warn
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


@dataclass
class SafetyMetrics:
    """Aggregate safety metrics."""

    total_violations: int
    violations_by_type: dict[str, int]
    blocks_issued: int
    overrides_issued: int
    warnings_issued: int


class SafetySanityAgent:
    """
    Enforces guardrails on belief ecology (spec 4.1 agent #15).
    Can warn, block, or override dangerous actions.
    """

    def __init__(
        self,
        min_confidence_for_usage: float = 0.2,
        max_mutation_depth: int = settings.max_mutation_depth,
        max_active_beliefs: int = settings.max_active_beliefs,
        max_beliefs_per_cluster: int = settings.max_beliefs_per_cluster,
        max_content_length: int = settings.max_belief_content_length,
        max_snapshot_size_mb: int = settings.max_snapshot_size_mb,
        deprecation_spike_threshold: float = 0.3,  # 30% deprecated in one pass
        core_tags: Optional[list[str]] = None,
    ):
        self._min_confidence = min_confidence_for_usage
        self._max_mutation_depth = max_mutation_depth
        self._max_active = max_active_beliefs
        self._max_per_cluster = max_beliefs_per_cluster
        self._max_content_length = max_content_length
        self._max_snapshot_size_bytes = max_snapshot_size_mb * 1024 * 1024
        self._deprecation_spike_threshold = deprecation_spike_threshold
        self._core_tags = set(core_tags or ["core_value", "identity", "essential"])

        self._violations: list[SafetyViolation] = []
        self._vetoed_mutations: set[UUID] = set()
        self._vetoed_deprecations: set[UUID] = set()

    def _record_violation(
        self,
        vtype: ViolationType,
        severity: str,
        message: str,
        affected: Optional[list[UUID]] = None,
        action: ActionType = ActionType.Warn,
        metadata: Optional[dict] = None,
    ) -> SafetyViolation:
        violation = SafetyViolation(
            violation_type=vtype,
            severity=severity,
            message=message,
            affected_beliefs=affected or [],
            action_taken=action,
            metadata=metadata or {},
        )
        self._violations.append(violation)

        if action == ActionType.Block:
            logger.warning(f"SAFETY BLOCK: {message}")
        elif action == ActionType.Override:
            logger.warning(f"SAFETY OVERRIDE: {message}")
        else:
            logger.info(f"safety warning: {message}")

        return violation

    def check_low_confidence_usage(
        self, beliefs: list[Belief], selected_ids: list[UUID]
    ) -> list[SafetyViolation]:
        """
        Flag beliefs being used despite very low confidence.
        """
        violations = []
        belief_map = {b.id: b for b in beliefs}

        for bid in selected_ids:
            belief = belief_map.get(bid)
            if belief and belief.confidence < self._min_confidence:
                v = self._record_violation(
                    ViolationType.LowConfidenceUsage,
                    "medium",
                    f"belief {bid} used with confidence {belief.confidence:.2f}",
                    affected=[bid],
                    metadata={"confidence": belief.confidence},
                )
                violations.append(v)

        return violations

    def check_mutation_depth(
        self, belief: Belief, all_beliefs: list[Belief]
    ) -> Optional[SafetyViolation]:
        """
        Check if a belief has exceeded max mutation depth.
        """
        belief_map = {b.id: b for b in all_beliefs}
        belief_map[belief.id] = belief

        depth = 0
        current = belief
        seen: set[UUID] = set()

        while current.parent_id and current.parent_id not in seen:
            seen.add(current.id)
            depth += 1
            parent = belief_map.get(current.parent_id)
            if parent is None:
                break
            current = parent

        if depth >= self._max_mutation_depth:
            self._vetoed_mutations.add(belief.id)
            return self._record_violation(
                ViolationType.RunawayMutation,
                "high",
                f"belief {belief.id} at mutation depth {depth} (max: {self._max_mutation_depth})",
                affected=[belief.id],
                action=ActionType.Block,
                metadata={"depth": depth},
            )

        return None

    def check_belief_count(self, beliefs: list[Belief]) -> Optional[SafetyViolation]:
        """
        Check if active belief count exceeds limit.
        """
        active = [b for b in beliefs if b.status == BeliefStatus.Active]

        if len(active) >= self._max_active:
            return self._record_violation(
                ViolationType.BeliefProliferation,
                "high",
                f"active beliefs ({len(active)}) at or above limit ({self._max_active})",
                action=ActionType.Block,
                metadata={"count": len(active), "limit": self._max_active},
            )

        # warn at 90%
        if len(active) >= self._max_active * 0.9:
            return self._record_violation(
                ViolationType.BeliefProliferation,
                "medium",
                f"active beliefs ({len(active)}) approaching limit ({self._max_active})",
                metadata={"count": len(active), "limit": self._max_active},
            )

        return None

    def check_cluster_sizes(
        self, beliefs: list[Belief]
    ) -> list[SafetyViolation]:
        """
        Check for clusters exceeding size limits.
        """
        from collections import Counter

        cluster_counts = Counter(
            b.cluster_id for b in beliefs if b.cluster_id is not None
        )
        violations = []

        for cluster_id, count in cluster_counts.items():
            if count >= self._max_per_cluster:
                v = self._record_violation(
                    ViolationType.ClusterOverflow,
                    "medium",
                    f"cluster {cluster_id} has {count} beliefs (limit: {self._max_per_cluster})",
                    action=ActionType.Warn,
                    metadata={"cluster_id": str(cluster_id), "count": count},
                )
                violations.append(v)

        return violations

    def check_content_length(self, belief: Belief) -> Optional[SafetyViolation]:
        """
        Check if belief content exceeds max length.
        """
        if len(belief.content) > self._max_content_length:
            return self._record_violation(
                ViolationType.ContentTooLong,
                "low",
                f"belief {belief.id} content length {len(belief.content)} exceeds {self._max_content_length}",
                affected=[belief.id],
                action=ActionType.Override,
                metadata={"length": len(belief.content)},
            )
        return None

    def check_core_belief_deprecation(
        self, beliefs_to_deprecate: list[Belief]
    ) -> list[SafetyViolation]:
        """
        Prevent deprecation of beliefs tagged as core/essential.
        """
        violations = []

        for belief in beliefs_to_deprecate:
            if self._core_tags & set(belief.tags):
                self._vetoed_deprecations.add(belief.id)
                v = self._record_violation(
                    ViolationType.CoreBeliefForgotten,
                    "critical",
                    f"attempted deprecation of core belief {belief.id}",
                    affected=[belief.id],
                    action=ActionType.Block,
                    metadata={"tags": belief.tags},
                )
                violations.append(v)

        return violations

    def check_deprecation_spike(
        self, total_active: int, to_deprecate_count: int
    ) -> Optional[SafetyViolation]:
        """
        Warn if too many beliefs are being deprecated at once.
        """
        if total_active == 0:
            return None

        ratio = to_deprecate_count / total_active

        if ratio >= self._deprecation_spike_threshold:
            return self._record_violation(
                ViolationType.DeprecationSpike,
                "high",
                f"deprecating {to_deprecate_count}/{total_active} ({ratio:.1%}) beliefs in one pass",
                action=ActionType.Warn,
                metadata={"ratio": ratio, "count": to_deprecate_count},
            )

        return None

    def check_snapshot_size(self, compressed_bytes: int) -> Optional[SafetyViolation]:
        """
        Check if a snapshot exceeds max size limit.
        """
        if compressed_bytes > self._max_snapshot_size_bytes:
            size_mb = compressed_bytes / (1024 * 1024)
            max_mb = self._max_snapshot_size_bytes / (1024 * 1024)
            return self._record_violation(
                ViolationType.SnapshotTooLarge,
                "high",
                f"snapshot size {size_mb:.1f}MB exceeds limit {max_mb:.0f}MB",
                action=ActionType.Block,
                metadata={"size_bytes": compressed_bytes, "limit_bytes": self._max_snapshot_size_bytes},
            )
        return None

    def is_mutation_vetoed(self, belief_id: UUID) -> bool:
        """Check if a mutation was blocked."""
        return belief_id in self._vetoed_mutations

    def is_deprecation_vetoed(self, belief_id: UUID) -> bool:
        """Check if a deprecation was blocked."""
        return belief_id in self._vetoed_deprecations

    def truncate_content(self, content: str) -> str:
        """Truncate content to max length with suffix."""
        if len(content) <= self._max_content_length:
            return content
        truncated = content[: self._max_content_length - 12]
        return truncated + " [truncated]"

    async def run_all_checks(
        self,
        beliefs: list[Belief],
        selected_ids: Optional[list[UUID]] = None,
        pending_deprecations: Optional[list[Belief]] = None,
    ) -> list[SafetyViolation]:
        """
        Run all safety checks and return violations.
        """
        violations = []

        # belief count
        v = self.check_belief_count(beliefs)
        if v:
            violations.append(v)

        # cluster sizes
        violations.extend(self.check_cluster_sizes(beliefs))

        # content lengths
        for belief in beliefs:
            v = self.check_content_length(belief)
            if v:
                violations.append(v)

        # low confidence usage
        if selected_ids:
            violations.extend(self.check_low_confidence_usage(beliefs, selected_ids))

        # core belief deprecation
        if pending_deprecations:
            violations.extend(self.check_core_belief_deprecation(pending_deprecations))

            # deprecation spike
            active_count = sum(1 for b in beliefs if b.status == BeliefStatus.Active)
            v = self.check_deprecation_spike(active_count, len(pending_deprecations))
            if v:
                violations.append(v)

        return violations

    def get_violations(
        self, since: Optional[datetime] = None
    ) -> list[SafetyViolation]:
        """Get violations, optionally filtered by time."""
        if since is None:
            return self._violations.copy()
        return [v for v in self._violations if v.timestamp >= since]

    def get_metrics(self) -> SafetyMetrics:
        """Compute aggregate safety metrics."""
        by_type: dict[str, int] = {}
        blocks = 0
        overrides = 0
        warnings = 0

        for v in self._violations:
            by_type[v.violation_type.value] = by_type.get(v.violation_type.value, 0) + 1
            if v.action_taken == ActionType.Block:
                blocks += 1
            elif v.action_taken == ActionType.Override:
                overrides += 1
            else:
                warnings += 1

        return SafetyMetrics(
            total_violations=len(self._violations),
            violations_by_type=by_type,
            blocks_issued=blocks,
            overrides_issued=overrides,
            warnings_issued=warnings,
        )

    def clear_vetoes(self) -> None:
        """Clear veto lists (for testing or reset)."""
        self._vetoed_mutations.clear()
        self._vetoed_deprecations.clear()


__all__ = [
    "SafetySanityAgent",
    "SafetyViolation",
    "SafetyMetrics",
    "ViolationType",
    "ActionType",
]
