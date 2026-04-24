# Author: Bradley R. Kinnard
"""
Metrics computation for belief ecology.
Tracks ecology health, agent performance, and system behavior.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus


@dataclass
class EcologyMetrics:
    """Aggregate metrics for belief ecology health."""

    # counts
    total_beliefs: int = 0
    active_beliefs: int = 0
    decaying_beliefs: int = 0
    deprecated_beliefs: int = 0
    mutated_beliefs: int = 0

    # averages
    avg_confidence: float = 0.0
    avg_tension: float = 0.0
    avg_use_count: float = 0.0

    # extremes
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    min_tension: float = 0.0
    max_tension: float = 0.0

    # clustering
    cluster_count: int = 0
    avg_cluster_size: float = 0.0
    orphan_beliefs: int = 0  # beliefs with no cluster

    # lineage
    max_mutation_depth: int = 0
    beliefs_with_parents: int = 0

    # timing
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentMetrics:
    """Metrics for a single agent's performance."""

    agent_name: str
    run_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    last_run: Optional[datetime] = None
    beliefs_processed: int = 0
    actions_taken: int = 0
    errors: int = 0


@dataclass
class IterationMetrics:
    """Metrics for a single BEL iteration."""

    iteration: int
    timestamp: datetime
    duration_ms: float

    # belief changes
    beliefs_created: int = 0
    beliefs_reinforced: int = 0
    beliefs_mutated: int = 0
    beliefs_deprecated: int = 0

    # tension
    contradictions_detected: int = 0
    tensions_updated: int = 0

    # agent activity
    agents_run: int = 0
    total_agent_time_ms: float = 0.0


class MetricsComputer:
    """Computes metrics from belief ecology state."""

    def compute_ecology_metrics(self, beliefs: list[Belief]) -> EcologyMetrics:
        """Compute aggregate ecology metrics from beliefs."""
        if not beliefs:
            return EcologyMetrics()

        metrics = EcologyMetrics(total_beliefs=len(beliefs))

        # count by status
        status_counts = {
            BeliefStatus.Active: 0,
            BeliefStatus.Decaying: 0,
            BeliefStatus.Deprecated: 0,
            BeliefStatus.Mutated: 0,
        }

        confidences = []
        tensions = []
        use_counts = []
        cluster_ids: set[UUID] = set()
        orphans = 0
        with_parents = 0

        for b in beliefs:
            status_counts[b.status] = status_counts.get(b.status, 0) + 1
            confidences.append(b.confidence)
            tensions.append(b.tension)
            use_counts.append(b.use_count)

            if b.cluster_id:
                cluster_ids.add(b.cluster_id)
            else:
                orphans += 1

            if b.parent_id:
                with_parents += 1

        metrics.active_beliefs = status_counts[BeliefStatus.Active]
        metrics.decaying_beliefs = status_counts[BeliefStatus.Decaying]
        metrics.deprecated_beliefs = status_counts[BeliefStatus.Deprecated]
        metrics.mutated_beliefs = status_counts[BeliefStatus.Mutated]

        metrics.avg_confidence = sum(confidences) / len(confidences)
        metrics.avg_tension = sum(tensions) / len(tensions)
        metrics.avg_use_count = sum(use_counts) / len(use_counts)

        metrics.min_confidence = min(confidences)
        metrics.max_confidence = max(confidences)
        metrics.min_tension = min(tensions)
        metrics.max_tension = max(tensions)

        metrics.cluster_count = len(cluster_ids)
        metrics.orphan_beliefs = orphans
        if cluster_ids:
            metrics.avg_cluster_size = (len(beliefs) - orphans) / len(cluster_ids)

        metrics.beliefs_with_parents = with_parents
        metrics.max_mutation_depth = self._compute_max_depth(beliefs)

        return metrics

    def _compute_max_depth(self, beliefs: list[Belief]) -> int:
        """Find the maximum mutation depth in the belief set."""
        belief_map = {b.id: b for b in beliefs}
        max_depth = 0

        for belief in beliefs:
            depth = 0
            current = belief
            seen: set[UUID] = set()

            while current.parent_id and current.parent_id not in seen:
                seen.add(current.id)
                depth += 1
                parent = belief_map.get(current.parent_id)
                if not parent:
                    break
                current = parent

            max_depth = max(max_depth, depth)

        return max_depth


class MetricsCollector:
    """Collects and stores metrics over time."""

    def __init__(self, max_history: int = 1000):
        self._max_history = max_history
        self._ecology_history: list[EcologyMetrics] = []
        self._iteration_history: list[IterationMetrics] = []
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._computer = MetricsComputer()

    def record_ecology(self, beliefs: list[Belief]) -> EcologyMetrics:
        """Compute and record ecology metrics."""
        metrics = self._computer.compute_ecology_metrics(beliefs)
        self._ecology_history.append(metrics)

        # trim history
        if len(self._ecology_history) > self._max_history:
            self._ecology_history = self._ecology_history[-self._max_history:]

        return metrics

    def record_iteration(self, metrics: IterationMetrics) -> None:
        """Record iteration metrics."""
        self._iteration_history.append(metrics)

        if len(self._iteration_history) > self._max_history:
            self._iteration_history = self._iteration_history[-self._max_history:]

    def record_agent_run(
        self,
        agent_name: str,
        duration_ms: float,
        beliefs_processed: int = 0,
        actions_taken: int = 0,
        error: bool = False,
    ) -> None:
        """Record an agent execution."""
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        am = self._agent_metrics[agent_name]
        am.run_count += 1
        am.total_duration_ms += duration_ms
        am.avg_duration_ms = am.total_duration_ms / am.run_count
        am.last_run = datetime.now(timezone.utc)
        am.beliefs_processed += beliefs_processed
        am.actions_taken += actions_taken
        if error:
            am.errors += 1

    def get_ecology_history(self, limit: Optional[int] = None) -> list[EcologyMetrics]:
        """Get ecology metrics history."""
        if limit:
            return self._ecology_history[-limit:]
        return self._ecology_history.copy()

    def get_iteration_history(self, limit: Optional[int] = None) -> list[IterationMetrics]:
        """Get iteration metrics history."""
        if limit:
            return self._iteration_history[-limit:]
        return self._iteration_history.copy()

    def get_agent_metrics(self, agent_name: Optional[str] = None) -> dict[str, AgentMetrics]:
        """Get agent metrics, optionally filtered by name."""
        if agent_name:
            am = self._agent_metrics.get(agent_name)
            return {agent_name: am} if am else {}
        return self._agent_metrics.copy()

    def get_latest_ecology(self) -> Optional[EcologyMetrics]:
        """Get most recent ecology metrics."""
        return self._ecology_history[-1] if self._ecology_history else None

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._ecology_history.clear()
        self._iteration_history.clear()
        self._agent_metrics.clear()


# module-level collector
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _collector
    _collector = None


__all__ = [
    "EcologyMetrics",
    "AgentMetrics",
    "IterationMetrics",
    "MetricsComputer",
    "MetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
]
