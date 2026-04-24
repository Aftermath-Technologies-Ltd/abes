# Author: Bradley R. Kinnard
"""
ConsistencyCheckerAgent - probes the system for internal consistency.
Periodically asks previously-asked questions and checks for drift.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyProbe:
    """A recorded probe for consistency checking."""

    query: str
    query_hash: str
    original_response: str
    original_beliefs: list[UUID]
    timestamp: datetime
    tags: list[str] = field(default_factory=list)


@dataclass
class ConsistencyResult:
    """Result of a consistency check."""

    probe: ConsistencyProbe
    current_response: str
    current_beliefs: list[UUID]
    similarity_score: float
    belief_overlap: float
    is_consistent: bool
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConsistencyMetrics:
    """Aggregated consistency metrics."""

    total_probes: int
    checks_performed: int
    consistent_count: int
    inconsistent_count: int
    avg_similarity: float
    avg_belief_overlap: float
    consistency_rate: float

    @classmethod
    def compute(cls, results: list[ConsistencyResult]) -> "ConsistencyMetrics":
        if not results:
            return cls(
                total_probes=0,
                checks_performed=0,
                consistent_count=0,
                inconsistent_count=0,
                avg_similarity=0.0,
                avg_belief_overlap=0.0,
                consistency_rate=0.0,
            )

        consistent = sum(1 for r in results if r.is_consistent)
        return cls(
            total_probes=len(results),
            checks_performed=len(results),
            consistent_count=consistent,
            inconsistent_count=len(results) - consistent,
            avg_similarity=sum(r.similarity_score for r in results) / len(results),
            avg_belief_overlap=sum(r.belief_overlap for r in results) / len(results),
            consistency_rate=consistent / len(results),
        )


def _hash_query(query: str) -> str:
    """Stable hash for query deduplication."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _text_similarity(text_a: str, text_b: str) -> float:
    """
    Simple token-overlap similarity (Jaccard).
    For production, use embedding similarity.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _belief_overlap(ids_a: list[UUID], ids_b: list[UUID]) -> float:
    """Jaccard overlap of belief ID lists."""
    set_a = set(ids_a)
    set_b = set(ids_b)

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# type alias for response generator
ResponseGenerator = Callable[[str, list[Belief]], str]


class ConsistencyCheckerAgent:
    """
    Checks consistency by replaying previous queries (spec 4.1 agent #13).
    Tracks drift over time and emits metrics.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        belief_overlap_threshold: float = 0.5,
        max_probes: int = 1000,
    ):
        self._similarity_threshold = similarity_threshold
        self._belief_overlap_threshold = belief_overlap_threshold
        self._max_probes = max_probes

        self._probes: dict[str, ConsistencyProbe] = {}  # keyed by query_hash
        self._results: list[ConsistencyResult] = []

    def record_probe(
        self,
        query: str,
        response: str,
        belief_ids: list[UUID],
        tags: Optional[list[str]] = None,
    ) -> ConsistencyProbe:
        """
        Record a query-response pair for future consistency checking.
        """
        query_hash = _hash_query(query)

        # skip if already recorded (dedup)
        if query_hash in self._probes:
            return self._probes[query_hash]

        probe = ConsistencyProbe(
            query=query,
            query_hash=query_hash,
            original_response=response,
            original_beliefs=belief_ids.copy(),
            timestamp=datetime.now(timezone.utc),
            tags=tags or [],
        )

        # enforce max probes
        if len(self._probes) >= self._max_probes:
            # remove oldest
            oldest_hash = min(self._probes, key=lambda h: self._probes[h].timestamp)
            del self._probes[oldest_hash]

        self._probes[query_hash] = probe
        return probe

    def get_probe(self, query: str) -> Optional[ConsistencyProbe]:
        """Retrieve a probe by query text."""
        query_hash = _hash_query(query)
        return self._probes.get(query_hash)

    def list_probes(self, tag: Optional[str] = None) -> list[ConsistencyProbe]:
        """List all probes, optionally filtered by tag."""
        probes = list(self._probes.values())
        if tag:
            probes = [p for p in probes if tag in p.tags]
        return probes

    def check_consistency(
        self,
        probe: ConsistencyProbe,
        current_response: str,
        current_belief_ids: list[UUID],
    ) -> ConsistencyResult:
        """
        Compare current response to original probe.
        """
        sim = _text_similarity(probe.original_response, current_response)
        overlap = _belief_overlap(probe.original_beliefs, current_belief_ids)

        is_consistent = (
            sim >= self._similarity_threshold
            and overlap >= self._belief_overlap_threshold
        )

        result = ConsistencyResult(
            probe=probe,
            current_response=current_response,
            current_beliefs=current_belief_ids.copy(),
            similarity_score=sim,
            belief_overlap=overlap,
            is_consistent=is_consistent,
        )

        self._results.append(result)

        if not is_consistent:
            logger.warning(
                f"inconsistency detected: query='{probe.query[:50]}...', "
                f"sim={sim:.2f}, overlap={overlap:.2f}"
            )

        return result

    async def run_checks(
        self,
        beliefs: list[Belief],
        response_generator: ResponseGenerator,
        sample_size: Optional[int] = None,
    ) -> list[ConsistencyResult]:
        """
        Run consistency checks on a sample of probes.

        Args:
            beliefs: Current active beliefs
            response_generator: Function that takes (query, beliefs) -> response
            sample_size: Max probes to check (None = all)

        Returns:
            List of consistency results
        """
        probes = list(self._probes.values())

        if sample_size and len(probes) > sample_size:
            import random
            probes = random.sample(probes, sample_size)

        results = []
        for probe in probes:
            # generate current response
            current_response = response_generator(probe.query, beliefs)

            # get current belief IDs (in real impl, would track which beliefs were used)
            current_ids = [b.id for b in beliefs[:len(probe.original_beliefs)]]

            result = self.check_consistency(probe, current_response, current_ids)
            results.append(result)

        return results

    def get_metrics(self) -> ConsistencyMetrics:
        """Compute aggregate consistency metrics."""
        return ConsistencyMetrics.compute(self._results)

    def get_recent_results(self, limit: int = 50) -> list[ConsistencyResult]:
        """Get most recent check results."""
        return self._results[-limit:]

    def get_inconsistencies(self) -> list[ConsistencyResult]:
        """Get all inconsistent results."""
        return [r for r in self._results if not r.is_consistent]

    def clear_results(self) -> None:
        """Clear result history."""
        self._results.clear()


__all__ = [
    "ConsistencyCheckerAgent",
    "ConsistencyProbe",
    "ConsistencyResult",
    "ConsistencyMetrics",
]
