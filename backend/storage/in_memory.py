"""In-memory belief store with tiered memory architecture."""

import asyncio
import logging
from typing import Dict, List, Optional
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus
from ..core.models.snapshot import BeliefSnapshot, Snapshot, SnapshotDiff
from .base import BeliefStoreABC, SnapshotStoreABC

logger = logging.getLogger(__name__)

# Configurable tier capacity caps. Override by passing a custom dict to
# InMemoryBeliefStore, or by modifying TIER_CAPS before instantiation.
TIER_CAPS: Dict[str, int] = {
    "L1": 50,    # Working memory: high-salience, checked every turn
    "L2": 200,   # Episodic memory: moderate activity
    "L3": 1000,  # Deep storage: archived, low-salience
}

# Keep legacy constants for backward compatibility
L1_MAX = TIER_CAPS["L1"]
L2_MAX = TIER_CAPS["L2"]
L3_MAX = TIER_CAPS["L3"]


def _belief_fitness(belief: Belief) -> float:
    """Compute fitness score used for tier demotion ordering.

    fitness = confidence * salience * max(use_count, 1)

    Higher fitness beliefs survive demotion pressure. Axioms receive a
    large bonus so they are never demoted from L1.
    """
    base = belief.confidence * belief.salience * max(belief.use_count, 1)
    return base + (1e6 if belief.is_axiom else 0.0)


class InMemoryBeliefStore(BeliefStoreABC):
    """Dict-based belief storage with L1/L2/L3 tiered memory."""

    def __init__(self, tier_caps: Optional[Dict[str, int]] = None):
        self._beliefs: dict[UUID, Belief] = {}
        self._lock = asyncio.Lock()
        self._tier_caps: Dict[str, int] = dict(tier_caps or TIER_CAPS)

    async def create(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id in self._beliefs:
                raise ValueError(f"belief {belief.id} exists already")
            # auto-assign tier based on axiom status and salience
            if belief.is_axiom:
                belief.memory_tier = "L1"
            elif belief.salience >= 0.8:
                l1_count = sum(1 for b in self._beliefs.values() if b.memory_tier == "L1")
                belief.memory_tier = "L1" if l1_count < self._tier_caps["L1"] else "L2"
            self._beliefs[belief.id] = belief
            return belief

    async def get(self, belief_id: UUID) -> Optional[Belief]:
        return self._beliefs.get(belief_id)

    async def list(
        self,
        status: Optional[BeliefStatus] = None,
        cluster_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
        user_id: Optional[UUID] = None,
        memory_tier: Optional[str] = None,
    ) -> List[Belief]:
        results = []
        for b in self._beliefs.values():
            # user_id is the ceiling - never cross-user
            if user_id is not None and b.user_id != user_id:
                continue
            if status and b.status != status:
                continue
            if cluster_id and b.cluster_id != cluster_id:
                continue
            if tags and not any(t in b.tags for t in tags):
                continue
            if min_confidence is not None and b.confidence < min_confidence:
                continue
            if max_confidence is not None and b.confidence > max_confidence:
                continue
            if session_id is not None and b.session_id != session_id:
                continue
            if memory_tier is not None and b.memory_tier != memory_tier:
                continue
            results.append(b)

        results.sort(key=lambda b: b.updated_at, reverse=True)
        return results[offset : offset + limit]

    async def update(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id not in self._beliefs:
                raise ValueError(f"no belief {belief.id}")
            self._beliefs[belief.id] = belief
            return belief

    async def delete(self, belief_id: UUID) -> bool:
        async with self._lock:
            if belief_id in self._beliefs:
                del self._beliefs[belief_id]
                return True
            return False

    async def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        status: Optional[BeliefStatus] = None,
    ) -> List[Belief]:
        # not implemented
        return []

    async def bulk_update(self, beliefs: List[Belief]) -> int:
        async with self._lock:
            for b in beliefs:
                if b.id in self._beliefs:
                    self._beliefs[b.id] = b
            return len(beliefs)

    # --- Tiered Memory Management ---

    async def _tier_count(self, tier: str) -> int:
        """Count beliefs in a given tier. Called within lock context."""
        return sum(1 for b in self._beliefs.values() if b.memory_tier == tier)

    async def rebalance_tiers(
        self,
        tier_caps: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """Promote or demote beliefs between tiers based on fitness.

        Fitness = confidence * salience * max(use_count, 1). Axioms always
        remain in L1. When a tier exceeds its cap, the lowest-fitness beliefs
        are demoted to the next lower tier. L3 overflow triggers deprecation
        of the lowest-fitness beliefs (they are marked deprecated, not deleted).
        Every demotion and deprecation is logged with the belief ID and reason.

        Args:
            tier_caps: Override caps for this call. Defaults to instance caps.

        Returns:
            Tier counts after rebalance.
        """
        caps = tier_caps or self._tier_caps

        async with self._lock:
            active = [
                b for b in self._beliefs.values()
                if b.status in (BeliefStatus.Active, BeliefStatus.Decaying)
            ]

            # axioms always stay L1
            axioms = [b for b in active if b.is_axiom]
            non_axioms = sorted(
                [b for b in active if not b.is_axiom],
                key=_belief_fitness,
                reverse=True,
            )

            l1_slots = max(0, caps["L1"] - len(axioms))
            l1_candidates = non_axioms[:l1_slots]
            after_l1 = non_axioms[l1_slots:]

            l2_slots = caps["L2"]
            l2_candidates = after_l1[:l2_slots]
            after_l2 = after_l1[l2_slots:]

            l3_slots = caps["L3"]
            l3_candidates = after_l2[:l3_slots]
            l3_overflow = after_l2[l3_slots:]

            # apply tier assignments and log demotions
            for b in axioms:
                if b.memory_tier != "L1":
                    logger.info(
                        "tier-rebalance: belief %s promoted to L1 (axiom)", b.id
                    )
                b.memory_tier = "L1"

            for b in l1_candidates:
                if b.memory_tier != "L1":
                    logger.info(
                        "tier-rebalance: belief %s assigned L1 (fitness=%.4f)",
                        b.id, _belief_fitness(b),
                    )
                b.memory_tier = "L1"

            for b in l2_candidates:
                if b.memory_tier == "L1":
                    logger.info(
                        "tier-rebalance: belief %s demoted L1->L2 (fitness=%.4f)",
                        b.id, _belief_fitness(b),
                    )
                b.memory_tier = "L2"

            for b in l3_candidates:
                prev = b.memory_tier
                if prev in ("L1", "L2"):
                    logger.info(
                        "tier-rebalance: belief %s demoted %s->L3 (fitness=%.4f)",
                        b.id, prev, _belief_fitness(b),
                    )
                b.memory_tier = "L3"

            # L3 overflow: deprecate lowest-fitness beliefs
            for b in l3_overflow:
                logger.info(
                    "tier-rebalance: belief %s deprecated (L3 overflow, fitness=%.4f)",
                    b.id, _belief_fitness(b),
                )
                b.deprecate(reason="L3 overflow during tier rebalance")
                b.memory_tier = "L3"

            # push deprecated/dormant to L3
            for b in self._beliefs.values():
                if b.status in (BeliefStatus.Deprecated, BeliefStatus.Dormant):
                    b.memory_tier = "L3"

        counts = {"L1": 0, "L2": 0, "L3": 0}
        for b in self._beliefs.values():
            if b.memory_tier in counts:
                counts[b.memory_tier] += 1

        logger.info(
            "tier rebalance complete: L1=%d/%d L2=%d/%d L3=%d/%d",
            counts["L1"], caps["L1"],
            counts["L2"], caps["L2"],
            counts["L3"], caps["L3"],
        )
        return counts

    async def get_tier_stats(self) -> Dict[str, object]:
        """Return tier distribution with counts, caps, and utilization percentages.

        Returns:
            Dict with total count and per-tier sub-dicts containing:
            - count: number of beliefs in tier
            - cap: configured cap for tier
            - utilization_pct: count / cap * 100, rounded to 1 decimal
        """
        counts: Dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}
        total = 0
        for b in self._beliefs.values():
            total += 1
            tier = b.memory_tier
            if tier in counts:
                counts[tier] += 1

        tiers: Dict[str, object] = {}
        for tier, count in counts.items():
            cap = self._tier_caps.get(tier, 1)
            tiers[tier] = {
                "count": count,
                "cap": cap,
                "utilization_pct": round(count / cap * 100, 1),
            }

        return {"total": total, "tiers": tiers}


class InMemorySnapshotStore(SnapshotStoreABC):
    """Dict-based snapshot storage. Time travel for dev mode."""

    def __init__(self, compress: bool = True):
        self._snapshots: dict[UUID, Snapshot | bytes] = {}
        self._compressed: dict[UUID, bool] = {}
        self._lock = asyncio.Lock()
        self.compress = compress

    async def save_snapshot(self, snapshot: Snapshot) -> Snapshot:
        # local import avoids circular reference
        from ..core.bel.snapshot_compression import compress_snapshot

        async with self._lock:
            if snapshot.id in self._snapshots:
                raise ValueError(f"snapshot {snapshot.id} exists")

            if self.compress:
                compressed = compress_snapshot(snapshot)
                self._snapshots[snapshot.id] = compressed
                self._compressed[snapshot.id] = True
            else:
                self._snapshots[snapshot.id] = snapshot
                self._compressed[snapshot.id] = False

            return snapshot

    async def get_snapshot(self, snapshot_id: UUID) -> Optional[Snapshot]:
        # local import avoids circular reference
        from ..core.bel.snapshot_compression import decompress_snapshot

        data = self._snapshots.get(snapshot_id)
        if data is None:
            return None

        if self._compressed.get(snapshot_id, False):
            return decompress_snapshot(data)

        return data

    async def get_compressed_size(self, snapshot_id: UUID) -> int:
        """Get size of compressed snapshot in bytes. Returns 0 if not found or not compressed."""
        data = self._snapshots.get(snapshot_id)
        if data is None:
            return 0

        if self._compressed.get(snapshot_id, False):
            return len(data)

        return 0

    async def list_snapshots(
        self,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Snapshot]:
        results = []
        for snapshot_id in self._snapshots.keys():
            s = await self.get_snapshot(snapshot_id)
            if s is None:
                continue

            # skip if outside iteration range
            if min_iteration is not None and s.metadata.iteration < min_iteration:
                continue
            if max_iteration is not None and s.metadata.iteration > max_iteration:
                continue
            results.append(s)

        results.sort(key=lambda s: s.metadata.iteration, reverse=True)
        return results[offset : offset + limit]

    async def list_all(self) -> List[Snapshot]:
        """Get all snapshots."""
        return await self.list_snapshots(limit=10000)

    async def get_by_iteration(self, iteration: int) -> Optional[Snapshot]:
        """Get snapshot by iteration number."""
        for snapshot_id in self._snapshots.keys():
            s = await self.get_snapshot(snapshot_id)
            if s and s.metadata.iteration == iteration:
                return s
        return None

    async def get_latest(self) -> Optional[Snapshot]:
        """Get the most recent snapshot."""
        snapshots = await self.list_snapshots(limit=1)
        return snapshots[0] if snapshots else None

    async def save(self, snapshot: Snapshot) -> Snapshot:
        """Alias for save_snapshot."""
        return await self.save_snapshot(snapshot)

    async def compare_snapshots(
        self, snapshot_id_1: UUID, snapshot_id_2: UUID
    ) -> SnapshotDiff:
        s1 = await self.get_snapshot(snapshot_id_1)
        s2 = await self.get_snapshot(snapshot_id_2)

        if not s1:
            raise ValueError(f"missing snapshot {snapshot_id_1}")
        if not s2:
            raise ValueError(f"missing snapshot {snapshot_id_2}")

        return Snapshot.diff(s1, s2)


__all__ = ["InMemoryBeliefStore", "InMemorySnapshotStore"]
