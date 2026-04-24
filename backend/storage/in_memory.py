"""In-memory belief store with tiered memory architecture."""

import asyncio
import logging
from typing import List, Optional
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus
from ..core.models.snapshot import BeliefSnapshot, Snapshot, SnapshotDiff
from .base import BeliefStoreABC, SnapshotStoreABC

logger = logging.getLogger(__name__)

# tier capacity limits
L1_MAX = 50    # working memory - checked every turn
L2_MAX = 2000  # episodic memory - checked during maintenance
L3_MAX = 50000 # deep storage - only awakened if salience spikes


class InMemoryBeliefStore(BeliefStoreABC):
    """Dict-based belief storage with L1/L2/L3 tiered memory."""

    def __init__(self):
        self._beliefs: dict[UUID, Belief] = {}
        self._lock = asyncio.Lock()

    async def create(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id in self._beliefs:
                raise ValueError(f"belief {belief.id} exists already")
            # auto-assign tier based on axiom status and salience
            if belief.is_axiom:
                belief.memory_tier = "L1"
            elif belief.salience >= 0.8:
                belief.memory_tier = "L1" if await self._tier_count("L1") < L1_MAX else "L2"
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

    async def rebalance_tiers(self) -> dict[str, int]:
        """
        Promote/demote beliefs between tiers based on salience and usage.
        L1: top-salience active beliefs (axioms always L1), capped at L1_MAX.
        L2: moderate salience, capped at L2_MAX. Overflow spills to L3.
        L3: everything else (deep storage, only awakened on salience spike).
        Returns tier counts after rebalance.
        """
        async with self._lock:
            active = [
                b for b in self._beliefs.values()
                if b.status in (BeliefStatus.Active, BeliefStatus.Decaying)
            ]

            # score = weighted combination of salience, confidence, recency
            def tier_score(b: Belief) -> float:
                recency = b.updated_at.timestamp() if b.updated_at else 0
                return (
                    b.salience * 0.5
                    + b.confidence * 0.3
                    + min(1.0, b.use_count / 20) * 0.2
                )

            # axioms always L1
            axioms = [b for b in active if b.is_axiom]
            non_axioms = [b for b in active if not b.is_axiom]
            non_axioms.sort(key=tier_score, reverse=True)

            l1_slots = max(0, L1_MAX - len(axioms))
            l1_candidates = non_axioms[:l1_slots]
            remaining = non_axioms[l1_slots:]

            l2_candidates = remaining[:L2_MAX]
            l3_candidates = remaining[L2_MAX:]

            for b in axioms:
                b.memory_tier = "L1"
            for b in l1_candidates:
                b.memory_tier = "L1"
            for b in l2_candidates:
                b.memory_tier = "L2"
            for b in l3_candidates:
                b.memory_tier = "L3"

            # also push deprecated/dormant to L3
            for b in self._beliefs.values():
                if b.status in (BeliefStatus.Deprecated, BeliefStatus.Dormant):
                    b.memory_tier = "L3"

        counts = {"L1": 0, "L2": 0, "L3": 0}
        for b in self._beliefs.values():
            if b.memory_tier in counts:
                counts[b.memory_tier] += 1

        logger.info("tier rebalance: L1=%d L2=%d L3=%d", counts["L1"], counts["L2"], counts["L3"])
        return counts

    async def get_tier_stats(self) -> dict:
        """Return tier distribution stats."""
        counts = {"L1": 0, "L2": 0, "L3": 0, "total": 0}
        for b in self._beliefs.values():
            counts["total"] += 1
            tier = b.memory_tier
            if tier in counts:
                counts[tier] += 1
        return counts


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
