# Author: Bradley R. Kinnard
"""
Tests for snapshot query functions (backend/storage/snapshot_queries.py).

Covers:
- list_snapshots ordering and limits
- get_snapshot_by_iteration
- get_latest_snapshot
- get_snapshot_range
- compare_snapshots
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.core.models.belief import BeliefStatus, OriginMetadata
from backend.core.models.snapshot import BeliefSnapshot, Snapshot, SnapshotMetadata
from backend.storage.in_memory import InMemorySnapshotStore
from backend.storage.snapshot_queries import (
    compare_snapshots,
    get_latest_snapshot,
    get_snapshot_by_iteration,
    get_snapshot_range,
    list_snapshots,
)


def utcnow():
    return datetime.now(timezone.utc)


def make_snapshot(iteration: int, belief_count: int = 1, global_tension: float = 0.2) -> Snapshot:
    """Helper to create test snapshots."""
    beliefs = [
        BeliefSnapshot(
            id=uuid4(),
            content=f"Belief {i} at iteration {iteration}",
            confidence=0.8,
            origin=OriginMetadata(source="test"),
            tags=["test"],
            tension=0.1,
            cluster_id=None,
            status=BeliefStatus.Active,
            parent_id=None,
            use_count=0,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        for i in range(belief_count)
    ]

    return Snapshot(
        metadata=SnapshotMetadata(iteration=iteration),
        beliefs=beliefs,
        global_tension=global_tension,
    )


@pytest.fixture
def snapshot_store():
    return InMemorySnapshotStore(compress=False)


class TestListSnapshots:
    """Tests for list_snapshots function."""

    @pytest.mark.asyncio
    async def test_empty_store(self, snapshot_store):
        result = await list_snapshots(snapshot_store)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_all_snapshots(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i, belief_count=i))

        result = await list_snapshots(snapshot_store)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_ordered_by_iteration_descending(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await list_snapshots(snapshot_store)
        iterations = [s.metadata.iteration for s in result]
        assert iterations == [5, 4, 3, 2, 1]

    @pytest.mark.asyncio
    async def test_respects_limit(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await list_snapshots(snapshot_store, limit=3)
        assert len(result) == 3
        iterations = [s.metadata.iteration for s in result]
        assert iterations == [5, 4, 3]

    @pytest.mark.asyncio
    async def test_limit_larger_than_count(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await list_snapshots(snapshot_store, limit=100)
        assert len(result) == 5


class TestGetSnapshotByIteration:
    """Tests for get_snapshot_by_iteration function."""

    @pytest.mark.asyncio
    async def test_finds_existing(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i, belief_count=i))

        result = await get_snapshot_by_iteration(snapshot_store, 3)
        assert result is not None
        assert result.metadata.iteration == 3
        assert len(result.beliefs) == 3

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_snapshot_by_iteration(snapshot_store, 99)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_store_returns_none(self, snapshot_store):
        result = await get_snapshot_by_iteration(snapshot_store, 1)
        assert result is None

    @pytest.mark.asyncio
    async def test_first_iteration(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i, belief_count=i))

        result = await get_snapshot_by_iteration(snapshot_store, 1)
        assert result is not None
        assert len(result.beliefs) == 1


class TestGetLatestSnapshot:
    """Tests for get_latest_snapshot function."""

    @pytest.mark.asyncio
    async def test_returns_highest_iteration(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_latest_snapshot(snapshot_store)
        assert result is not None
        assert result.metadata.iteration == 5

    @pytest.mark.asyncio
    async def test_empty_store_returns_none(self, snapshot_store):
        result = await get_latest_snapshot(snapshot_store)
        assert result is None

    @pytest.mark.asyncio
    async def test_single_snapshot(self, snapshot_store):
        snap = make_snapshot(42)
        await snapshot_store.save_snapshot(snap)

        result = await get_latest_snapshot(snapshot_store)
        assert result is not None
        assert result.metadata.iteration == 42


class TestGetSnapshotRange:
    """Tests for get_snapshot_range function."""

    @pytest.mark.asyncio
    async def test_full_range(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_snapshot_range(snapshot_store, 1, 5)
        assert len(result) == 5
        iterations = [s.metadata.iteration for s in result]
        assert iterations == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_partial_range(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_snapshot_range(snapshot_store, 2, 4)
        assert len(result) == 3
        iterations = [s.metadata.iteration for s in result]
        assert iterations == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_single_iteration_range(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_snapshot_range(snapshot_store, 3, 3)
        assert len(result) == 1
        assert result[0].metadata.iteration == 3

    @pytest.mark.asyncio
    async def test_range_beyond_data(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_snapshot_range(snapshot_store, 10, 20)
        assert result == []

    @pytest.mark.asyncio
    async def test_partial_overlap(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        result = await get_snapshot_range(snapshot_store, 4, 10)
        assert len(result) == 2
        iterations = [s.metadata.iteration for s in result]
        assert iterations == [4, 5]

    @pytest.mark.asyncio
    async def test_empty_store(self, snapshot_store):
        result = await get_snapshot_range(snapshot_store, 1, 10)
        assert result == []


class TestCompareSnapshots:
    """Tests for compare_snapshots function."""

    @pytest.mark.asyncio
    async def test_compare_existing(self, snapshot_store):
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i, belief_count=i))

        all_snaps = await list_snapshots(snapshot_store)
        snap_a = all_snaps[-1]  # iteration 1
        snap_b = all_snaps[0]   # iteration 5

        diff = await compare_snapshots(snapshot_store, snap_a.id, snap_b.id)

        # iteration 5 has 5 beliefs, iteration 1 has 1
        assert len(diff.added) == 5
        assert len(diff.removed) == 1
        assert diff.belief_count_delta == 4

    @pytest.mark.asyncio
    async def test_compare_missing_first(self, snapshot_store):
        await snapshot_store.save_snapshot(make_snapshot(1))
        all_snaps = await list_snapshots(snapshot_store)
        snap_b = all_snaps[0]

        with pytest.raises(ValueError, match="not found"):
            await compare_snapshots(snapshot_store, uuid4(), snap_b.id)

    @pytest.mark.asyncio
    async def test_compare_missing_second(self, snapshot_store):
        await snapshot_store.save_snapshot(make_snapshot(1))
        all_snaps = await list_snapshots(snapshot_store)
        snap_a = all_snaps[0]

        with pytest.raises(ValueError, match="not found"):
            await compare_snapshots(snapshot_store, snap_a.id, uuid4())

    @pytest.mark.asyncio
    async def test_compare_same_snapshot(self, snapshot_store):
        await snapshot_store.save_snapshot(make_snapshot(1))
        all_snaps = await list_snapshots(snapshot_store)
        snap = all_snaps[0]

        diff = await compare_snapshots(snapshot_store, snap.id, snap.id)

        assert diff.added == []
        assert diff.removed == []
        assert diff.mutated == []
        assert diff.belief_count_delta == 0


class TestSnapshotDiffContent:
    """Tests for diff detection of belief changes."""

    @pytest.mark.asyncio
    async def test_detects_confidence_change(self, snapshot_store):
        belief_id = uuid4()

        beliefs_1 = [
            BeliefSnapshot(
                id=belief_id,
                content="Test belief",
                confidence=0.8,
                origin=OriginMetadata(source="test"),
                tags=[],
                tension=0.0,
                cluster_id=None,
                status=BeliefStatus.Active,
                parent_id=None,
                use_count=0,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        ]
        snap_1 = Snapshot(
            metadata=SnapshotMetadata(iteration=1),
            beliefs=beliefs_1,
        )
        await snapshot_store.save_snapshot(snap_1)

        beliefs_2 = [
            BeliefSnapshot(
                id=belief_id,
                content="Test belief",
                confidence=0.5,
                origin=OriginMetadata(source="test"),
                tags=[],
                tension=0.0,
                cluster_id=None,
                status=BeliefStatus.Active,
                parent_id=None,
                use_count=0,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        ]
        snap_2 = Snapshot(
            metadata=SnapshotMetadata(iteration=2),
            beliefs=beliefs_2,
        )
        await snapshot_store.save_snapshot(snap_2)

        diff = await compare_snapshots(snapshot_store, snap_1.id, snap_2.id)

        assert len(diff.mutated) == 1
        old_b, new_b = diff.mutated[0]
        assert old_b.confidence == 0.8
        assert new_b.confidence == 0.5

    @pytest.mark.asyncio
    async def test_tension_delta_computed(self, snapshot_store):
        snap_1 = Snapshot(
            metadata=SnapshotMetadata(iteration=1),
            beliefs=[],
            global_tension=0.2,
        )
        snap_2 = Snapshot(
            metadata=SnapshotMetadata(iteration=2),
            beliefs=[],
            global_tension=0.7,
        )
        await snapshot_store.save_snapshot(snap_1)
        await snapshot_store.save_snapshot(snap_2)

        diff = await compare_snapshots(snapshot_store, snap_1.id, snap_2.id)

        assert abs(diff.tension_delta - 0.5) < 0.01
