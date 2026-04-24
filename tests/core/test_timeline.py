# Author: Bradley R. Kinnard
"""
Tests for SnapshotTimeline (backend/core/bel/timeline.py).

Covers:
- Replay functionality with callbacks
- Key moment detection (high tension, event spikes)
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import pytest

from backend.core.models.belief import BeliefStatus, OriginMetadata
from backend.core.models.snapshot import BeliefSnapshot, Snapshot, SnapshotDiff, SnapshotMetadata
from backend.core.bel.timeline import SnapshotTimeline
from backend.storage.in_memory import InMemorySnapshotStore


def utcnow():
    return datetime.now(timezone.utc)


def make_snapshot(
    iteration: int,
    global_tension: float = 0.2,
    events: list = None,
    belief_count: int = 1,
) -> Snapshot:
    """Helper to create test snapshots."""
    beliefs = [
        BeliefSnapshot(
            id=uuid4(),
            content=f"Belief {i}",
            confidence=0.8,
            origin=OriginMetadata(source="test"),
            tags=[],
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
        events=events or [],
    )


@pytest.fixture
def snapshot_store():
    return InMemorySnapshotStore(compress=False)


@pytest.fixture
def timeline(snapshot_store):
    return SnapshotTimeline(snapshot_store)


class TestReplay:
    """Tests for the replay function."""

    @pytest.mark.asyncio
    async def test_replay_empty_range(self, timeline, snapshot_store):
        # no snapshots in store
        results = []

        async def callback(snap, diff):
            results.append((snap, diff))

        await timeline.replay(1, 10, callback)

        assert results == []

    @pytest.mark.asyncio
    async def test_replay_calls_callback_for_each(self, timeline, snapshot_store):
        # add snapshots
        for i in range(1, 6):
            await snapshot_store.save_snapshot(make_snapshot(i))

        results = []

        async def callback(snap: Snapshot, diff: Optional[SnapshotDiff]):
            results.append((snap.metadata.iteration, diff is not None))

        await timeline.replay(1, 5, callback)

        assert len(results) == 5
        # first callback has diff=None
        assert results[0] == (1, False)
        # subsequent have diff
        assert results[1] == (2, True)
        assert results[4] == (5, True)

    @pytest.mark.asyncio
    async def test_replay_partial_range(self, timeline, snapshot_store):
        for i in range(1, 11):
            await snapshot_store.save_snapshot(make_snapshot(i))

        iterations_seen = []

        async def callback(snap, diff):
            iterations_seen.append(snap.metadata.iteration)

        await timeline.replay(3, 7, callback)

        assert iterations_seen == [3, 4, 5, 6, 7]

    @pytest.mark.asyncio
    async def test_replay_diff_has_content(self, timeline, snapshot_store):
        # create snapshots with different belief counts
        await snapshot_store.save_snapshot(make_snapshot(1, belief_count=2))
        await snapshot_store.save_snapshot(make_snapshot(2, belief_count=5))

        diffs = []

        async def callback(snap, diff):
            if diff:
                diffs.append(diff)

        await timeline.replay(1, 2, callback)

        assert len(diffs) == 1
        diff = diffs[0]
        # should detect belief count changed
        assert diff.belief_count_delta == 3  # 5 - 2


class TestGetKeyMoments:
    """Tests for key moment detection."""

    @pytest.mark.asyncio
    async def test_empty_store(self, timeline, snapshot_store):
        moments = await timeline.get_key_moments()
        assert moments == []

    @pytest.mark.asyncio
    async def test_detects_high_tension(self, timeline, snapshot_store):
        # normal snapshots
        await snapshot_store.save_snapshot(make_snapshot(1, global_tension=0.2))
        await snapshot_store.save_snapshot(make_snapshot(2, global_tension=0.3))
        # high tension snapshot
        await snapshot_store.save_snapshot(make_snapshot(3, global_tension=0.8))
        # back to normal
        await snapshot_store.save_snapshot(make_snapshot(4, global_tension=0.25))

        moments = await timeline.get_key_moments(threshold_tension=0.6)

        assert len(moments) == 1
        assert moments[0].metadata.iteration == 3

    @pytest.mark.asyncio
    async def test_detects_event_spikes(self, timeline, snapshot_store):
        # normal events
        await snapshot_store.save_snapshot(make_snapshot(1, events=["a", "b"]))
        # event spike (> 5 events)
        await snapshot_store.save_snapshot(
            make_snapshot(2, events=["a", "b", "c", "d", "e", "f", "g"])
        )
        await snapshot_store.save_snapshot(make_snapshot(3, events=[]))

        moments = await timeline.get_key_moments()

        assert len(moments) == 1
        assert moments[0].metadata.iteration == 2

    @pytest.mark.asyncio
    async def test_multiple_key_moments(self, timeline, snapshot_store):
        await snapshot_store.save_snapshot(make_snapshot(1, global_tension=0.1))
        await snapshot_store.save_snapshot(make_snapshot(2, global_tension=0.7))  # key
        await snapshot_store.save_snapshot(make_snapshot(3, global_tension=0.2))
        await snapshot_store.save_snapshot(
            make_snapshot(4, events=list(range(10)))
        )  # key
        await snapshot_store.save_snapshot(make_snapshot(5, global_tension=0.8))  # key

        moments = await timeline.get_key_moments(threshold_tension=0.6)

        assert len(moments) == 3
        iterations = [m.metadata.iteration for m in moments]
        assert iterations == [2, 4, 5]  # sorted by iteration

    @pytest.mark.asyncio
    async def test_custom_tension_threshold(self, timeline, snapshot_store):
        await snapshot_store.save_snapshot(make_snapshot(1, global_tension=0.4))
        await snapshot_store.save_snapshot(make_snapshot(2, global_tension=0.5))
        await snapshot_store.save_snapshot(make_snapshot(3, global_tension=0.6))

        moments_high = await timeline.get_key_moments(threshold_tension=0.55)
        moments_low = await timeline.get_key_moments(threshold_tension=0.35)

        assert len(moments_high) == 1  # only 0.6
        assert len(moments_low) == 3   # all >= 0.35

    @pytest.mark.asyncio
    async def test_sorted_by_iteration(self, timeline, snapshot_store):
        # add out of order
        await snapshot_store.save_snapshot(make_snapshot(5, global_tension=0.9))
        await snapshot_store.save_snapshot(make_snapshot(1, global_tension=0.8))
        await snapshot_store.save_snapshot(make_snapshot(3, global_tension=0.7))

        moments = await timeline.get_key_moments(threshold_tension=0.6)

        iterations = [m.metadata.iteration for m in moments]
        assert iterations == [1, 3, 5]
