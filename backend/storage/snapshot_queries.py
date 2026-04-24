"""
Snapshot query functions for retrieving and analyzing ecology state over time.
Works with SnapshotStoreABC implementations.
"""

from typing import List, Optional
from uuid import UUID

from ..core.models.snapshot import Snapshot, SnapshotDiff
from .base import SnapshotStoreABC


async def list_snapshots(
    store: SnapshotStoreABC,
    run_id: Optional[str] = None,
    limit: int = 1000,
) -> List[Snapshot]:
    """
    List snapshots, optionally filtered by run_id.
    Returns most recent first, up to limit.
    """
    # for now, get all and sort (in-memory implementation)
    # TODO: add run_id filtering when runs are implemented
    snapshots = await store.list_snapshots(limit=limit)

    # sort by iteration descending (most recent first)
    snapshots.sort(key=lambda s: s.metadata.iteration, reverse=True)

    return snapshots[:limit]


async def get_snapshot_by_iteration(
    store: SnapshotStoreABC,
    iteration: int,
) -> Optional[Snapshot]:
    """Retrieve snapshot for a specific iteration number."""
    snapshots = await store.list_snapshots(limit=10000)  # get all

    for snapshot in snapshots:
        if snapshot.metadata.iteration == iteration:
            return snapshot

    return None


async def get_latest_snapshot(store: SnapshotStoreABC) -> Optional[Snapshot]:
    """Get the most recent snapshot by iteration number."""
    snapshots = await store.list_snapshots(limit=10000)

    if not snapshots:
        return None

    # find max iteration
    latest = max(snapshots, key=lambda s: s.metadata.iteration)
    return latest


async def get_snapshot_range(
    store: SnapshotStoreABC,
    start_iter: int,
    end_iter: int,
) -> List[Snapshot]:
    """
    Get snapshots within an iteration range (inclusive).
    Returns sorted by iteration ascending.
    """
    snapshots = await store.list_snapshots(limit=10000)

    # filter to range
    in_range = [s for s in snapshots if start_iter <= s.metadata.iteration <= end_iter]

    # sort by iteration ascending
    in_range.sort(key=lambda s: s.metadata.iteration)

    return in_range


async def compare_snapshots(
    store: SnapshotStoreABC,
    snapshot_id_a: UUID,
    snapshot_id_b: UUID,
) -> SnapshotDiff:
    """
    Compare two snapshots by ID and return the diff.
    Treats snapshot_a as 'before' and snapshot_b as 'after'.
    """
    snapshot_a = await store.get_snapshot(snapshot_id_a)
    snapshot_b = await store.get_snapshot(snapshot_id_b)

    if snapshot_a is None:
        raise ValueError(f"Snapshot {snapshot_id_a} not found")
    if snapshot_b is None:
        raise ValueError(f"Snapshot {snapshot_id_b} not found")

    # use Snapshot.diff classmethod
    return Snapshot.diff(snapshot_a, snapshot_b)


__all__ = [
    "list_snapshots",
    "get_snapshot_by_iteration",
    "get_latest_snapshot",
    "get_snapshot_range",
    "compare_snapshots",
]
