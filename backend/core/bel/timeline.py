"""
Timeline utilities for snapshot replay and key moment detection.
Powers the Time Explorer UI.
"""

from typing import Awaitable, Callable, List

from ...storage import (
    Snapshot,
    SnapshotDiff,
    SnapshotStoreABC,
    get_snapshot_range,
    list_snapshots,
)


class SnapshotTimeline:
    """Helper for replaying ecology history and finding interesting moments"""

    def __init__(self, store: SnapshotStoreABC):
        self.store = store

    async def replay(
        self,
        start_iter: int,
        end_iter: int,
        callback: Callable[[Snapshot, SnapshotDiff | None], Awaitable[None]],
    ):
        """
        Step through snapshots between start and end iteration.
        Calls your callback with each snapshot plus the diff from previous.
        First one gets None for diff.
        """
        snapshots = await get_snapshot_range(self.store, start_iter, end_iter)

        if not snapshots:
            return

        for i, snapshot in enumerate(snapshots):
            if i > 0:
                diff = Snapshot.diff(snapshots[i - 1], snapshot)
                await callback(snapshot, diff)
            else:
                await callback(snapshot, None)

    async def get_key_moments(self, threshold_tension: float = 0.6) -> List[Snapshot]:
        """
        Pull out snapshots with high tension or lots of events.
        Useful for showing major turning points in the UI.
        """
        snapshots_all = await list_snapshots(self.store, limit=10000)

        moments = []
        for snap in snapshots_all:
            # tension spike takes priority
            if snap.global_tension >= threshold_tension:
                moments.append(snap)
                continue

            # event log can also spike during major changes
            if snap.events and len(snap.events) > 5:
                moments.append(snap)

        moments.sort(key=lambda s: s.metadata.iteration)

        return moments


__all__ = ["SnapshotTimeline"]
