"""
Snapshot logging for belief ecology state capture.
Persists full ecology state for replay, analysis, and debugging.
"""

from typing import List

from ...storage import Belief, Snapshot, SnapshotMetadata
from ..events import EventLog
from ..models.snapshot import BeliefSnapshot


async def log_snapshot(
    beliefs: List[Belief],
    ranked_stack: List[Belief],  # reserved for future ranking stats
    context: str,
    iteration: int,
    agent_actions: list[dict] | None = None,
    event_log: EventLog | None = None,
    snapshot_store=None,
) -> Snapshot:
    """
    Capture and persist current ecology state as a snapshot.

    Returns the created snapshot for immediate use or inspection.
    """
    if beliefs:
        global_tension = sum(b.tension for b in beliefs) / len(beliefs)
    else:
        global_tension = 0.0

    # count beliefs per cluster
    cluster_metrics: dict[str, dict] = {}
    for b in beliefs:
        if b.cluster_id:
            cid = str(b.cluster_id)
            cluster_metrics[cid] = cluster_metrics.get(cid, {"count": 0})
            cluster_metrics[cid]["count"] = cluster_metrics[cid].get("count", 0) + 1

    belief_snapshots = [
        BeliefSnapshot(
            id=b.id,
            content=b.content,
            confidence=b.confidence,
            origin=b.origin,
            tags=b.tags,
            tension=b.tension,
            cluster_id=b.cluster_id,
            status=b.status,
            parent_id=b.parent_id,
            use_count=b.use_count,
            created_at=b.created_at,
            updated_at=b.updated_at,
        )
        for b in beliefs
    ]

    # grab events from this iteration only
    iteration_events = []
    if event_log:
        iteration_events = [e for e in event_log if e.iteration == iteration]

    snapshot = Snapshot(
        metadata=SnapshotMetadata(
            iteration=iteration,
            context_summary=context[:200] if context else None,
        ),
        beliefs=belief_snapshots,
        global_tension=global_tension,
        cluster_metrics=cluster_metrics,
        agent_actions=agent_actions or [],
        events=iteration_events,
        rl_state_action=None,  # TODO: RL integration
    )

    # persist if store provided
    if snapshot_store:
        await snapshot_store.save_snapshot(snapshot)

    return snapshot


__all__ = ["log_snapshot"]
