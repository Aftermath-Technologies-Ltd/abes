# Author: Bradley R. Kinnard
"""
Snapshot API routes.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    SnapshotMetadataResponse,
    SnapshotListResponse,
    SnapshotDetailResponse,
    BeliefResponse,
)
from ...core.deps import get_snapshot_store

router = APIRouter(prefix="/snapshots", tags=["snapshots"])


@router.get("", response_model=SnapshotListResponse)
async def list_snapshots(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List snapshots with pagination."""
    store = get_snapshot_store()

    # get all snapshots ordered by iteration
    all_snapshots = await store.list_all()
    all_snapshots.sort(key=lambda s: s.metadata.iteration, reverse=True)

    total = len(all_snapshots)
    start = (page - 1) * page_size
    end = start + page_size
    page_snapshots = all_snapshots[start:end]

    return SnapshotListResponse(
        snapshots=[
            SnapshotMetadataResponse(
                id=s.id,
                iteration=s.metadata.iteration,
                timestamp=s.metadata.timestamp,
                belief_count=len(s.beliefs),
                global_tension=s.global_tension,
            )
            for s in page_snapshots
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/latest", response_model=Optional[SnapshotMetadataResponse])
async def get_latest_snapshot():
    """Get the most recent snapshot."""
    store = get_snapshot_store()
    snapshot = await store.get_latest()

    if not snapshot:
        return None

    return SnapshotMetadataResponse(
        id=snapshot.id,
        iteration=snapshot.metadata.iteration,
        timestamp=snapshot.metadata.timestamp,
        belief_count=len(snapshot.beliefs),
        global_tension=snapshot.global_tension,
    )


@router.get("/by-iteration/{iteration}", response_model=SnapshotDetailResponse)
async def get_snapshot_by_iteration(iteration: int):
    """Get snapshot by iteration number."""
    store = get_snapshot_store()
    snapshot = await store.get_by_iteration(iteration)

    if not snapshot:
        raise HTTPException(404, f"No snapshot for iteration {iteration}")

    return SnapshotDetailResponse(
        id=snapshot.id,
        iteration=snapshot.metadata.iteration,
        timestamp=snapshot.metadata.timestamp,
        beliefs=[
            BeliefResponse(
                id=b.id,
                content=b.content,
                confidence=b.confidence,
                status=b.status.value,
                tension=b.tension,
                cluster_id=b.cluster_id,
                parent_id=b.parent_id,
                use_count=b.use_count,
                tags=b.tags,
                source=b.origin.source,
                created_at=b.created_at,
                updated_at=b.updated_at,
            )
            for b in snapshot.beliefs
        ],
        global_tension=snapshot.global_tension,
        agent_actions=snapshot.agent_actions,
        contradiction_edges=snapshot.contradiction_edges,
        support_edges=snapshot.support_edges,
        lineage_edges=snapshot.lineage_edges,
    )


@router.get("/{snapshot_id}", response_model=SnapshotDetailResponse)
async def get_snapshot(snapshot_id: UUID):
    """Get full snapshot detail."""
    store = get_snapshot_store()
    snapshot = await store.get_snapshot(snapshot_id)

    if not snapshot:
        raise HTTPException(404, f"Snapshot {snapshot_id} not found")

    return SnapshotDetailResponse(
        id=snapshot.id,
        iteration=snapshot.metadata.iteration,
        timestamp=snapshot.metadata.timestamp,
        beliefs=[
            BeliefResponse(
                id=b.id,
                content=b.content,
                confidence=b.confidence,
                status=b.status.value,
                tension=b.tension,
                cluster_id=b.cluster_id,
                parent_id=b.parent_id,
                use_count=b.use_count,
                tags=b.tags,
                source=b.origin.source,
                created_at=b.created_at,
                updated_at=b.updated_at,
            )
            for b in snapshot.beliefs
        ],
        global_tension=snapshot.global_tension,
        agent_actions=snapshot.agent_actions,
        contradiction_edges=snapshot.contradiction_edges,
        support_edges=snapshot.support_edges,
        lineage_edges=snapshot.lineage_edges,
    )
