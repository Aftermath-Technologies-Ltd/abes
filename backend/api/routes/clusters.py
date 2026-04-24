# Author: Bradley R. Kinnard
"""
Cluster API routes.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from ..schemas import ClusterResponse, ClusterListResponse
from ...core.deps import get_cluster_manager

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get("", response_model=ClusterListResponse)
async def list_clusters():
    """List all clusters."""
    manager = get_cluster_manager()
    clusters = manager.get_all_clusters()

    return ClusterListResponse(
        clusters=[
            ClusterResponse(
                id=c.id,
                size=c.size,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in clusters
        ],
        total=len(clusters),
    )


@router.get("/stats")
async def get_cluster_stats():
    """Get clustering statistics."""
    manager = get_cluster_manager()
    return manager.get_stats()


@router.post("/maintenance")
async def run_maintenance():
    """Trigger cluster maintenance (merge/split/cleanup)."""
    manager = get_cluster_manager()
    stats = manager.run_maintenance(force=True)
    return stats


@router.get("/{cluster_id}", response_model=ClusterResponse)
async def get_cluster(cluster_id: UUID):
    """Get a single cluster."""
    manager = get_cluster_manager()
    cluster = manager.get_cluster(cluster_id)

    if not cluster:
        raise HTTPException(404, f"Cluster {cluster_id} not found")

    return ClusterResponse(
        id=cluster.id,
        size=cluster.size,
        created_at=cluster.created_at,
        updated_at=cluster.updated_at,
    )


@router.get("/{cluster_id}/beliefs", response_model=list[UUID])
async def get_cluster_beliefs(cluster_id: UUID):
    """Get belief IDs in a cluster."""
    manager = get_cluster_manager()
    cluster = manager.get_cluster(cluster_id)

    if not cluster:
        raise HTTPException(404, f"Cluster {cluster_id} not found")

    return manager.get_cluster_members(cluster_id)
