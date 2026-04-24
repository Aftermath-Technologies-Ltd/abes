# Author: Bradley R. Kinnard
"""
BEL (Belief Ecology Loop) API routes.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from ..schemas import IterationRequest, IterationResponse, StatsResponse, HealthResponse
from ...core.deps import get_bel, get_belief_store, get_snapshot_store
from ...core.models.belief import BeliefStatus

router = APIRouter(prefix="/bel", tags=["bel"])

VERSION = "0.1.0"


@router.post("/iterate", response_model=IterationResponse)
async def run_iteration(req: IterationRequest):
    """Run a single BEL iteration."""
    bel = get_bel()
    snapshot_store = get_snapshot_store()

    start = datetime.now(timezone.utc)
    snapshot = await bel.run_iteration(context=req.context)
    duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

    # log snapshot
    await snapshot_store.save(snapshot)

    return IterationResponse(
        iteration=snapshot.metadata.iteration,
        beliefs_processed=len(snapshot.beliefs),
        snapshot_id=snapshot.id,
        actions_taken=snapshot.agent_actions,
        global_tension=snapshot.global_tension,
        duration_ms=duration,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get current system statistics."""
    from ...core.config import settings

    belief_store = get_belief_store()
    snapshot_store = get_snapshot_store()

    all_beliefs = await belief_store.list(limit=10000)
    active = [b for b in all_beliefs if b.status == BeliefStatus.Active]
    deprecated = [b for b in all_beliefs if b.status == BeliefStatus.Deprecated]

    snapshots = await snapshot_store.list_all()

    avg_conf = sum(b.confidence for b in active) / len(active) if active else 0.0
    avg_tens = sum(b.tension for b in active) / len(active) if active else 0.0

    # count unique clusters
    cluster_ids = {b.cluster_id for b in all_beliefs if b.cluster_id}

    return StatsResponse(
        total_beliefs=len(all_beliefs),
        active_beliefs=len(active),
        deprecated_beliefs=len(deprecated),
        cluster_count=len(cluster_ids),
        snapshot_count=len(snapshots),
        avg_confidence=avg_conf,
        avg_tension=avg_tens,
        storage_backend=settings.storage_backend,
        llm_provider=settings.llm_provider,
        llm_fallback_enabled=settings.llm_fallback_enabled,
        decay_profile=settings.decay_profile.value,
        embedding_model=settings.embedding_model,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    belief_store = get_belief_store()
    snapshot_store = get_snapshot_store()

    all_beliefs = await belief_store.list(limit=1)
    snapshots = await snapshot_store.list_all()

    # get actual counts
    all_beliefs_full = await belief_store.list(limit=10000)

    return HealthResponse(
        status="healthy",
        version=VERSION,
        belief_count=len(all_beliefs_full),
        snapshot_count=len(snapshots),
    )
