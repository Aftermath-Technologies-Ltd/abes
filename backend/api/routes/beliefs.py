# Author: Bradley R. Kinnard
"""
Belief API routes.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    BeliefCreate,
    BeliefUpdate,
    BeliefResponse,
    BeliefListResponse,
    BeliefLinkResponse,
    BeliefEcologyResponse,
)
from ...core.deps import get_belief_store
from ...core.models.belief import Belief, BeliefStatus, OriginMetadata

router = APIRouter(prefix="/beliefs", tags=["beliefs"])


def _belief_to_response(b: Belief) -> BeliefResponse:
    """Convert internal Belief to API response."""
    return BeliefResponse(
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


@router.get("", response_model=BeliefListResponse)
async def list_beliefs(
    status: Optional[str] = Query(None, description="Filter by status"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List beliefs with optional filtering and pagination."""
    store = get_belief_store()

    # parse status filter
    status_filter = None
    if status:
        try:
            status_filter = BeliefStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    # get beliefs
    beliefs = await store.list(
        status=status_filter,
        limit=page_size,
        offset=(page - 1) * page_size,
    )

    # filter by tag if specified
    if tag:
        beliefs = [b for b in beliefs if tag in b.tags]

    # get total count
    all_beliefs = await store.list(status=status_filter, limit=10000)
    total = len(all_beliefs)

    return BeliefListResponse(
        beliefs=[_belief_to_response(b) for b in beliefs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{belief_id}", response_model=BeliefResponse)
async def get_belief(belief_id: UUID):
    """Get a single belief by ID."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    return _belief_to_response(belief)


@router.get("/{belief_id}/ecology", response_model=BeliefEcologyResponse)
async def get_belief_ecology(belief_id: UUID):
    """Full internal ecology state — salience, evidence, links, dormancy."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    return BeliefEcologyResponse(
        id=belief.id,
        content=belief.content,
        confidence=belief.confidence,
        status=belief.status.value,
        tension=belief.tension,
        salience=belief.salience,
        half_life_days=belief.half_life_days,
        evidence_for_count=len(belief.evidence_for),
        evidence_against_count=len(belief.evidence_against),
        evidence_balance=belief.evidence_balance,
        links=[
            BeliefLinkResponse(
                target_id=link.target_id,
                relation=link.relation,
                weight=link.weight,
            )
            for link in belief.links
        ],
        link_count=len(belief.links),
        created_at=belief.created_at,
        updated_at=belief.updated_at,
    )


@router.post("", response_model=BeliefResponse, status_code=201)
async def create_belief(req: BeliefCreate):
    """Create a new belief."""
    store = get_belief_store()

    belief = Belief(
        content=req.content,
        confidence=req.confidence,
        origin=OriginMetadata(
            source=req.source,
        ),
        tags=req.tags,
    )

    await store.create(belief)
    return _belief_to_response(belief)


@router.patch("/{belief_id}", response_model=BeliefResponse)
async def update_belief(belief_id: UUID, req: BeliefUpdate):
    """Update an existing belief."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    if req.confidence is not None:
        belief.confidence = req.confidence

    if req.tags is not None:
        belief.tags = req.tags

    if req.status is not None:
        try:
            belief.status = BeliefStatus(req.status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {req.status}")

    belief.updated_at = datetime.now(timezone.utc)
    await store.update(belief)

    return _belief_to_response(belief)


@router.delete("/{belief_id}", status_code=204)
async def delete_belief(belief_id: UUID):
    """Delete a belief (marks as deprecated)."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    belief.status = BeliefStatus.Deprecated
    belief.updated_at = datetime.now(timezone.utc)
    await store.update(belief)


@router.post("/{belief_id}/reinforce", response_model=BeliefResponse)
async def reinforce_belief(
    belief_id: UUID,
    boost: float = Query(0.1, ge=0.0, le=0.5),
):
    """Reinforce a belief, boosting its confidence."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    belief.confidence = min(1.0, belief.confidence + boost)
    belief.origin.last_reinforced = datetime.now(timezone.utc)
    belief.use_count += 1
    belief.updated_at = datetime.now(timezone.utc)

    await store.update(belief)
    return _belief_to_response(belief)


@router.post("/clear")
async def clear_all_beliefs():
    """Clear all beliefs from the store. Use with caution."""
    store = get_belief_store()

    # Get all beliefs and delete them
    all_beliefs = await store.list(limit=10000)
    count = 0
    for belief in all_beliefs:
        await store.delete(belief.id)
        count += 1

    return {"cleared": count, "message": f"Deleted {count} beliefs"}


@router.post("/axioms", response_model=BeliefResponse)
async def create_axiom(req: BeliefCreate):
    """Create an immutable axiom belief. Axioms cannot be decayed, deprecated, or mutated."""
    store = get_belief_store()
    belief = Belief(
        content=req.content,
        confidence=1.0,
        origin=OriginMetadata(source="axiom"),
        tags=req.tags or ["axiom", "core_value"],
        is_axiom=True,
        memory_tier="L1",
        salience=1.0,
    )
    created = await store.create(belief)
    return _belief_to_response(created)


@router.post("/{belief_id}/promote-axiom", response_model=BeliefResponse)
async def promote_to_axiom(belief_id: UUID):
    """Promote an existing belief to axiom status. Irreversible."""
    store = get_belief_store()
    belief = await store.get(belief_id)
    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")
    belief.promote_to_axiom()
    await store.update(belief)
    return _belief_to_response(belief)


@router.post("/simulate-time")
async def simulate_time(hours: float = Query(12.0, ge=0.1, le=720.0)):
    """Shift belief timestamps backward to simulate elapsed time for demo decay."""
    store = get_belief_store()
    beliefs = await store.list(limit=10000)
    shift = timedelta(hours=hours)

    adjusted = 0
    for belief in beliefs:
        belief.created_at -= shift
        belief.updated_at -= shift
        belief.origin.timestamp -= shift
        belief.origin.last_reinforced -= shift
        await store.update(belief)
        adjusted += 1

    return {
        "adjusted": adjusted,
        "hours": hours,
    }
