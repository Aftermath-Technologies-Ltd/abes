"""
Snapshot models for capturing belief ecology state at specific iterations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .belief import BeliefStatus, OriginMetadata, utcnow

__all__ = ["SnapshotMetadata", "BeliefSnapshot", "Snapshot", "SnapshotDiff"]


class SnapshotBaseModel(BaseModel):
    model_config = {
        "from_attributes": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class SnapshotMetadata(SnapshotBaseModel):
    """When and why a snapshot was taken"""

    iteration: int
    timestamp: datetime = Field(default_factory=utcnow)
    context_summary: Optional[str] = None

    @field_validator("iteration")
    @classmethod
    def validate_iteration(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"iteration cannot be negative, got {v}")
        return v


class BeliefSnapshot(SnapshotBaseModel):
    """
    Frozen belief state at snapshot time.
    Can't mutate fields after creation but origin/tags refs are still mutable.
    """

    id: UUID
    content: str
    confidence: float
    origin: OriginMetadata
    tags: List[str]
    tension: float
    cluster_id: Optional[UUID]
    status: BeliefStatus
    parent_id: Optional[UUID]
    use_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
        "frozen": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class Snapshot(SnapshotBaseModel):
    """
    Complete ecology state at one iteration. Used for time travel, replay,
    and analysis of how beliefs evolved.
    """

    id: UUID = Field(default_factory=uuid4)
    metadata: SnapshotMetadata
    beliefs: List[BeliefSnapshot]
    global_tension: float = 0.0
    cluster_metrics: Dict[str, dict] = Field(default_factory=dict)
    agent_actions: List[dict] = Field(default_factory=lambda: [])
    events: List = Field(default_factory=list)  # List[BaseEvent] from events.py
    rl_state_action: Optional[dict] = None

    # Edge relationships per spec 3.7
    contradiction_edges: List[Tuple[UUID, UUID, float]] = Field(default_factory=list)
    support_edges: List[Tuple[UUID, UUID, float]] = Field(default_factory=list)
    lineage_edges: List[Tuple[UUID, UUID]] = Field(default_factory=list)

    @field_validator("global_tension")
    @classmethod
    def validate_global_tension(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"global_tension cannot be negative, got {v}")
        return v

    @classmethod
    def diff(cls, prev: Optional["Snapshot"], current: "Snapshot") -> "SnapshotDiff":
        """
        Compute differences between two snapshots.
        If prev is None, treats all current beliefs as added.
        """
        if prev is None:
            # everything in current is new
            return SnapshotDiff(
                added=current.beliefs,
                removed=[],
                mutated=[],
                tension_delta=current.global_tension,
                belief_count_delta=len(current.beliefs),
                notable_events=[
                    f"Initial snapshot with {len(current.beliefs)} beliefs"
                ],
            )

        # build maps for efficient lookup
        prev_beliefs = {b.id: b for b in prev.beliefs}
        current_beliefs = {b.id: b for b in current.beliefs}

        # compute added/removed
        added_ids = set(current_beliefs.keys()) - set(prev_beliefs.keys())
        removed_ids = set(prev_beliefs.keys()) - set(current_beliefs.keys())

        added = [current_beliefs[bid] for bid in added_ids]
        removed = [prev_beliefs[bid] for bid in removed_ids]

        # compute mutated (same ID but changed)
        mutated: List[Tuple[BeliefSnapshot, BeliefSnapshot]] = []
        for bid in set(current_beliefs.keys()) & set(prev_beliefs.keys()):
            old_b = prev_beliefs[bid]
            new_b = current_beliefs[bid]

            # check if anything meaningful changed
            if (
                old_b.content != new_b.content
                or old_b.confidence != new_b.confidence
                or old_b.status != new_b.status
                or old_b.tension != new_b.tension
            ):
                mutated.append((old_b, new_b))

        # compute deltas
        tension_delta = current.global_tension - prev.global_tension
        belief_count_delta = len(current.beliefs) - len(prev.beliefs)

        # generate notable events
        notable_events: List[str] = []
        if len(added) > 0:
            notable_events.append(f"{len(added)} beliefs added")
        if len(removed) > 0:
            notable_events.append(f"{len(removed)} beliefs removed")
        if len(mutated) > 0:
            notable_events.append(f"{len(mutated)} beliefs mutated")
        if abs(tension_delta) > 0.1:
            direction = "increased" if tension_delta > 0 else "decreased"
            notable_events.append(
                f"Global tension {direction} by {abs(tension_delta):.2f}"
            )

        return SnapshotDiff(
            added=added,
            removed=removed,
            mutated=mutated,
            tension_delta=tension_delta,
            belief_count_delta=belief_count_delta,
            notable_events=notable_events,
        )


class SnapshotDiff(SnapshotBaseModel):
    """Tracks what changed between two snapshots"""

    added: List[BeliefSnapshot] = Field(default_factory=list)
    removed: List[BeliefSnapshot] = Field(default_factory=list)
    mutated: List[Tuple[BeliefSnapshot, BeliefSnapshot]] = Field(default_factory=list)
    tension_delta: float = 0.0
    belief_count_delta: int = 0
    notable_events: List[str] = Field(default_factory=lambda: [])
