# Author: Bradley R. Kinnard
"""
API Request/Response schemas using Pydantic.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============ Belief Schemas ============

class BeliefCreate(BaseModel):
    """Request to create a new belief."""
    content: str = Field(..., min_length=1, max_length=2000)
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    source: str = Field("api", max_length=100)
    tags: list[str] = Field(default_factory=list)


class BeliefUpdate(BaseModel):
    """Request to update an existing belief."""
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[list[str]] = None
    status: Optional[str] = None


class BeliefResponse(BaseModel):
    """Belief in API responses."""
    id: UUID
    content: str
    confidence: float
    status: str
    tension: float
    cluster_id: Optional[UUID]
    parent_id: Optional[UUID]
    use_count: int
    tags: list[str]
    source: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BeliefListResponse(BaseModel):
    """Paginated belief list."""
    beliefs: list[BeliefResponse]
    total: int
    page: int
    page_size: int


class BeliefLinkResponse(BaseModel):
    """Single graph edge between beliefs."""
    target_id: UUID
    relation: str
    weight: float


class BeliefEcologyResponse(BaseModel):
    """Full internal ecology state — exposes fields hidden from BeliefResponse."""
    id: UUID
    content: str
    confidence: float
    status: str
    tension: float
    salience: float
    half_life_days: float
    evidence_for_count: int
    evidence_against_count: int
    evidence_balance: float
    links: list[BeliefLinkResponse]
    link_count: int
    created_at: datetime
    updated_at: datetime


# ============ Snapshot Schemas ============

class SnapshotMetadataResponse(BaseModel):
    """Snapshot metadata in responses."""
    id: UUID
    iteration: int
    timestamp: datetime
    belief_count: int
    global_tension: float


class SnapshotListResponse(BaseModel):
    """Paginated snapshot list."""
    snapshots: list[SnapshotMetadataResponse]
    total: int
    page: int
    page_size: int


class SnapshotDetailResponse(BaseModel):
    """Full snapshot detail."""
    id: UUID
    iteration: int
    timestamp: datetime
    beliefs: list[BeliefResponse]
    global_tension: float
    agent_actions: list[dict]
    contradiction_edges: list[tuple]
    support_edges: list[tuple]
    lineage_edges: list[tuple]


# ============ Agent Schemas ============

class AgentStatus(BaseModel):
    """Agent status information."""
    name: str
    phase: str
    enabled: bool
    last_run: Optional[datetime] = None
    run_count: int = 0


class AgentListResponse(BaseModel):
    """List of agents with status."""
    agents: list[AgentStatus]


class AgentToggle(BaseModel):
    """Toggle agent enabled state."""
    enabled: bool


# ============ BEL Iteration Schemas ============

class IterationRequest(BaseModel):
    """Request to run a BEL iteration."""
    context: str = Field("", max_length=10000)


class IterationResponse(BaseModel):
    """Response from running a BEL iteration."""
    iteration: int
    beliefs_processed: int
    snapshot_id: UUID
    actions_taken: list[dict]
    global_tension: float
    duration_ms: float


# ============ Cluster Schemas ============

class ClusterResponse(BaseModel):
    """Cluster information."""
    id: UUID
    size: int
    created_at: datetime
    updated_at: datetime


class ClusterListResponse(BaseModel):
    """List of clusters."""
    clusters: list[ClusterResponse]
    total: int


# ============ Health/Stats Schemas ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    belief_count: int
    snapshot_count: int


class StatsResponse(BaseModel):
    """System statistics."""
    total_beliefs: int
    active_beliefs: int
    deprecated_beliefs: int
    cluster_count: int
    snapshot_count: int
    avg_confidence: float
    avg_tension: float
    # Configuration info
    storage_backend: str = "memory"
    llm_provider: str = "ollama"
    llm_fallback_enabled: bool = True
    decay_profile: str = "moderate"
    embedding_model: str = "all-MiniLM-L6-v2"


__all__ = [
    "BeliefCreate",
    "BeliefUpdate",
    "BeliefResponse",
    "BeliefListResponse",
    "BeliefLinkResponse",
    "BeliefEcologyResponse",
    "SnapshotMetadataResponse",
    "SnapshotListResponse",
    "SnapshotDetailResponse",
    "AgentStatus",
    "AgentListResponse",
    "AgentToggle",
    "IterationRequest",
    "IterationResponse",
    "ClusterResponse",
    "ClusterListResponse",
    "HealthResponse",
    "StatsResponse",
]
