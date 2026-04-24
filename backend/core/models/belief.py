# Author: Bradley R. Kinnard
"""Pydantic models for belief storage in the Belief Ecology engine.
Extended with salience/energy, evidence ledger, and graph edges."""

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# Python 3.10 compat - StrEnum added in 3.11
class StrEnum(str, Enum):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _epoch() -> datetime:
    """Epoch timestamp so new beliefs aren't born on cooldown."""
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


class BeliefBaseModel(BaseModel):
    model_config = {
        "from_attributes": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class OriginMetadata(BeliefBaseModel):
    """Tracks where a belief came from and when it was last reinforced."""

    source: str
    turn_index: Optional[int] = None
    episode_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utcnow)
    last_reinforced: datetime = Field(default_factory=_epoch)


class EvidenceRef(BeliefBaseModel):
    """One piece of evidence for or against a belief. Lightweight ledger entry."""

    source_id: Optional[UUID] = None  # originating belief or external ref
    content: str  # what was said / observed
    direction: str = "supports"  # "supports" | "attacks"
    weight: float = 1.0  # how strong this evidence is (0-∞)
    timestamp: datetime = Field(default_factory=utcnow)

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("supports", "attacks"):
            raise ValueError(f"direction must be 'supports' or 'attacks', got {v!r}")
        return v


class BeliefLink(BeliefBaseModel):
    """Weighted edge between beliefs. Type is 'reinforces' or 'contradicts'."""

    target_id: UUID
    relation: str  # "reinforces" | "contradicts"
    weight: float = 0.0  # strength of the link

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, v: str) -> str:
        if v not in ("reinforces", "contradicts"):
            raise ValueError(f"relation must be 'reinforces' or 'contradicts', got {v!r}")
        return v


class BeliefStatus(StrEnum):
    Active = "active"
    Decaying = "decaying"
    Dormant = "dormant"  # low salience, not dead — can reawaken
    Mutated = "mutated"
    Deprecated = "deprecated"


class Belief(BeliefBaseModel):
    """Belief record. Tracks confidence, salience, tension, lineage, evidence ledger, and graph edges."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    confidence: float  # 0.0 to 1.0
    origin: OriginMetadata
    tags: List[str] = Field(default_factory=list)
    tension: float = 0.0  # contradiction pressure
    cluster_id: Optional[UUID] = None
    status: BeliefStatus = BeliefStatus.Active
    parent_id: Optional[UUID] = None  # tracks mutations
    use_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    session_id: Optional[str] = None  # for chat session grouping
    user_id: Optional[UUID] = None  # owner of this belief

    # --- Axiomatic Beliefs ---
    # Axioms are immutable core beliefs: no decay, no deprecation, no mutation.
    # Use for hard invariants (e.g., "harm to others is wrong").
    is_axiom: bool = False

    # --- Memory Tier ---
    # L1=working (top 50, checked every turn), L2=episodic (recent 2k), L3=deep (10k+)
    memory_tier: str = Field(default="L2")  # "L1" | "L2" | "L3"

    # --- Salience / Energy (distinct from confidence) ---
    salience: float = Field(default=1.0, ge=0.0, le=1.0)  # how much it matters RIGHT NOW
    half_life_days: float = Field(default=7.0, gt=0.0)  # salience half-life in days

    # --- Evidence Ledger ---
    evidence_for: List[EvidenceRef] = Field(default_factory=list)
    evidence_against: List[EvidenceRef] = Field(default_factory=list)

    # --- Graph Edges (persistent weighted links) ---
    links: List[BeliefLink] = Field(default_factory=list)

    # computed fields for ranking (set by BEL loop, not persisted)
    relevance: Optional[float] = None  # similarity to current context
    score: Optional[float] = None  # composite ranking score

    # NOTE: updated_at only refreshed via helper methods (increment_use, reinforce, apply_decay)
    # or dict reconstruction. Direct field assignment (belief.confidence = X) will NOT update it.
    # This is intentional - use helpers for tracked changes, or manage timestamps at repo layer.

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("tension")
    @classmethod
    def validate_tension(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"tension cannot be negative, got {v}")
        return v

    @field_validator("memory_tier")
    @classmethod
    def validate_memory_tier(cls, v: str) -> str:
        if v not in ("L1", "L2", "L3"):
            raise ValueError(f"memory_tier must be L1, L2, or L3, got {v!r}")
        return v

    @field_validator("use_count")
    @classmethod
    def validate_use_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"use_count cannot be negative, got {v}")
        return v

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Strip whitespace; reject empty content."""
        if not v or not v.strip():
            raise ValueError("content cannot be empty or whitespace-only")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def auto_update_timestamp(cls, values):
        # used when reconstructing an existing belief from dict data; sets updated_at if missing
        if isinstance(values, dict) and "id" in values and "updated_at" not in values:
            values["updated_at"] = utcnow()
        return values

    @field_validator("salience")
    @classmethod
    def validate_salience(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"salience must be between 0.0 and 1.0, got {v}")
        return v

    def increment_use(self) -> None:
        self.use_count += 1
        self.updated_at = utcnow()

    def reinforce(self) -> None:
        self.origin.last_reinforced = utcnow()
        self.updated_at = utcnow()

    def apply_decay(self, decay_factor: float) -> None:
        """Apply multiplicative confidence decay. Axioms are immune to decay."""
        if self.is_axiom:
            return  # axioms never decay
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be in (0.0, 1.0], got {decay_factor}")

        self.confidence *= decay_factor
        self.updated_at = utcnow()

        # auto status transition
        if self.confidence < 0.3 and self.status == BeliefStatus.Active:
            self.status = BeliefStatus.Decaying

    # --- Salience Dynamics ---

    def decay_salience(self, hours_elapsed: float) -> float:
        """Exponential salience decay based on half-life. Returns new salience."""
        if hours_elapsed <= 0:
            return self.salience
        half_life_hours = self.half_life_days * 24.0
        decay_factor = math.pow(0.5, hours_elapsed / half_life_hours)
        self.salience = max(0.0, min(1.0, self.salience * decay_factor))
        self.updated_at = utcnow()
        return self.salience

    def boost_salience(self, amount: float = 0.1) -> float:
        """Bump salience on use/reinforcement. Capped at 1.0."""
        self.salience = min(1.0, self.salience + amount)
        self.updated_at = utcnow()
        return self.salience

    # --- Evidence Ledger Helpers ---

    def add_evidence(self, ref: "EvidenceRef") -> None:
        """Append evidence and recompute confidence via naive Bayes-like update."""
        if ref.direction == "supports":
            self.evidence_for.append(ref)
        else:
            self.evidence_against.append(ref)
        self._bayes_update()
        self.updated_at = utcnow()

    def _bayes_update(self) -> None:
        """Recompute confidence from evidence ledger. Simple weighted ratio."""
        w_for = sum(e.weight for e in self.evidence_for) or 0.0
        w_against = sum(e.weight for e in self.evidence_against) or 0.0
        total = w_for + w_against
        if total == 0:
            return  # no evidence -> keep current confidence
        # nudge toward evidence ratio, blended with prior
        evidence_ratio = w_for / total
        # 70% evidence, 30% prior — so a single piece doesn't override everything
        # Cap at 0.95 to preserve reinforcement headroom
        self.confidence = max(0.01, min(0.95, 0.7 * evidence_ratio + 0.3 * self.confidence))

    @property
    def evidence_balance(self) -> float:
        """Net evidence weight. Positive = more support, negative = more attacks."""
        return (
            sum(e.weight for e in self.evidence_for)
            - sum(e.weight for e in self.evidence_against)
        )

    # --- Graph Edge Helpers ---

    def add_link(self, target_id: UUID, relation: str, weight: float = 1.0) -> None:
        """Add or update a weighted edge to another belief."""
        for link in self.links:
            if link.target_id == target_id and link.relation == relation:
                link.weight = weight
                self.updated_at = utcnow()
                return
        self.links.append(BeliefLink(target_id=target_id, relation=relation, weight=weight))
        self.updated_at = utcnow()

    def get_links(self, relation: Optional[str] = None) -> List["BeliefLink"]:
        """Get links, optionally filtered by relation type."""
        if relation is None:
            return self.links
        return [lnk for lnk in self.links if lnk.relation == relation]

    # --- Dormancy ---

    def hibernate(self) -> None:
        """Put belief to sleep. Axioms never hibernate."""
        if self.is_axiom:
            return
        self.status = BeliefStatus.Dormant
        self.salience = 0.0
        self.updated_at = utcnow()

    def deprecate(self, reason: str = "") -> None:
        """Mark belief as deprecated. Axioms cannot be deprecated."""
        if self.is_axiom:
            return
        self.status = BeliefStatus.Deprecated
        self.updated_at = utcnow()

    def mutate(self) -> None:
        """Mark belief as mutated. Axioms cannot be mutated."""
        if self.is_axiom:
            return
        self.status = BeliefStatus.Mutated
        self.updated_at = utcnow()

    def promote_to_axiom(self) -> None:
        """Promote this belief to axiom status. Permanent, irreversible."""
        self.is_axiom = True
        self.memory_tier = "L1"
        self.salience = 1.0
        self.status = BeliefStatus.Active
        self.updated_at = utcnow()

    def reawaken(self, salience_boost: float = 0.5) -> None:
        """Bring a dormant belief back to active duty."""
        if self.status == BeliefStatus.Dormant:
            self.status = BeliefStatus.Active
            self.salience = min(1.0, salience_boost)
            self.updated_at = utcnow()


__all__ = [
    "utcnow",
    "OriginMetadata",
    "BeliefStatus",
    "Belief",
    "EvidenceRef",
    "BeliefLink",
]
