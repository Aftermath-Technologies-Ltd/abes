"""
Event models for tracking belief ecology changes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List
from uuid import UUID

from .models.belief import utcnow
from .models.snapshot import BeliefSnapshot


@dataclass
class BaseEvent:
    """Base class for all ecology events."""

    type: str
    timestamp: datetime
    iteration: int


@dataclass(init=False)
class BeliefCreatedEvent(BaseEvent):
    """New belief enters the ecology"""

    belief: BeliefSnapshot

    def __init__(self, belief: BeliefSnapshot, iteration: int):
        super().__init__(
            type="belief_created",
            timestamp=utcnow(),
            iteration=iteration,
        )
        self.belief = belief


@dataclass(init=False)
class BeliefReinforcedEvent(BaseEvent):
    """Existing belief was reinforced."""

    belief_id: UUID
    new_confidence: float

    def __init__(self, belief_id: UUID, new_confidence: float, iteration: int):
        super().__init__("belief_reinforced", utcnow(), iteration)
        self.belief_id = belief_id
        self.new_confidence = new_confidence


@dataclass(init=False)
class BeliefDecayedEvent(BaseEvent):
    """
    Belief confidence dropped due to decay.
    old_confidence must be >= new_confidence.
    """

    belief_id: UUID
    old_confidence: float
    new_confidence: float

    def __init__(
        self,
        belief_id: UUID,
        old_confidence: float,
        new_confidence: float,
        iteration: int,
    ):
        if new_confidence > old_confidence:
            raise ValueError("decay cannot increase confidence")
        super().__init__(type="belief_decayed", timestamp=utcnow(), iteration=iteration)
        self.belief_id = belief_id
        self.old_confidence = old_confidence
        self.new_confidence = new_confidence


@dataclass(init=False)
class ContradictionDetectedEvent(BaseEvent):
    """Two beliefs contradict"""

    belief_id_a: UUID
    belief_id_b: UUID
    score: float

    def __init__(
        self, belief_id_a: UUID, belief_id_b: UUID, score: float, iteration: int
    ):
        super().__init__(
            type="contradiction_detected",
            timestamp=utcnow(),
            iteration=iteration,
        )
        self.belief_id_a = belief_id_a
        self.belief_id_b = belief_id_b
        self.score = score


@dataclass(init=False)
class MutationProposedEvent(BaseEvent):
    """Mutated belief variant created from parent"""

    parent_id: UUID
    new_belief: BeliefSnapshot

    def __init__(self, parent_id: UUID, new_belief: BeliefSnapshot, iteration: int):
        self.parent_id = parent_id
        self.new_belief = new_belief
        super().__init__(
            type="mutation_proposed",
            timestamp=utcnow(),
            iteration=iteration,
        )


@dataclass(init=False)
class SafetyLimitExceededEvent(BaseEvent):
    """A safety limit was exceeded."""

    limit_type: str
    current_value: float
    max_value: float
    action_taken: str  # "blocked", "warned", "pruned"
    affected_ids: List[UUID]

    def __init__(
        self,
        limit_type: str,
        current_value: float,
        max_value: float,
        action_taken: str,
        iteration: int,
        affected_ids: List[UUID] = None,
    ):
        super().__init__(
            type="safety_limit_exceeded",
            timestamp=utcnow(),
            iteration=iteration,
        )
        self.limit_type = limit_type
        self.current_value = current_value
        self.max_value = max_value
        self.action_taken = action_taken
        self.affected_ids = affected_ids or []


# type alias for event logs
EventLog = List[BaseEvent]


__all__ = [
    "BaseEvent",
    "BeliefCreatedEvent",
    "BeliefReinforcedEvent",
    "BeliefDecayedEvent",
    "ContradictionDetectedEvent",
    "MutationProposedEvent",
    "SafetyLimitExceededEvent",
    "EventLog",
]
