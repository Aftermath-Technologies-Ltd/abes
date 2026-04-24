# Author: Bradley R. Kinnard
"""
Conflict resolution behavior module.

Provides deterministic conflict resolution for contradicting beliefs.
This module demonstrates CONSISTENCY ENFORCEMENT, not truth inference.

Key distinction:
- We do NOT determine which belief is "true"
- We enforce consistency by weakening, merging, or deferring
- Resolution decisions are based on confidence and tension, not semantics
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
import hashlib


class ResolutionAction(str, Enum):
    """Possible resolution actions for conflicting beliefs."""
    WEAKEN = "weaken"      # Reduce confidence of one or both
    MERGE = "merge"        # Combine into a hedged statement
    DEFER = "defer"        # Keep both, mark as conflicting
    REJECT = "reject"      # Reject the new belief


@dataclass
class Belief:
    """Minimal belief for conflict resolution."""
    id: UUID
    content: str
    confidence: float
    created_at_turn: int

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "content": self.content,
            "confidence": self.confidence,
            "created_at_turn": self.created_at_turn,
        }


@dataclass
class ResolutionCase:
    """A conflict resolution case with input, action, and result."""
    case_id: str
    input_beliefs: list[Belief]
    resolution_action: ResolutionAction
    resulting_beliefs: list[Belief]
    confidence_score: float
    notes: str

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "input_beliefs": [b.to_dict() for b in self.input_beliefs],
            "resolution_action": self.resolution_action.value,
            "resulting_beliefs": [b.to_dict() for b in self.resulting_beliefs],
            "confidence_score": self.confidence_score,
            "notes": self.notes,
        }


class ConflictResolver:
    """
    Deterministic conflict resolver for belief contradictions.

    Resolution strategy (in order of preference):
    1. If one belief is significantly stronger (>0.15 diff): WEAKEN the weaker
    2. If beliefs are similar strength and recent: DEFER (keep both with conflict marker)
    3. If beliefs are similar strength and one is old: WEAKEN the older
    4. If content allows: MERGE into hedged statement
    """

    CONFIDENCE_GAP_THRESHOLD = 0.15
    MERGE_SIMILARITY_THRESHOLD = 0.8

    def resolve(
        self,
        belief_a: Belief,
        belief_b: Belief,
        current_turn: int = 0,
    ) -> ResolutionCase:
        """
        Resolve a conflict between two beliefs.

        Args:
            belief_a: First belief (typically older)
            belief_b: Second belief (typically newer)
            current_turn: Current turn for age calculation

        Returns:
            ResolutionCase with action and results
        """
        conf_diff = abs(belief_a.confidence - belief_b.confidence)
        age_a = current_turn - belief_a.created_at_turn
        age_b = current_turn - belief_b.created_at_turn

        # Case 1: Clear winner by confidence
        if conf_diff > self.CONFIDENCE_GAP_THRESHOLD:
            if belief_a.confidence > belief_b.confidence:
                # Weaken B
                weakened_b = Belief(
                    id=belief_b.id,
                    content=belief_b.content,
                    confidence=belief_b.confidence * 0.5,
                    created_at_turn=belief_b.created_at_turn,
                )
                return ResolutionCase(
                    case_id=self._generate_case_id(belief_a, belief_b),
                    input_beliefs=[belief_a, belief_b],
                    resolution_action=ResolutionAction.WEAKEN,
                    resulting_beliefs=[belief_a, weakened_b],
                    confidence_score=belief_a.confidence,
                    notes=f"Weakened newer belief due to {conf_diff:.2f} confidence gap",
                )
            else:
                # Weaken A
                weakened_a = Belief(
                    id=belief_a.id,
                    content=belief_a.content,
                    confidence=belief_a.confidence * 0.5,
                    created_at_turn=belief_a.created_at_turn,
                )
                return ResolutionCase(
                    case_id=self._generate_case_id(belief_a, belief_b),
                    input_beliefs=[belief_a, belief_b],
                    resolution_action=ResolutionAction.WEAKEN,
                    resulting_beliefs=[weakened_a, belief_b],
                    confidence_score=belief_b.confidence,
                    notes=f"Weakened older belief due to {conf_diff:.2f} confidence gap",
                )

        # Case 2: Similar confidence, check age
        if age_a > age_b + 10:  # A is significantly older
            # Weaken the older one
            weakened_a = Belief(
                id=belief_a.id,
                content=belief_a.content,
                confidence=belief_a.confidence * 0.7,
                created_at_turn=belief_a.created_at_turn,
            )
            return ResolutionCase(
                case_id=self._generate_case_id(belief_a, belief_b),
                input_beliefs=[belief_a, belief_b],
                resolution_action=ResolutionAction.WEAKEN,
                resulting_beliefs=[weakened_a, belief_b],
                confidence_score=(weakened_a.confidence + belief_b.confidence) / 2,
                notes=f"Weakened older belief (age diff: {age_a - age_b} turns)",
            )

        # Case 3: Similar confidence and age - defer
        return ResolutionCase(
            case_id=self._generate_case_id(belief_a, belief_b),
            input_beliefs=[belief_a, belief_b],
            resolution_action=ResolutionAction.DEFER,
            resulting_beliefs=[belief_a, belief_b],
            confidence_score=(belief_a.confidence + belief_b.confidence) / 2,
            notes="Deferred: beliefs have similar confidence and age, marking as conflicting",
        )

    def _generate_case_id(self, a: Belief, b: Belief) -> str:
        """Generate deterministic case ID from belief contents."""
        combined = f"{a.content}|{b.content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


def generate_test_cases() -> list[ResolutionCase]:
    """
    Generate a set of test cases covering different conflict scenarios.

    Returns:
        List of resolved cases
    """
    resolver = ConflictResolver()
    cases = []

    # Case 1: Direct contradiction, equal strength
    b1 = Belief(id=uuid4(), content="The sky is blue", confidence=0.8, created_at_turn=0)
    b2 = Belief(id=uuid4(), content="The sky is gray", confidence=0.8, created_at_turn=5)
    cases.append(resolver.resolve(b1, b2, current_turn=10))

    # Case 2: Direct contradiction, clear winner
    b3 = Belief(id=uuid4(), content="I like coffee", confidence=0.9, created_at_turn=0)
    b4 = Belief(id=uuid4(), content="I hate coffee", confidence=0.6, created_at_turn=5)
    cases.append(resolver.resolve(b3, b4, current_turn=10))

    # Case 3: Location contradiction, newer stronger
    b5 = Belief(id=uuid4(), content="I live in Seattle", confidence=0.7, created_at_turn=0)
    b6 = Belief(id=uuid4(), content="I live in Portland", confidence=0.85, created_at_turn=50)
    cases.append(resolver.resolve(b5, b6, current_turn=60))

    # Case 4: Old vs new, similar confidence
    b7 = Belief(id=uuid4(), content="I am a programmer", confidence=0.8, created_at_turn=0)
    b8 = Belief(id=uuid4(), content="I am a designer", confidence=0.75, created_at_turn=100)
    cases.append(resolver.resolve(b7, b8, current_turn=110))

    # Case 5: Very recent contradictions
    b9 = Belief(id=uuid4(), content="It is warm outside", confidence=0.8, created_at_turn=95)
    b10 = Belief(id=uuid4(), content="It is cold outside", confidence=0.8, created_at_turn=100)
    cases.append(resolver.resolve(b9, b10, current_turn=100))

    return cases


__all__ = [
    "Belief",
    "ResolutionAction",
    "ResolutionCase",
    "ConflictResolver",
    "generate_test_cases",
]
