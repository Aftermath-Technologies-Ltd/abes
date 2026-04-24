# Author: Bradley R. Kinnard
"""
Append-only memory baseline.

Simulates a naive memory system that stores all beliefs without:
- Contradiction detection
- Belief decay
- Confidence evolution
- Mutation or resolution

Used to measure the value of belief ecology's active management.

LIMITATIONS:
- Beliefs accumulate without bound
- Contradictions are never resolved
- All beliefs have equal weight
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4
import hashlib
import math
import random
from collections import Counter


@dataclass
class SimpleBelief:
    """Minimal belief representation for append-only store."""
    id: UUID
    content: str
    created_at_turn: int

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "content": self.content,
            "created_at_turn": self.created_at_turn,
        }


@dataclass
class AppendOnlyState:
    """State snapshot for append-only memory."""
    turn_count: int
    beliefs: list[SimpleBelief]

    def to_dict(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "beliefs": [b.to_dict() for b in self.beliefs],
            "contradictions": 0,  # Never detected
        }


class AppendOnlyMemory:
    """
    Baseline: Simple append-only memory store.

    Every extracted fact is stored forever with no management:
    - No decay
    - No contradiction detection
    - No confidence tracking
    - No mutation or evolution

    This represents a naive approach to persistent memory.
    """

    def __init__(self, seed: int = 12345):
        self._seed = seed
        self._rng = random.Random(seed)
        self._turn_count = 0
        self._beliefs: list[SimpleBelief] = []

        # Simple patterns for fact extraction (deterministic)
        self._fact_patterns = [
            "my name is",
            "i am",
            "i have",
            "i like",
            "i love",
            "i hate",
            "i prefer",
            "i work",
            "i live",
        ]

    def reset(self) -> None:
        """Reset to initial state."""
        self._rng = random.Random(self._seed)
        self._turn_count = 0
        self._beliefs = []

    def _extract_facts(self, message: str) -> list[str]:
        """
        Simple fact extraction based on patterns.
        Deterministic - no LLM involved.
        """
        facts = []
        lower = message.lower()

        for pattern in self._fact_patterns:
            if pattern in lower:
                # Extract the sentence containing this pattern
                sentences = message.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|")
                for sent in sentences:
                    if pattern in sent.lower():
                        facts.append(sent.strip())

        # If no patterns matched but message is substantive, store it
        if not facts and len(message.split()) >= 3:
            # Use deterministic hashing to decide
            h = int(hashlib.sha256(f"{self._seed}:{message}".encode()).hexdigest()[:8], 16)
            if h % 3 == 0:  # ~33% chance to store as fact
                facts.append(message)

        return facts

    def process_turn(self, user_message: str) -> dict:
        """
        Process a single turn. Extract and store facts.
        """
        self._turn_count += 1

        # Extract facts from message
        new_facts = self._extract_facts(user_message)

        for fact in new_facts:
            belief = SimpleBelief(
                id=uuid4(),
                content=fact,
                created_at_turn=self._turn_count,
            )
            self._beliefs.append(belief)

        return {
            "turn": self._turn_count,
            "facts_extracted": len(new_facts),
            "beliefs_active": len(self._beliefs),
            "contradictions_detected": 0,  # Never detects
            "belief_entropy": self.compute_entropy(),
        }

    def get_state(self) -> AppendOnlyState:
        """Return current state snapshot."""
        return AppendOnlyState(
            turn_count=self._turn_count,
            beliefs=self._beliefs.copy(),
        )

    def get_belief_count(self) -> int:
        """Return total belief count (always growing)."""
        return len(self._beliefs)

    def get_contradiction_count(self) -> int:
        """Always 0 - no contradiction detection."""
        return 0

    def compute_entropy(self) -> float:
        """
        Compute entropy of belief distribution by content similarity.

        Groups beliefs by first 3 words and measures distribution entropy.
        Higher entropy = more diverse beliefs.
        """
        if not self._beliefs:
            return 0.0

        # Group by prefix
        prefixes = []
        for b in self._beliefs:
            words = b.content.lower().split()[:3]
            prefixes.append(" ".join(words))

        counts = Counter(prefixes)
        total = sum(counts.values())

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy


__all__ = ["AppendOnlyMemory", "AppendOnlyState", "SimpleBelief"]
