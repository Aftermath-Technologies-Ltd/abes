# Author: Bradley R. Kinnard
"""
Plain LLM runner baseline.

Simulates an LLM with NO persistent memory. Each turn is independent.
Used as a control to measure the value of belief persistence.

LIMITATIONS:
- No actual LLM calls (deterministic simulation)
- Returns canned responses based on input patterns
- Does not track conversation history between turns
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import random


@dataclass
class PlainLLMState:
    """State snapshot for a plain LLM (essentially empty)."""
    turn_count: int = 0
    # No beliefs, no memory, just a counter

    def to_dict(self) -> dict:
        return {"turn_count": self.turn_count, "beliefs": [], "contradictions": 0}


class PlainLLMRunner:
    """
    Baseline: LLM with no persistent memory.

    Each response is generated independently with no context carryover.
    This represents the worst-case scenario for belief consistency.
    """

    def __init__(self, seed: int = 12345):
        self._seed = seed
        self._rng = random.Random(seed)
        self._turn_count = 0

    def reset(self) -> None:
        """Reset to initial state."""
        self._rng = random.Random(self._seed)
        self._turn_count = 0

    def process_turn(self, user_message: str) -> dict:
        """
        Process a single turn. Returns metrics for this turn.

        Since there's no memory, each turn is independent.
        """
        self._turn_count += 1

        # Simulate response generation (deterministic based on seed and turn)
        response_hash = hashlib.sha256(
            f"{self._seed}:{self._turn_count}:{user_message}".encode()
        ).hexdigest()[:16]

        return {
            "turn": self._turn_count,
            "response_id": response_hash,
            "beliefs_active": 0,  # No beliefs tracked
            "contradictions_detected": 0,  # Cannot detect without memory
            "belief_entropy": 0.0,  # No distribution to measure
        }

    def get_state(self) -> PlainLLMState:
        """Return current state snapshot."""
        return PlainLLMState(turn_count=self._turn_count)

    def get_belief_count(self) -> int:
        """Always 0 - no persistent beliefs."""
        return 0

    def get_contradiction_count(self) -> int:
        """Always 0 - cannot detect contradictions without memory."""
        return 0

    def compute_entropy(self) -> float:
        """Always 0 - no belief distribution."""
        return 0.0


__all__ = ["PlainLLMRunner", "PlainLLMState"]
