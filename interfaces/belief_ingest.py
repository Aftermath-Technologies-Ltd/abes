# Author: Bradley R. Kinnard
"""
Belief ingest interface for determinism verification.

Provides a minimal, deterministic belief ingestion pathway
that can be tested for byte-for-byte reproducibility.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid5, NAMESPACE_DNS
import re


# Fixed namespace for deterministic UUID generation
BELIEF_NAMESPACE = uuid5(NAMESPACE_DNS, "abes.belief.namespace")


@dataclass
class IngestResult:
    """Result of belief ingestion."""
    beliefs: list[dict]
    state_hash: str

    def to_dict(self) -> dict:
        return {
            "beliefs": self.beliefs,
            "state_hash": self.state_hash,
        }


class DeterministicBeliefIngest:
    """
    Deterministic belief ingestion for verification.

    Key properties:
    - Same input always produces same output
    - State can be hashed for comparison
    - No randomness or external dependencies
    """

    def __init__(self, seed: int = 12345):
        self._seed = seed
        self._beliefs: list[dict] = []
        self._turn = 0

    def reset(self) -> None:
        """Reset to initial state."""
        self._beliefs = []
        self._turn = 0

    def _generate_deterministic_id(self, content: str, turn: int) -> str:
        """Generate a deterministic UUID-like ID from content and turn."""
        combined = f"{self._seed}:{turn}:{content}"
        return str(uuid5(BELIEF_NAMESPACE, combined))

    def _extract_beliefs(self, text: str) -> list[str]:
        """
        Extract belief candidates from text.

        Deterministic extraction using pattern matching.
        """
        beliefs = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Check for personal statement patterns
            lower = sent.lower()
            patterns = [
                r'\bmy\s+\w+\s+(is|are)\b',
                r'\bi\s+(am|have|like|love|prefer|work|live)\b',
                r"\bi'm\b",
            ]

            for pat in patterns:
                if re.search(pat, lower):
                    beliefs.append(sent)
                    break

        return beliefs

    def ingest(self, text: str) -> IngestResult:
        """
        Ingest text and extract beliefs.

        Returns IngestResult with beliefs and state hash.
        """
        self._turn += 1

        # Extract beliefs
        extracted = self._extract_beliefs(text)

        # Add to state
        for content in extracted:
            belief = {
                "id": self._generate_deterministic_id(content, self._turn),
                "content": content,
                "confidence": 0.8,
                "turn": self._turn,
            }
            self._beliefs.append(belief)

        # Compute state hash
        state_hash = self._compute_state_hash()

        return IngestResult(
            beliefs=self._beliefs.copy(),
            state_hash=state_hash,
        )

    def _compute_state_hash(self) -> str:
        """Compute SHA-256 hash of current state."""
        # Sort beliefs for consistent ordering
        sorted_beliefs = sorted(self._beliefs, key=lambda b: b["id"])

        # Serialize to JSON (sorted keys for determinism)
        state_json = json.dumps(sorted_beliefs, sort_keys=True)

        return hashlib.sha256(state_json.encode()).hexdigest()

    def get_state_hash(self) -> str:
        """Get current state hash."""
        return self._compute_state_hash()

    def get_beliefs(self) -> list[dict]:
        """Get current beliefs."""
        return self._beliefs.copy()


__all__ = ["DeterministicBeliefIngest", "IngestResult"]
