# Author: Bradley R. Kinnard
"""
Belief ranking with configurable weighted scoring.
Combines confidence, relevance, recency, tension, and salience.
"""

from typing import List
from uuid import UUID

from ...storage import Belief
from ..models.belief import utcnow

# default ranking weights (now includes salience)
DEFAULT_WEIGHTS = {
    "confidence": 0.30,
    "relevance": 0.30,
    "salience": 0.20,
    "recency": 0.10,
    "tension": 0.10,
}


def rank_beliefs(
    beliefs: List[Belief],
    relevance_scores: dict[UUID, float],
    weights: dict[str, float] | None = None,
) -> List[Belief]:
    """Rank beliefs by weighted score. Returns sorted list (highest first)."""
    if not beliefs:
        return []

    if weights is None:
        weights = DEFAULT_WEIGHTS

    now = utcnow()
    scored_beliefs: list[tuple[Belief, float]] = []

    for b in beliefs:
        relevance = relevance_scores.get(b.id, 0.0)

        age_seconds = (now - b.updated_at).total_seconds()
        recency = max(0.0, 1.0 / (1.0 + age_seconds / 3600.0))

        score = (
            weights.get("confidence", 0.0) * b.confidence
            + weights.get("relevance", 0.0) * relevance
            + weights.get("salience", 0.0) * b.salience
            + weights.get("recency", 0.0) * recency
            + weights.get("tension", 0.0) * b.tension
        )

        scored_beliefs.append((b, score))

    scored_beliefs.sort(key=lambda x: x[1], reverse=True)

    return [b for b, _ in scored_beliefs]


__all__ = ["rank_beliefs"]
