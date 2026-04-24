# Author: Bradley R. Kinnard
"""Tests for updated ranking with salience weight."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.core.bel.ranking import rank_beliefs, DEFAULT_WEIGHTS
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "Test",
    confidence: float = 0.5,
    salience: float = 0.5,
    tension: float = 0.0,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        tension=tension,
        origin=OriginMetadata(source="test"),
    )


class TestRankingWithSalience:
    def test_default_weights_include_salience(self):
        assert "salience" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["salience"] > 0

    def test_salience_affects_ranking(self):
        b_low_sal = _make_belief(content="low salience", salience=0.1, confidence=0.5)
        b_high_sal = _make_belief(content="high salience", salience=0.9, confidence=0.5)
        ranked = rank_beliefs(
            [b_low_sal, b_high_sal],
            relevance_scores={b_low_sal.id: 0.5, b_high_sal.id: 0.5},
        )
        assert ranked[0].content == "high salience"

    def test_salience_zero_doesnt_crash(self):
        b = _make_belief(salience=0.0)
        ranked = rank_beliefs([b], relevance_scores={b.id: 0.5})
        assert len(ranked) == 1

    def test_empty_beliefs(self):
        assert rank_beliefs([], {}) == []

    def test_custom_weights_override(self):
        b1 = _make_belief(content="b1", salience=0.9, confidence=0.1)
        b2 = _make_belief(content="b2", salience=0.1, confidence=0.9)
        # weight only salience
        ranked = rank_beliefs(
            [b1, b2],
            relevance_scores={},
            weights={"salience": 1.0, "confidence": 0.0, "relevance": 0.0, "recency": 0.0, "tension": 0.0},
        )
        assert ranked[0].content == "b1"

    def test_weights_sum_behavior(self):
        """All weights at zero produces zero scores but doesn't crash."""
        b = _make_belief()
        ranked = rank_beliefs(
            [b],
            relevance_scores={b.id: 0.5},
            weights={"salience": 0.0, "confidence": 0.0, "relevance": 0.0, "recency": 0.0, "tension": 0.0},
        )
        assert len(ranked) == 1
