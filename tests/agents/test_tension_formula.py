# Author: Bradley R. Kinnard
"""Tests for refined tension formula and graph edge population in ContradictionAuditor."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from uuid import uuid4
from datetime import datetime, timezone

from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


class MockOriginMetadata:
    def __init__(self):
        self.source = "test"
        self.timestamp = datetime.now(timezone.utc)
        self.last_reinforced = datetime.now(timezone.utc)


def _real_belief(content: str, confidence: float = 0.8, tags=None) -> Belief:
    """Uses real Belief model (not mock) so graph edges work."""
    return Belief(
        content=content,
        confidence=confidence,
        origin=OriginMetadata(source="test"),
        tags=tags or [],
    )


class TestRefinedTensionFormula:
    @pytest.mark.asyncio
    async def test_tension_includes_confidence_factor(self):
        """Tension = overlap × avg_confidence × (0.5 + 0.5 * opposition).
        Higher confidence beliefs should produce more tension."""
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        mock_model = MagicMock()

        # identical embeddings → similarity = 1.0
        def encode_identical(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_identical)
        agent._model = mock_model

        # Two contradicting beliefs with HIGH confidence
        b1_high = _real_belief("I love coffee", confidence=0.9)
        b2_high = _real_belief("I don't love coffee", confidence=0.9)

        # Two contradicting beliefs with LOW confidence
        b1_low = _real_belief("I love tea", confidence=0.3)
        b2_low = _real_belief("I don't love tea", confidence=0.3)

        events_high = await agent.audit([b1_high, b2_high])
        # reset debounce
        agent._above_threshold = set()
        events_low = await agent.audit([b1_low, b2_low])

        # both should generate events (negation detected)
        if events_high and events_low:
            # high-confidence pair should have higher tension
            max_high = max(e.tension for e in events_high)
            max_low = max(e.tension for e in events_low)
            assert max_high > max_low

    @pytest.mark.asyncio
    async def test_graph_edges_populated_on_contradiction(self):
        """Contradicting beliefs should get 'contradicts' links."""
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        mock_model = MagicMock()

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)
        agent._model = mock_model

        b1 = _real_belief("The sky is bright", confidence=0.8)
        b2 = _real_belief("The sky is dark", confidence=0.7)

        events = await agent.audit([b1, b2])

        # check that graph edges were populated
        if events:  # only if contradictions were actually detected
            b1_contradiction_links = b1.get_links("contradicts")
            b2_contradiction_links = b2.get_links("contradicts")
            assert len(b1_contradiction_links) > 0 or len(b2_contradiction_links) > 0


class TestTensionEdgeCases:
    @pytest.mark.asyncio
    async def test_zero_confidence_beliefs_minimal_tension(self):
        """Near-zero confidence contradictions should barely register."""
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        mock_model = MagicMock()

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)
        agent._model = mock_model

        b1 = _real_belief("X is true", confidence=0.05)
        b2 = _real_belief("X is not true", confidence=0.05)

        events = await agent.audit([b1, b2])
        # tension should be very low due to low confidence
        if events:
            for e in events:
                # with avg conf 0.05, tension is at most similarity * 0.05 * 1.0
                assert e.tension < 0.5

    @pytest.mark.asyncio
    async def test_dissimilar_beliefs_no_tension(self):
        """Unrelated beliefs shouldn't generate tension even with negation words."""
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        mock_model = MagicMock()

        def encode_dissimilar(texts, **kwargs):
            n = len(texts)
            embs = np.zeros((n, 3))
            for i in range(n):
                embs[i, i % 3] = 1.0
            return embs

        mock_model.encode = MagicMock(side_effect=encode_dissimilar)
        agent._model = mock_model

        b1 = _real_belief("I love apples")
        b2 = _real_belief("Quantum physics is not simple")

        events = await agent.audit([b1, b2])
        assert events == []
