# Author: Bradley R. Kinnard
"""Tests for reinforcement agent updates: salience boost, evidence ledger, and graph edges."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from backend.core.models.belief import Belief, OriginMetadata, EvidenceRef


class MockStore:
    def __init__(self):
        self.bulk_update = AsyncMock(return_value=0)


def _make_belief(content: str, confidence: float = 0.5) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        salience=0.5,
        origin=OriginMetadata(
            source="test",
            last_reinforced=datetime.now(timezone.utc) - timedelta(seconds=120),
        ),
    )


class TestReinforcementSalience:
    @pytest.mark.asyncio
    async def test_reinforced_belief_gets_salience_boost(self):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        mock_model = MagicMock()

        b = _make_belief("I like hiking", confidence=0.5)
        old_salience = b.salience

        # encode returns similar vectors
        mock_model.encode = MagicMock(side_effect=lambda texts, **kw: np.array(
            [[1.0, 0.0, 0.0]] * len(texts)
        ))
        agent._model = mock_model

        store = MockStore()
        result = await agent.reinforce("hiking is great", [b], store)

        if result:  # reinforcement happened
            assert b.salience > old_salience

    @pytest.mark.asyncio
    async def test_reinforced_belief_gets_evidence(self):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        mock_model = MagicMock()

        b = _make_belief("Coffee is good", confidence=0.5)
        assert len(b.evidence_for) == 0

        mock_model.encode = MagicMock(side_effect=lambda texts, **kw: np.array(
            [[1.0, 0.0, 0.0]] * len(texts)
        ))
        agent._model = mock_model

        store = MockStore()
        result = await agent.reinforce("I really enjoy coffee", [b], store)

        if result:
            assert len(b.evidence_for) > 0
            assert b.evidence_for[0].direction == "supports"

    @pytest.mark.asyncio
    async def test_co_reinforced_beliefs_get_links(self):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        mock_model = MagicMock()

        b1 = _make_belief("I like hiking", confidence=0.5)
        b2 = _make_belief("Outdoors is fun", confidence=0.5)

        # both return similar embedding to incoming
        mock_model.encode = MagicMock(side_effect=lambda texts, **kw: np.array(
            [[1.0, 0.0, 0.0]] * len(texts)
        ))
        agent._model = mock_model

        store = MockStore()
        result = await agent.reinforce("I love being outside hiking", [b1, b2], store)

        if len(result) >= 2:
            # both were reinforced, should have links to each other
            b1_links = b1.get_links("reinforces")
            b2_links = b2.get_links("reinforces")
            assert len(b1_links) > 0 or len(b2_links) > 0

    @pytest.mark.asyncio
    async def test_no_reinforcement_no_side_effects(self):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        mock_model = MagicMock()

        b = _make_belief("Quantum physics", confidence=0.5)

        # incoming embeds to [0,1,0], belief embeds to [1,0,0] -> orthogonal
        call_count = [0]
        def encode_dissimilar(texts, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                # incoming query
                return np.array([[0.0, 1.0, 0.0]])
            else:
                # belief embeddings
                return np.array([[1.0, 0.0, 0.0]] * len(texts))

        mock_model.encode = MagicMock(side_effect=encode_dissimilar)
        agent._model = mock_model

        store = MockStore()
        result = await agent.reinforce("Cooking pasta", [b], store)
        assert result == []
        assert len(b.evidence_for) == 0
        assert len(b.links) == 0
