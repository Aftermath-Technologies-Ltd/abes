# Author: Bradley R. Kinnard
"""Tests for ContradictionAuditorAgent."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone


class MockOriginMetadata:
    def __init__(self):
        self.source = "test"
        self.timestamp = datetime.now(timezone.utc)
        self.last_reinforced = datetime.now(timezone.utc)


class MockBelief:
    def __init__(self, content: str, belief_id=None, tags=None):
        self.id = belief_id or uuid4()
        self.content = content
        self.confidence = 0.8
        self.origin = MockOriginMetadata()
        self.tags = tags or []
        self.tension = 0.0
        self.status = "active"
        self.salience = 0.5
        self.links = []
        self.evidence_for = []
        self.evidence_against = []

    def add_link(self, target_id, relation, weight=1.0):
        self.links.append({"target_id": target_id, "relation": relation, "weight": weight})

    def get_links(self, relation=None):
        if relation is None:
            return self.links
        return [l for l in self.links if l["relation"] == relation]


@pytest.fixture
def mock_model():
    model = MagicMock()
    return model


class TestAudit:
    """Test contradiction audit logic."""

    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        result = await agent.audit([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_contradictions(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        # embeddings that are dissimilar
        def encode_dissimilar(texts, **kwargs):
            return np.array([[float(i), 0.0, 0.0] for i in range(len(texts))])

        mock_model.encode = MagicMock(side_effect=encode_dissimilar)

        beliefs = [
            MockBelief("I like coffee"),
            MockBelief("The weather is nice"),
        ]

        result = await agent.audit(beliefs)
        assert result == []

    @pytest.mark.asyncio
    async def test_detects_contradiction(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        # similar embeddings
        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        # texts with negation
        beliefs = [
            MockBelief("I like coffee"),
            MockBelief("I don't like coffee"),
        ]

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            result = await agent.audit(beliefs)

            # both should have events (bidirectional tension)
            assert len(result) == 2
            assert result[0].tension > 0.5
            assert result[0].threshold == 0.5

    @pytest.mark.asyncio
    async def test_event_contains_belief_id(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        belief1 = MockBelief("always true")
        belief2 = MockBelief("never true")

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            result = await agent.audit([belief1, belief2])

            ids = [e.belief_id for e in result]
            assert belief1.id in ids
            assert belief2.id in ids

    @pytest.mark.asyncio
    async def test_below_threshold_not_emitted(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        beliefs = [
            MockBelief("I like coffee"),
            MockBelief("I don't like coffee"),
        ]

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            # very high threshold - nothing should pass
            mock_settings.tension_threshold_high = 10.0

            result = await agent.audit(beliefs)
            assert result == []

    @pytest.mark.asyncio
    async def test_sensitive_tag_lowers_threshold(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        # one belief with critical tag
        belief1 = MockBelief("always safe", tags=["critical"])
        belief2 = MockBelief("never safe")

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            # high default threshold
            mock_settings.tension_threshold_high = 10.0

            result = await agent.audit([belief1, belief2])

            # belief1 should trigger due to lower threshold from tag
            ids = [e.belief_id for e in result]
            assert belief1.id in ids

    @pytest.mark.asyncio
    async def test_debounce_no_duplicate_events(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        beliefs = [
            MockBelief("I like coffee"),
            MockBelief("I don't like coffee"),
        ]

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            # first audit - should emit
            result1 = await agent.audit(beliefs)
            assert len(result1) == 2

            # second audit same beliefs - should not emit (debounced)
            result2 = await agent.audit(beliefs)
            assert len(result2) == 0

    @pytest.mark.asyncio
    async def test_debounce_emits_when_crosses_again(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        def encode_dissimilar(texts, **kwargs):
            return np.array([[float(i), 0.0, 0.0] for i in range(len(texts))])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        beliefs = [
            MockBelief("I like coffee"),
            MockBelief("I don't like coffee"),
        ]

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            # first: above threshold
            result1 = await agent.audit(beliefs)
            assert len(result1) == 2

            # simulate: beliefs now dissimilar (drops below)
            # clear cache so new embeddings get used
            agent._embedding_cache.clear()
            mock_model.encode = MagicMock(side_effect=encode_dissimilar)
            result2 = await agent.audit(beliefs)
            assert len(result2) == 0

            # back to similar - should emit again
            agent._embedding_cache.clear()
            mock_model.encode = MagicMock(side_effect=encode_similar)
            result3 = await agent.audit(beliefs)
            assert len(result3) == 2


class TestLazyLoading:
    def test_model_not_loaded_on_init(self):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        assert agent._model is None


class TestPersistence:
    """Test optional state persistence."""

    @pytest.mark.asyncio
    async def test_persists_state_when_store_supports(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent
        from unittest.mock import AsyncMock

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        store = MagicMock()
        store.load_contradiction_state = AsyncMock(return_value=[])
        store.save_contradiction_state = AsyncMock()

        beliefs = [
            MockBelief("always good"),
            MockBelief("never good"),
        ]

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            await agent.audit(beliefs, store=store)

            store.save_contradiction_state.assert_called_once()
            saved_ids = store.save_contradiction_state.call_args[0][0]
            assert len(saved_ids) == 2

    @pytest.mark.asyncio
    async def test_loads_persisted_state(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent
        from unittest.mock import AsyncMock

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        belief1 = MockBelief("always good")
        belief2 = MockBelief("never good")

        store = MagicMock()
        # pretend these were already above threshold
        store.load_contradiction_state = AsyncMock(return_value=[belief1.id, belief2.id])
        store.save_contradiction_state = AsyncMock()

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            result = await agent.audit([belief1, belief2], store=store)

            # should not emit - they were already above
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fallback_to_memory_without_store_methods(self, mock_model):
        from backend.agents.contradiction_auditor import ContradictionAuditorAgent

        agent = ContradictionAuditorAgent()
        agent._model = mock_model

        def encode_similar(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_similar)

        # store without persistence methods
        store = MagicMock(spec=[])

        beliefs = [
            MockBelief("always good"),
            MockBelief("never good"),
        ]

        with patch("backend.agents.contradiction_auditor.settings") as mock_settings:
            mock_settings.tension_threshold_high = 0.5

            # should work fine, just use in-memory state
            result = await agent.audit(beliefs, store=store)
            assert len(result) == 2


__all__ = []
