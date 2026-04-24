# Author: Bradley R. Kinnard
"""Tests for RelevanceCuratorAgent."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np

from backend.agents.relevance_curator import (
    RelevanceCuratorAgent,
    RankedBelief,
    _hours_since,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.8,
    tension: float = 0.0,
    source: str = "test",
    tags: list[str] = None,
    last_reinforced: datetime = None,
) -> Belief:
    lr = last_reinforced or datetime.now(timezone.utc)
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        origin=OriginMetadata(source=source, last_reinforced=lr),
        tags=tags or [],
    )


class TestHoursSince:
    def test_recent(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=2)
        hours = _hours_since(dt)
        assert 1.9 < hours < 2.1

    def test_old(self):
        dt = datetime.now(timezone.utc) - timedelta(days=7)
        hours = _hours_since(dt)
        assert 167 < hours < 169


class TestRecency:
    def test_just_reinforced(self):
        agent = RelevanceCuratorAgent(recency_window_hours=168)
        b = _make_belief(last_reinforced=datetime.now(timezone.utc))
        recency = agent._compute_recency(b)
        assert recency > 0.99

    def test_half_window(self):
        agent = RelevanceCuratorAgent(recency_window_hours=168)
        lr = datetime.now(timezone.utc) - timedelta(hours=84)
        b = _make_belief(last_reinforced=lr)
        recency = agent._compute_recency(b)
        assert 0.45 < recency < 0.55

    def test_beyond_window(self):
        agent = RelevanceCuratorAgent(recency_window_hours=168)
        lr = datetime.now(timezone.utc) - timedelta(hours=200)
        b = _make_belief(last_reinforced=lr)
        recency = agent._compute_recency(b)
        assert recency == 0.0


class TestRankScore:
    def test_default_weights(self):
        agent = RelevanceCuratorAgent()
        # relevance=1.0, confidence=1.0, recency=1.0, tension=0.0
        # score = 0.4*1 + 0.3*1 + 0.2*1 - 0.1*0 = 0.9
        score = agent._compute_rank_score(1.0, 1.0, 1.0, 0.0)
        assert abs(score - 0.9) < 0.001

    def test_tension_penalty(self):
        agent = RelevanceCuratorAgent()
        # same as above but tension=1.0
        # score = 0.4 + 0.3 + 0.2 - 0.1 = 0.8
        score = agent._compute_rank_score(1.0, 1.0, 1.0, 1.0)
        assert abs(score - 0.8) < 0.001

    def test_custom_weights(self):
        agent = RelevanceCuratorAgent(
            weights={"relevance": 0.5, "confidence": 0.3, "recency": 0.1, "tension": 0.1}
        )
        # 0.5*1 + 0.3*0.5 + 0.1*1 - 0.1*0 = 0.75
        score = agent._compute_rank_score(1.0, 0.5, 1.0, 0.0)
        assert abs(score - 0.75) < 0.001


class TestUpdateWeights:
    def test_updates_within_bounds(self):
        agent = RelevanceCuratorAgent()
        agent.update_weights({"relevance": 0.5})
        assert agent._weights["relevance"] == 0.5

    def test_clamps_high(self):
        agent = RelevanceCuratorAgent()
        agent.update_weights({"relevance": 0.9})  # default bounds are (0.1, 0.6)
        assert agent._weights["relevance"] == 0.6

    def test_clamps_low(self):
        agent = RelevanceCuratorAgent()
        agent.update_weights({"relevance": 0.05})
        assert agent._weights["relevance"] == 0.1

    def test_ignores_unknown_keys(self):
        agent = RelevanceCuratorAgent()
        original = agent._weights.copy()
        agent.update_weights({"unknown_key": 0.5})
        assert agent._weights == original


class TestLazyLoading:
    def test_model_not_loaded_on_init(self):
        agent = RelevanceCuratorAgent()
        assert agent._model is None


class TestComputeRelevance:
    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = RelevanceCuratorAgent()
        result = await agent.compute_relevance_scores([], "some context")
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_context(self):
        agent = RelevanceCuratorAgent()
        b = _make_belief()
        result = await agent.compute_relevance_scores([b], "")
        assert result == {}

    @pytest.mark.asyncio
    async def test_uses_stored_embeddings(self):
        agent = RelevanceCuratorAgent()
        b = _make_belief(content="coffee")

        # mock embedding that's similar to "coffee" context
        stored = {b.id: [1.0] + [0.0] * 383}  # 384-dim

        with patch.object(agent, "_get_model") as mock_get:
            mock_model = MagicMock()
            # context embedding
            mock_model.encode.return_value = np.array([1.0] + [0.0] * 383)
            mock_get.return_value = mock_model

            result = await agent.compute_relevance_scores([b], "coffee", stored)

            # should have called encode only once (for context)
            assert mock_model.encode.call_count == 1
            assert b.id in result


class TestRankBeliefs:
    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = RelevanceCuratorAgent()
        result = await agent.rank_beliefs([], "context")
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_below_threshold(self):
        agent = RelevanceCuratorAgent(relevance_threshold=0.5)

        with patch.object(agent, "compute_relevance_scores") as mock_rel:
            b1 = _make_belief(content="relevant")
            b2 = _make_belief(content="not relevant")
            mock_rel.return_value = {b1.id: 0.8, b2.id: 0.2}

            result = await agent.rank_beliefs([b1, b2], "context")

            assert len(result) == 1
            assert result[0].belief.id == b1.id

    @pytest.mark.asyncio
    async def test_sorts_by_rank_descending(self):
        agent = RelevanceCuratorAgent(relevance_threshold=0.0)

        with patch.object(agent, "compute_relevance_scores") as mock_rel:
            b1 = _make_belief(content="low", confidence=0.3)
            b2 = _make_belief(content="high", confidence=0.9)
            mock_rel.return_value = {b1.id: 0.5, b2.id: 0.5}

            result = await agent.rank_beliefs([b1, b2], "context")

            # b2 should rank higher due to higher confidence
            assert result[0].belief.id == b2.id
            assert result[1].belief.id == b1.id

    @pytest.mark.asyncio
    async def test_uses_tension_map(self):
        agent = RelevanceCuratorAgent(relevance_threshold=0.0)

        with patch.object(agent, "compute_relevance_scores") as mock_rel:
            b1 = _make_belief(content="a", tension=0.0)
            b2 = _make_belief(content="b", tension=0.0)
            mock_rel.return_value = {b1.id: 0.5, b2.id: 0.5}

            # override tensions
            tension_map = {b1.id: 5.0, b2.id: 0.0}

            result = await agent.rank_beliefs([b1, b2], "context", tension_map=tension_map)

            # b2 should rank higher due to lower tension
            assert result[0].belief.id == b2.id


class TestGetTopBeliefs:
    @pytest.mark.asyncio
    async def test_returns_top_k(self):
        agent = RelevanceCuratorAgent(relevance_threshold=0.0)

        with patch.object(agent, "compute_relevance_scores") as mock_rel:
            beliefs = [_make_belief(content=f"b{i}", confidence=0.1 * i) for i in range(5)]
            mock_rel.return_value = {b.id: 0.5 for b in beliefs}

            result = await agent.get_top_beliefs(beliefs, "context", top_k=3)

            assert len(result) == 3
            # highest confidence should be first
            assert result[0].confidence == 0.4

    @pytest.mark.asyncio
    async def test_returns_all_if_less_than_k(self):
        agent = RelevanceCuratorAgent(relevance_threshold=0.0)

        with patch.object(agent, "compute_relevance_scores") as mock_rel:
            beliefs = [_make_belief(content=f"b{i}") for i in range(2)]
            mock_rel.return_value = {b.id: 0.5 for b in beliefs}

            result = await agent.get_top_beliefs(beliefs, "context", top_k=10)

            assert len(result) == 2
