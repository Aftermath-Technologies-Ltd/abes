# Author: Bradley R. Kinnard
"""Tests for the ConsolidationAgent."""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from backend.agents.consolidation import ConsolidationAgent, ConsolidationEvent
from backend.core.models.belief import Belief, BeliefStatus, EvidenceRef, OriginMetadata


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.7,
    salience: float = 0.5,
    status: BeliefStatus = BeliefStatus.Active,
    cluster_id=None,
    parent_id=None,
    **kwargs,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        status=status,
        cluster_id=cluster_id,
        parent_id=parent_id,
        origin=OriginMetadata(source="test"),
        **kwargs,
    )


class TestConsolidationAgent:
    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = ConsolidationAgent()
        events, new, deprecated = await agent.consolidate([])
        assert events == []
        assert new == []
        assert deprecated == []

    @pytest.mark.asyncio
    async def test_no_merges_when_different_clusters(self):
        """Beliefs in different clusters shouldn't merge."""
        c1, c2 = uuid4(), uuid4()
        b1 = _make_belief(content="cats are furry", cluster_id=c1)
        b2 = _make_belief(content="dogs are loyal", cluster_id=c2)

        agent = ConsolidationAgent()
        # mock model to return similar embeddings
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.array([[1, 0, 0], [0, 1, 0]]))
        agent._model = mock_model

        events, new, deprecated = await agent.consolidate([b1, b2])
        merged_events = [e for e in events if e.event_type == "merged"]
        assert len(merged_events) == 0

    @pytest.mark.asyncio
    async def test_merges_near_duplicates_in_same_cluster(self):
        """Beliefs with >0.92 similarity in same cluster get merged."""
        cluster = uuid4()
        b1 = _make_belief(content="cats are furry", confidence=0.9, salience=0.8, cluster_id=cluster)
        b2 = _make_belief(content="cats are very furry", confidence=0.5, salience=0.3, cluster_id=cluster)

        agent = ConsolidationAgent(merge_threshold=0.9)
        mock_model = MagicMock()
        # return identical normalized embeddings -> similarity = 1.0
        emb = np.array([[0.6, 0.8, 0.0], [0.6, 0.8, 0.0]])
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        mock_model.encode = MagicMock(return_value=emb / norm)
        agent._model = mock_model

        events, new, deprecated = await agent.consolidate([b1, b2])
        merged_events = [e for e in events if e.event_type == "merged"]
        assert len(merged_events) == 1
        assert len(deprecated) == 1
        # the loser should be deprecated
        assert deprecated[0].status == BeliefStatus.Deprecated

    @pytest.mark.asyncio
    async def test_merge_absorbs_evidence(self):
        """Winner absorbs losers' evidence."""
        cluster = uuid4()
        b1 = _make_belief(content="fact A", confidence=0.9, salience=0.9, cluster_id=cluster)
        b2 = _make_belief(content="fact A also", confidence=0.4, salience=0.2, cluster_id=cluster)
        b2.evidence_for.append(EvidenceRef(content="some proof", direction="supports"))

        agent = ConsolidationAgent(merge_threshold=0.9)
        mock_model = MagicMock()
        emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mock_model.encode = MagicMock(return_value=emb)
        agent._model = mock_model

        events, new, deprecated = await agent.consolidate([b1, b2])
        # b1 is the winner (higher conf*sal), should have b2's evidence
        assert len(b1.evidence_for) == 1

    @pytest.mark.asyncio
    async def test_no_merge_below_threshold(self):
        """Beliefs with low similarity shouldn't merge even in same cluster."""
        cluster = uuid4()
        b1 = _make_belief(content="cats", cluster_id=cluster)
        b2 = _make_belief(content="quantum physics", cluster_id=cluster)

        agent = ConsolidationAgent(merge_threshold=0.92)
        mock_model = MagicMock()
        # orthogonal embeddings -> similarity = 0
        emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mock_model.encode = MagicMock(return_value=emb)
        agent._model = mock_model

        events, new, deprecated = await agent.consolidate([b1, b2])
        assert len(deprecated) == 0


class TestLineageCompression:
    @pytest.mark.asyncio
    async def test_compresses_deep_lineage(self):
        """Chains deeper than max_depth get compressed."""
        agent = ConsolidationAgent(max_depth=3)
        agent._model = MagicMock()  # avoid loading real model

        # build a chain of depth 4: b4 -> b3 -> b2 -> b1
        b1 = _make_belief(content="original claim")
        b2 = _make_belief(content="mutation 1", parent_id=b1.id)
        b3 = _make_belief(content="mutation 2", parent_id=b2.id)
        b4 = _make_belief(content="mutation 3", parent_id=b3.id)

        # need all in the beliefs list so the belief_map is populated
        beliefs = [b4, b3, b2, b1]

        events, new, deprecated = await agent.consolidate(beliefs)
        compressed = [e for e in events if e.event_type == "compressed"]
        assert len(compressed) == 1
        # tip (b4) should survive, ancestors deprecated
        assert b4.parent_id is None  # lineage reset

    @pytest.mark.asyncio
    async def test_no_compression_for_shallow_lineage(self):
        agent = ConsolidationAgent(max_depth=5)
        agent._model = MagicMock()

        b1 = _make_belief(content="claim")
        b2 = _make_belief(content="mutation", parent_id=b1.id)

        events, new, deprecated = await agent.consolidate([b2, b1])
        compressed = [e for e in events if e.event_type == "compressed"]
        assert len(compressed) == 0

    @pytest.mark.asyncio
    async def test_compression_absorbs_ancestor_evidence(self):
        agent = ConsolidationAgent(max_depth=2)
        agent._model = MagicMock()

        b1 = _make_belief(content="root")
        b1.evidence_for.append(EvidenceRef(content="old proof", direction="supports"))
        b2 = _make_belief(content="child", parent_id=b1.id)
        b3 = _make_belief(content="grandchild", parent_id=b2.id)

        events, new, deprecated = await agent.consolidate([b3, b2, b1])
        # tip (b3) should have absorbed b1's evidence
        assert len(b3.evidence_for) >= 1
        assert any(e.content == "old proof" for e in b3.evidence_for)
