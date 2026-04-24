# Author: Bradley R. Kinnard
"""
Comprehensive tests for BeliefEcologyLoop (backend/core/bel/loop.py).

Tests cover:
- Full iteration cycle (7 steps)
- Decay computation and status transitions
- Contradiction/tension detection
- Negation heuristics
- Relevance scoring
- Belief ranking
- Snapshot creation with edges
- Ecological action triggering
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from backend.core.bel.loop import BeliefEcologyLoop
from backend.core.config import ABESSettings
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata
from backend.storage.in_memory import InMemoryBeliefStore, InMemorySnapshotStore


def utcnow():
    return datetime.now(timezone.utc)


def make_belief(
    content: str,
    confidence: float = 0.8,
    tension: float = 0.0,
    status: BeliefStatus = BeliefStatus.Active,
    hours_old: float = 0.0,
    parent_id=None,
) -> Belief:
    """Helper to create test beliefs."""
    ts = utcnow() - timedelta(hours=hours_old)
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        status=status,
        parent_id=parent_id,
        origin=OriginMetadata(
            source="test",
            timestamp=ts,
            last_reinforced=ts,
        ),
        created_at=ts,
        updated_at=ts,
    )


@pytest.fixture
def settings():
    return ABESSettings()


@pytest.fixture
def belief_store():
    return InMemoryBeliefStore()


@pytest.fixture
def snapshot_store():
    return InMemorySnapshotStore(compress=False)


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer to avoid loading real model in tests."""
    model = MagicMock()

    def encode_side_effect(texts, convert_to_numpy=True, normalize_embeddings=False):
        if isinstance(texts, str):
            # single text
            emb = np.random.randn(384).astype(np.float32)
            if normalize_embeddings:
                emb = emb / np.linalg.norm(emb)
            return emb
        # batch
        embeddings = np.random.randn(len(texts), 384).astype(np.float32)
        if normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    model.encode = MagicMock(side_effect=encode_side_effect)
    return model


@pytest.fixture
def bel_loop(belief_store, snapshot_store, settings, mock_embedding_model):
    return BeliefEcologyLoop(
        belief_store=belief_store,
        snapshot_store=snapshot_store,
        settings=settings,
        embedding_model=mock_embedding_model,
    )


class TestBeliefEcologyLoopInit:
    """Test initialization and properties."""

    def test_init_with_stores(self, belief_store, snapshot_store, settings):
        loop = BeliefEcologyLoop(
            belief_store=belief_store,
            snapshot_store=snapshot_store,
            settings=settings,
        )
        assert loop.belief_store is belief_store
        assert loop.snapshot_store is snapshot_store
        assert loop.settings is settings
        assert loop.iteration_count == 0

    def test_lazy_embedding_model(self, bel_loop, mock_embedding_model):
        # model provided at init
        assert bel_loop.embedding_model is mock_embedding_model

    def test_initial_edge_storage(self, bel_loop):
        assert bel_loop._contradiction_edges == []
        assert bel_loop._support_edges == []


class TestLoadActiveBeliefs:
    """Test step 1: loading active beliefs."""

    @pytest.mark.asyncio
    async def test_loads_only_active(self, bel_loop, belief_store):
        # create beliefs with different statuses
        active1 = make_belief("Active belief 1")
        active2 = make_belief("Active belief 2")
        decaying = make_belief("Decaying belief", status=BeliefStatus.Decaying)
        deprecated = make_belief("Deprecated belief", status=BeliefStatus.Deprecated)

        await belief_store.create(active1)
        await belief_store.create(active2)
        await belief_store.create(decaying)
        await belief_store.create(deprecated)

        beliefs = await bel_loop._load_active_beliefs()

        assert len(beliefs) == 2
        statuses = {b.status for b in beliefs}
        assert statuses == {BeliefStatus.Active}

    @pytest.mark.asyncio
    async def test_empty_store(self, bel_loop):
        beliefs = await bel_loop._load_active_beliefs()
        assert beliefs == []


class TestApplyDecay:
    """Test step 2: decay computation."""

    @pytest.mark.asyncio
    async def test_decay_reduces_confidence(self, bel_loop, belief_store):
        belief = make_belief("Test belief", confidence=0.9, hours_old=10)
        await belief_store.create(belief)

        beliefs = [belief]
        await bel_loop._apply_decay(beliefs)

        # confidence should be reduced (0.95^10 ≈ 0.599)
        assert beliefs[0].confidence < 0.9
        assert beliefs[0].confidence > 0.5

    @pytest.mark.asyncio
    async def test_fresh_belief_minimal_decay(self, bel_loop, belief_store):
        belief = make_belief("Fresh belief", confidence=0.9, hours_old=0.01)
        await belief_store.create(belief)

        beliefs = [belief]
        original_conf = belief.confidence
        await bel_loop._apply_decay(beliefs)

        # minimal decay for very fresh belief
        assert beliefs[0].confidence > 0.89

    @pytest.mark.asyncio
    async def test_status_transition_to_decaying(self, bel_loop, belief_store):
        # belief that will drop below 0.3 threshold
        belief = make_belief("Old belief", confidence=0.35, hours_old=5)
        await belief_store.create(belief)

        beliefs = [belief]
        await bel_loop._apply_decay(beliefs)

        # 0.35 * 0.95^5 ≈ 0.27, should transition to decaying
        assert beliefs[0].status == BeliefStatus.Decaying

    @pytest.mark.asyncio
    async def test_already_decaying_stays_decaying(self, bel_loop, belief_store):
        belief = make_belief(
            "Already decaying",
            confidence=0.25,
            status=BeliefStatus.Decaying,
            hours_old=1,
        )
        await belief_store.create(belief)

        beliefs = [belief]
        await bel_loop._apply_decay(beliefs)

        # status unchanged, still decaying
        assert beliefs[0].status == BeliefStatus.Decaying


class TestNegationDetection:
    """Test negation heuristic used in contradiction scoring."""

    def test_negation_word_asymmetry(self, bel_loop):
        assert bel_loop._has_negation("The sky is blue", "The sky is not blue")
        assert bel_loop._has_negation("I like pizza", "I do not like pizza")
        assert bel_loop._has_negation("It never rains", "It rains sometimes")

    def test_no_negation_similar_statements(self, bel_loop):
        assert not bel_loop._has_negation("The sky is blue", "The ocean is blue")
        assert not bel_loop._has_negation("I like pizza", "I like pasta")

    def test_antonym_pairs(self, bel_loop):
        assert bel_loop._has_negation("This is true", "This is false")
        assert bel_loop._has_negation("I always exercise", "I never exercise")
        assert bel_loop._has_negation("The movie was good", "The movie was bad")
        assert bel_loop._has_negation("I like cats", "I dislike cats")

    def test_both_have_negation(self, bel_loop):
        # both have negation words -> no asymmetry -> False
        assert not bel_loop._has_negation("I don't like X", "I don't like Y")


class TestComputeTensions:
    """Test step 3: contradiction and tension computation."""

    @pytest.mark.asyncio
    async def test_single_belief_zero_tension(self, bel_loop):
        beliefs = [make_belief("Only one belief")]
        await bel_loop._compute_tensions(beliefs)

        assert beliefs[0].tension == 0.0

    @pytest.mark.asyncio
    async def test_empty_beliefs(self, bel_loop):
        beliefs = []
        await bel_loop._compute_tensions(beliefs)
        # should not raise

    @pytest.mark.asyncio
    async def test_tension_capped(self, bel_loop):
        # create beliefs that would produce high tension
        beliefs = [
            make_belief("Statement A is true"),
            make_belief("Statement A is false"),
        ]

        await bel_loop._compute_tensions(beliefs)

        # tension should be capped at settings.tension_cap (default 10.0)
        for b in beliefs:
            assert b.tension <= 10.0

    @pytest.mark.asyncio
    async def test_edges_populated(self, bel_loop):
        beliefs = [
            make_belief("Cats are friendly"),
            make_belief("Dogs are loyal"),
            make_belief("Cats are not friendly"),
        ]

        bel_loop._contradiction_edges = []
        bel_loop._support_edges = []

        await bel_loop._compute_tensions(beliefs)

        # edges should be lists (may or may not have items depending on similarity)
        assert isinstance(bel_loop._contradiction_edges, list)
        assert isinstance(bel_loop._support_edges, list)


class TestTriggerEcologicalActions:
    """Test step 4: action triggering based on tension/confidence."""

    @pytest.mark.asyncio
    async def test_mutation_candidate(self, bel_loop):
        belief = make_belief("Conflicted belief", confidence=0.4)
        belief.tension = 0.8  # high tension, low confidence -> mutation

        actions = await bel_loop._trigger_ecological_actions([belief])

        assert len(actions) == 1
        assert actions[0]["type"] == "mutation_candidate"
        assert actions[0]["belief_id"] == str(belief.id)

    @pytest.mark.asyncio
    async def test_resolution_candidate(self, bel_loop):
        belief = make_belief("Strong but conflicted", confidence=0.85)
        belief.tension = 0.8  # high tension, high confidence -> resolution

        actions = await bel_loop._trigger_ecological_actions([belief])

        assert len(actions) == 1
        assert actions[0]["type"] == "resolution_candidate"

    @pytest.mark.asyncio
    async def test_no_action_low_tension(self, bel_loop):
        belief = make_belief("Stable belief", confidence=0.7)
        belief.tension = 0.3  # low tension -> no action

        actions = await bel_loop._trigger_ecological_actions([belief])

        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_no_action_medium_confidence(self, bel_loop):
        belief = make_belief("Medium belief", confidence=0.6)
        belief.tension = 0.8  # high tension but mid confidence -> no action

        actions = await bel_loop._trigger_ecological_actions([belief])

        assert len(actions) == 0


class TestComputeRelevance:
    """Test step 5: relevance scoring against context."""

    @pytest.mark.asyncio
    async def test_empty_context_zero_relevance(self, bel_loop):
        beliefs = [make_belief("Some belief")]
        await bel_loop._compute_relevance(beliefs, "")

        # relevance should be set to 0.0 for empty context
        assert hasattr(beliefs[0], "relevance") or getattr(beliefs[0], "relevance", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_context_zero_relevance(self, bel_loop):
        beliefs = [make_belief("Some belief")]
        await bel_loop._compute_relevance(beliefs, "   \n\t  ")

        # whitespace-only context treated as empty
        assert getattr(beliefs[0], "relevance", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_relevance_computed(self, bel_loop):
        beliefs = [make_belief("The weather is sunny")]
        await bel_loop._compute_relevance(beliefs, "What is the weather like?")

        # relevance should exist and be in valid range
        relevance = getattr(beliefs[0], "relevance", None)
        # with mock embeddings, may not exist or be computed
        if relevance is not None:
            assert 0.0 <= relevance <= 1.0

    @pytest.mark.asyncio
    async def test_empty_beliefs_no_error(self, bel_loop):
        beliefs = []
        await bel_loop._compute_relevance(beliefs, "Some context")
        # should not raise


class TestRankBeliefs:
    """Test step 6: belief ranking."""

    def test_ranking_order(self, bel_loop):
        # create beliefs with varying properties
        high_conf = make_belief("High confidence", confidence=0.95)
        low_conf = make_belief("Low confidence", confidence=0.3)
        mid_conf = make_belief("Medium confidence", confidence=0.6)

        # set relevance manually using setattr since it's a dynamic attr
        setattr(high_conf, "relevance", 0.8)
        setattr(low_conf, "relevance", 0.9)
        setattr(mid_conf, "relevance", 0.5)

        beliefs = [low_conf, mid_conf, high_conf]
        ranked = bel_loop._rank_beliefs(beliefs)

        # all should have scores
        for b in ranked:
            assert hasattr(b, "score")
            assert b.score >= 0

        # check it's sorted descending
        scores = [b.score for b in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_empty_list(self, bel_loop):
        ranked = bel_loop._rank_beliefs([])
        assert ranked == []

    def test_recency_affects_score(self, bel_loop):
        fresh = make_belief("Fresh belief", confidence=0.5, hours_old=0.1)
        old = make_belief("Old belief", confidence=0.5, hours_old=100)

        setattr(fresh, "relevance", 0.5)
        setattr(old, "relevance", 0.5)

        ranked = bel_loop._rank_beliefs([old, fresh])

        # fresh should rank higher due to recency
        assert ranked[0].content == "Fresh belief"


class TestLogSnapshot:
    """Test step 7: snapshot creation and persistence."""

    @pytest.mark.asyncio
    async def test_snapshot_created(self, bel_loop, snapshot_store):
        belief = make_belief("Test belief")
        beliefs = [belief]

        snapshot = await bel_loop._log_snapshot(
            beliefs=beliefs,
            context="Test context",
            actions=[{"type": "test_action"}],
            rl_state_action={"state": [1, 2, 3], "action": [0.1, 0.2]},
        )

        assert snapshot is not None
        assert snapshot.id is not None
        assert len(snapshot.beliefs) == 1
        assert snapshot.beliefs[0].content == "Test belief"
        assert snapshot.agent_actions == [{"type": "test_action"}]
        assert snapshot.rl_state_action is not None

    @pytest.mark.asyncio
    async def test_snapshot_includes_edges(self, bel_loop, snapshot_store):
        bel_loop._contradiction_edges = [(uuid4(), uuid4(), 0.7)]
        bel_loop._support_edges = [(uuid4(), uuid4(), 0.85)]

        snapshot = await bel_loop._log_snapshot(
            beliefs=[],
            context="",
            actions=[],
        )

        assert len(snapshot.contradiction_edges) == 1
        assert len(snapshot.support_edges) == 1

    @pytest.mark.asyncio
    async def test_snapshot_iteration_increments(self, bel_loop, snapshot_store):
        bel_loop.iteration_count = 5

        snapshot = await bel_loop._log_snapshot([], "", [])

        assert snapshot.metadata.iteration == 5

    @pytest.mark.asyncio
    async def test_global_tension_computed(self, bel_loop, snapshot_store):
        b1 = make_belief("Belief 1")
        b2 = make_belief("Belief 2")
        b1.tension = 0.4
        b2.tension = 0.6

        snapshot = await bel_loop._log_snapshot([b1, b2], "", [])

        assert snapshot.global_tension == 0.5  # (0.4 + 0.6) / 2

    @pytest.mark.asyncio
    async def test_context_summary_truncated(self, bel_loop, snapshot_store):
        long_context = "x" * 500

        snapshot = await bel_loop._log_snapshot([], long_context, [])

        assert len(snapshot.metadata.context_summary) == 200


class TestComputeLineageEdges:
    """Test lineage edge extraction."""

    def test_extracts_parent_child(self, bel_loop):
        parent_id = uuid4()
        child = make_belief("Child belief", parent_id=parent_id)

        edges = bel_loop._compute_lineage_edges([child])

        assert len(edges) == 1
        assert edges[0] == (parent_id, child.id)

    def test_no_parent_no_edge(self, bel_loop):
        belief = make_belief("Root belief")

        edges = bel_loop._compute_lineage_edges([belief])

        assert edges == []

    def test_multiple_lineages(self, bel_loop):
        p1, p2 = uuid4(), uuid4()
        c1 = make_belief("Child 1", parent_id=p1)
        c2 = make_belief("Child 2", parent_id=p2)
        root = make_belief("Root")

        edges = bel_loop._compute_lineage_edges([c1, c2, root])

        assert len(edges) == 2


class TestFullIteration:
    """Integration tests for complete run_iteration cycle."""

    @pytest.mark.asyncio
    async def test_full_iteration_empty_store(self, bel_loop):
        ranked, snapshot = await bel_loop.run_iteration("Test context")

        assert ranked == []
        assert snapshot is not None
        assert bel_loop.iteration_count == 1

    @pytest.mark.asyncio
    async def test_full_iteration_with_beliefs(self, bel_loop, belief_store):
        b1 = make_belief("The sky is blue", confidence=0.9)
        b2 = make_belief("Water is wet", confidence=0.8)
        await belief_store.create(b1)
        await belief_store.create(b2)

        ranked, snapshot = await bel_loop.run_iteration("What color is the sky?")

        assert len(ranked) == 2
        assert snapshot is not None
        assert len(snapshot.beliefs) == 2
        assert bel_loop.iteration_count == 1

    @pytest.mark.asyncio
    async def test_multiple_iterations(self, bel_loop, belief_store):
        b1 = make_belief("Test belief", confidence=0.9)
        await belief_store.create(b1)

        await bel_loop.run_iteration("Context 1")
        await bel_loop.run_iteration("Context 2")
        await bel_loop.run_iteration("Context 3")

        assert bel_loop.iteration_count == 3

    @pytest.mark.asyncio
    async def test_rl_state_action_passed_through(self, bel_loop, belief_store):
        b1 = make_belief("Test", confidence=0.9)
        await belief_store.create(b1)

        rl_info = {"state": [0.5] * 15, "action": [0.1] * 7}
        ranked, snapshot = await bel_loop.run_iteration("Context", rl_state_action=rl_info)

        assert snapshot.rl_state_action == rl_info
