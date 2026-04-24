# Author: Bradley R. Kinnard
"""Tests for BeliefCreatorAgent."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone


# Mock the belief module before importing BeliefCreatorAgent
# to avoid StrEnum import error on Python < 3.11
class MockBeliefStatus:
    Active = "active"
    Decaying = "decaying"


class MockOriginMetadata:
    def __init__(self, source: str, **kwargs):
        self.source = source
        self.turn_index = kwargs.get("turn_index")
        self.episode_id = kwargs.get("episode_id")
        self.timestamp = kwargs.get("timestamp", datetime.now(timezone.utc))
        self.last_reinforced = kwargs.get("last_reinforced", datetime.now(timezone.utc))


class MockBelief:
    def __init__(self, content: str, confidence: float, origin, tags=None, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.content = content
        self.confidence = confidence
        self.origin = origin
        self.tags = tags or []
        self.tension = kwargs.get("tension", 0.0)
        self.cluster_id = kwargs.get("cluster_id")
        self.status = kwargs.get("status", "active")
        self.parent_id = kwargs.get("parent_id")
        self.use_count = kwargs.get("use_count", 0)
        self.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
        self.updated_at = kwargs.get("updated_at", datetime.now(timezone.utc))


@pytest.fixture
def mock_store():
    """Create a mock store with async methods."""
    store = MagicMock()
    store.create = AsyncMock(side_effect=lambda b: b)
    store.list = AsyncMock(return_value=[])
    store.search_by_embedding = AsyncMock(return_value=[])
    # Don't define save_embedding by default - tests that need it add it
    # Use spec=[] to prevent hasattr from returning True for undefined attrs
    del store.save_embedding
    return store


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer."""
    model = MagicMock()
    # Return distinct embeddings for each input
    def encode_side_effect(texts, **kwargs):
        return np.array([[i * 0.1] * 384 for i in range(len(texts))])
    model.encode = MagicMock(side_effect=encode_side_effect)
    return model


@pytest.fixture
def origin():
    return MockOriginMetadata(source="test")


class TestCosineBatch:
    """Test the _cosine_batch helper function."""

    def test_identical_vectors(self):
        from backend.agents.belief_creator import _cosine_batch
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([[1.0, 0.0, 0.0]])
        result = _cosine_batch(query, candidates)
        assert result[0] == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from backend.agents.belief_creator import _cosine_batch
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([[0.0, 1.0, 0.0]])
        result = _cosine_batch(query, candidates)
        assert result[0] == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from backend.agents.belief_creator import _cosine_batch
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([[-1.0, 0.0, 0.0]])
        result = _cosine_batch(query, candidates)
        assert result[0] == pytest.approx(-1.0)

    def test_batch_multiple(self):
        from backend.agents.belief_creator import _cosine_batch
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        result = _cosine_batch(query, candidates)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(-1.0)


class TestAssignTags:
    """Test keyword-based tag assignment."""

    def test_cache_tag(self):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            tags = agent._assign_tags("the cache is full")
            assert "infra.cache" in tags

    def test_weight_tag(self):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            tags = agent._assign_tags("weights diverged")
            assert "model.weights" in tags

    def test_multiple_tags(self):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            tags = agent._assign_tags("cache timeout during training")
            assert "infra.cache" in tags
            assert "perf.timeout" in tags

    def test_no_tags(self):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            tags = agent._assign_tags("hello world")
            assert tags == []


class TestCreateBeliefs:
    """Test the main create_beliefs method."""

    @pytest.mark.asyncio
    async def test_empty_candidates(self, mock_store, origin):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            result = await agent.create_beliefs([], origin, mock_store)
            assert result == []
            mock_store.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_novel_belief(self, mock_store, mock_model, origin):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            agent._model = mock_model

            # mock store.create to return the belief with an id
            async def create_belief(b):
                b.id = uuid4()
                return b
            mock_store.create = AsyncMock(side_effect=create_belief)

            result = await agent.create_beliefs(["the cache is full"], origin, mock_store)

            assert len(result) == 1
            assert result[0].content == "the cache is full"
            assert result[0].confidence == 0.8
            assert "infra.cache" in result[0].tags
            mock_store.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplicates_identical_candidates(self, mock_store, mock_model, origin):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            agent._model = mock_model

            # make encode return identical embeddings for duplicates
            def encode_identical(texts, **kwargs):
                return np.array([[0.5] * 384 for _ in texts])
            mock_model.encode = MagicMock(side_effect=encode_identical)

            # first candidate creates, second should be duplicate
            async def create_belief(b):
                b.id = uuid4()
                return b
            mock_store.create = AsyncMock(side_effect=create_belief)

            # simulate existing belief with same embedding
            existing = MockBelief(content="the cache is full", confidence=0.8, origin=origin)
            mock_store.list = AsyncMock(return_value=[existing])

            result = await agent.create_beliefs(
                ["the cache is full", "the cache is full"],
                origin,
                mock_store
            )

            # both should be deduplicated against existing
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_creates_distinct_beliefs(self, mock_store, mock_model, origin):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            agent._model = mock_model

            # return very different embeddings
            call_count = [0]
            def encode_distinct(texts, **kwargs):
                result = []
                for i, _ in enumerate(texts):
                    # each text gets a unique embedding
                    emb = [0.0] * 384
                    emb[call_count[0] + i] = 1.0
                    result.append(emb)
                call_count[0] += len(texts)
                return np.array(result)
            mock_model.encode = MagicMock(side_effect=encode_distinct)

            async def create_belief(b):
                b.id = uuid4()
                return b
            mock_store.create = AsyncMock(side_effect=create_belief)

            result = await agent.create_beliefs(
                ["cache error", "tensor mismatch"],
                origin,
                mock_store
            )

            assert len(result) == 2
            assert result[0].content == "cache error"
            assert result[1].content == "tensor mismatch"

    @pytest.mark.asyncio
    async def test_saves_embedding_if_supported(self, mock_store, mock_model, origin):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            agent._model = mock_model

            # return orthogonal embeddings (no duplicates)
            def encode_orthogonal(texts, **kwargs):
                result = []
                for i, _ in enumerate(texts):
                    emb = [0.0] * 384
                    emb[i] = 1.0
                    result.append(emb)
                return np.array(result)
            mock_model.encode = MagicMock(side_effect=encode_orthogonal)

            async def create_belief(b):
                b.id = uuid4()
                return b
            mock_store.create = AsyncMock(side_effect=create_belief)
            mock_store.save_embedding = AsyncMock()

            result = await agent.create_beliefs(["test belief"], origin, mock_store)

            assert len(result) == 1
            mock_store.save_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_store_without_save_embedding(self, mock_store, mock_model, origin):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            agent._model = mock_model

            def encode_orthogonal(texts, **kwargs):
                result = []
                for i, _ in enumerate(texts):
                    emb = [0.0] * 384
                    emb[i] = 1.0
                    result.append(emb)
                return np.array(result)
            mock_model.encode = MagicMock(side_effect=encode_orthogonal)

            async def create_belief(b):
                b.id = uuid4()
                return b
            mock_store.create = AsyncMock(side_effect=create_belief)

            # remove save_embedding
            if hasattr(mock_store, "save_embedding"):
                delattr(mock_store, "save_embedding")

            # should not raise
            result = await agent.create_beliefs(["test belief"], origin, mock_store)
            assert len(result) == 1


class TestLazyModelLoading:
    """Test lazy model initialization."""

    def test_model_not_loaded_on_init(self):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            assert agent._model is None

    def test_raises_without_sentence_transformers(self):
        with patch.dict("sys.modules", {
            "backend.core.models.belief": MagicMock(Belief=MockBelief, OriginMetadata=MockOriginMetadata),
            "backend.storage.base": MagicMock(),
            "sentence_transformers": None,
        }):
            from backend.agents.belief_creator import BeliefCreatorAgent
            agent = BeliefCreatorAgent()
            agent._model = None

            with patch.dict("sys.modules", {"sentence_transformers": None}):
                with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
                    agent._get_model()
