# Author: Bradley R. Kinnard
"""Tests for BaselineMemoryBridgeAgent."""

import pytest
from datetime import datetime, timezone

from backend.agents.baseline_memory_bridge import (
    BaselineMemoryBridgeAgent,
    RAGBackend,
    ChatHistoryBackend,
    BeliefEcologyBackend,
    RetrievalResult,
    ComparisonResult,
)
from backend.core.models.belief import Belief, OriginMetadata


def _make_belief(content: str, confidence: float = 0.8) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        origin=OriginMetadata(source="test"),
    )


class TestRetrievalResult:
    def test_create_result(self):
        result = RetrievalResult(
            content="test",
            score=0.9,
            source="rag",
            metadata={"key": "value"},
        )
        assert result.content == "test"
        assert result.score == 0.9
        assert result.source == "rag"


class TestComparisonResult:
    def test_overlap_empty(self):
        comp = ComparisonResult(
            query="test",
            results_by_source={"a": [], "b": []},
        )
        assert comp.overlap_score("a", "b") == 0.0

    def test_overlap_identical(self):
        results = [RetrievalResult(content="x", score=1.0, source="a")]
        comp = ComparisonResult(
            query="test",
            results_by_source={"a": results, "b": results},
        )
        assert comp.overlap_score("a", "b") == 1.0

    def test_overlap_partial(self):
        results_a = [
            RetrievalResult(content="x", score=1.0, source="a"),
            RetrievalResult(content="y", score=0.9, source="a"),
        ]
        results_b = [
            RetrievalResult(content="x", score=1.0, source="b"),
            RetrievalResult(content="z", score=0.9, source="b"),
        ]
        comp = ComparisonResult(
            query="test",
            results_by_source={"a": results_a, "b": results_b},
        )
        # intersection = {x}, union = {x, y, z} -> 1/3
        assert abs(comp.overlap_score("a", "b") - 1 / 3) < 0.01

    def test_overlap_missing_source(self):
        comp = ComparisonResult(query="test", results_by_source={})
        assert comp.overlap_score("a", "b") == 0.0


class TestChatHistoryBackend:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        backend = ChatHistoryBackend()
        await backend.store("message 1", {"role": "user"})
        await backend.store("message 2", {"role": "assistant"})
        await backend.store("message 3", {"role": "user"})

        results = await backend.retrieve("query", top_k=2)
        assert len(results) == 2
        # most recent first
        assert results[0].content == "message 3"
        assert results[1].content == "message 2"

    @pytest.mark.asyncio
    async def test_recency_score_decay(self):
        backend = ChatHistoryBackend()
        for i in range(5):
            await backend.store(f"msg {i}", {})

        results = await backend.retrieve("query", top_k=5)
        # scores should decrease
        for i in range(len(results) - 1):
            assert results[i].score > results[i + 1].score

    @pytest.mark.asyncio
    async def test_max_history_limit(self):
        backend = ChatHistoryBackend(max_history=3)
        for i in range(5):
            await backend.store(f"msg {i}", {})

        results = await backend.retrieve("query", top_k=10)
        assert len(results) == 3
        # should have kept the last 3
        contents = {r.content for r in results}
        assert "msg 2" in contents
        assert "msg 3" in contents
        assert "msg 4" in contents

    def test_name(self):
        backend = ChatHistoryBackend()
        assert backend.name() == "chat_history"


class TestBaselineMemoryBridgeAgent:
    def test_register_backend(self):
        agent = BaselineMemoryBridgeAgent()
        backend = ChatHistoryBackend()
        agent.register_backend(backend)
        assert "chat_history" in agent.list_backends()

    def test_unregister_backend(self):
        agent = BaselineMemoryBridgeAgent()
        backend = ChatHistoryBackend()
        agent.register_backend(backend)
        agent.unregister_backend("chat_history")
        assert "chat_history" not in agent.list_backends()

    @pytest.mark.asyncio
    async def test_retrieve_from(self):
        agent = BaselineMemoryBridgeAgent()
        backend = ChatHistoryBackend()
        await backend.store("hello world", {})
        agent.register_backend(backend)

        results = await agent.retrieve_from("chat_history", "test")
        assert len(results) == 1
        assert results[0].content == "hello world"

    @pytest.mark.asyncio
    async def test_retrieve_from_unknown_raises(self):
        agent = BaselineMemoryBridgeAgent()
        with pytest.raises(ValueError):
            await agent.retrieve_from("nonexistent", "query")

    @pytest.mark.asyncio
    async def test_retrieve_all(self):
        agent = BaselineMemoryBridgeAgent()
        chat = ChatHistoryBackend()
        await chat.store("chat msg", {})
        agent.register_backend(chat)

        results = await agent.retrieve_all("query")
        assert "chat_history" in results
        assert len(results["chat_history"]) == 1

    @pytest.mark.asyncio
    async def test_compare(self):
        agent = BaselineMemoryBridgeAgent()
        chat = ChatHistoryBackend()
        await chat.store("msg", {})
        agent.register_backend(chat)

        comparison = await agent.compare("query")
        assert comparison.query == "query"
        assert "chat_history" in comparison.results_by_source

    @pytest.mark.asyncio
    async def test_comparison_log(self):
        agent = BaselineMemoryBridgeAgent()
        chat = ChatHistoryBackend()
        agent.register_backend(chat)

        await agent.compare("q1")
        await agent.compare("q2")

        log = agent.get_comparison_log()
        assert len(log) == 2
        assert log[0].query == "q1"
        assert log[1].query == "q2"

    def test_clear_comparison_log(self):
        agent = BaselineMemoryBridgeAgent()
        agent._comparison_log.append(
            ComparisonResult(query="test", results_by_source={})
        )
        agent.clear_comparison_log()
        assert len(agent.get_comparison_log()) == 0

    @pytest.mark.asyncio
    async def test_store_to_all(self):
        agent = BaselineMemoryBridgeAgent()
        chat = ChatHistoryBackend()
        agent.register_backend(chat)

        await agent.store_to_all("new content", {"source": "test"})

        results = await chat.retrieve("query")
        assert any(r.content == "new content" for r in results)


class TestBeliefEcologyBackend:
    def test_name(self):
        backend = BeliefEcologyBackend()
        assert backend.name() == "belief_ecology"

    def test_set_beliefs(self):
        backend = BeliefEcologyBackend()
        beliefs = [_make_belief("test")]
        backend.set_beliefs(beliefs)
        assert backend._beliefs == beliefs

    @pytest.mark.asyncio
    async def test_store_is_noop(self):
        backend = BeliefEcologyBackend()
        await backend.store("content", {})
        # should not raise, but doesn't store anything
        assert backend._beliefs == []

    @pytest.mark.asyncio
    async def test_retrieve_empty(self):
        backend = BeliefEcologyBackend()
        results = await backend.retrieve("query")
        assert results == []


class TestRAGBackend:
    def test_name(self):
        backend = RAGBackend()
        assert backend.name() == "rag"

    @pytest.mark.asyncio
    async def test_retrieve_empty(self):
        backend = RAGBackend()
        results = await backend.retrieve("query")
        assert results == []
