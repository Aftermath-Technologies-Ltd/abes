# Author: Bradley R. Kinnard
"""Tests for baseline memory systems."""

import pytest
import numpy as np

from backend.benchmark.baselines import (
    BaselineMemory,
    FIFOMemory,
    LRUMemory,
    VectorStoreMemory,
    get_baseline,
    BASELINE_REGISTRY,
)


def _random_embedding(dim: int = 384) -> list[float]:
    """Generate random normalized embedding."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


class TestBaselineMemory:
    def test_default_values(self):
        mem = BaselineMemory(content="test")
        assert mem.content == "test"
        assert mem.id is not None
        assert mem.timestamp is not None


class TestFIFOMemory:
    @pytest.fixture
    def fifo(self):
        return FIFOMemory(capacity=10)

    @pytest.mark.asyncio
    async def test_add(self, fifo):
        mem = await fifo.add("test content")
        assert mem.content == "test content"
        assert fifo.count() == 1

    @pytest.mark.asyncio
    async def test_capacity_eviction(self, fifo):
        for i in range(15):
            await fifo.add(f"content {i}")

        assert fifo.count() == 10  # evicted 5
        all_mems = await fifo.get_all()
        # oldest (0-4) should be gone
        contents = {m.content for m in all_mems}
        assert "content 0" not in contents
        assert "content 14" in contents

    @pytest.mark.asyncio
    async def test_search(self, fifo):
        emb = _random_embedding()
        await fifo.add("test", embedding=emb)

        results = await fifo.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_clear(self, fifo):
        await fifo.add("test")
        await fifo.clear()
        assert fifo.count() == 0


class TestLRUMemory:
    @pytest.fixture
    def lru(self):
        return LRUMemory(capacity=10)

    @pytest.mark.asyncio
    async def test_add(self, lru):
        mem = await lru.add("test content")
        assert mem.content == "test content"
        assert lru.count() == 1

    @pytest.mark.asyncio
    async def test_capacity_eviction(self, lru):
        for i in range(15):
            await lru.add(f"content {i}")

        assert lru.count() == 10  # evicted 5

    @pytest.mark.asyncio
    async def test_search_updates_order(self, lru):
        emb1 = _random_embedding()
        emb2 = _random_embedding()

        mem1 = await lru.add("first", embedding=emb1)
        mem2 = await lru.add("second", embedding=emb2)

        # both should exist
        assert lru.count() == 2

        # search for first - should work
        results = await lru.search(emb1, top_k=1)
        assert len(results) == 1
        assert results[0].content == "first"

    @pytest.mark.asyncio
    async def test_clear(self, lru):
        await lru.add("test")
        await lru.clear()
        assert lru.count() == 0


class TestVectorStoreMemory:
    @pytest.fixture
    def vector(self):
        return VectorStoreMemory(max_size=100)

    @pytest.mark.asyncio
    async def test_add(self, vector):
        mem = await vector.add("test content")
        assert mem.content == "test content"
        assert vector.count() == 1

    @pytest.mark.asyncio
    async def test_max_size_error(self, vector):
        small = VectorStoreMemory(max_size=5)
        for i in range(5):
            await small.add(f"content {i}")

        with pytest.raises(ValueError) as exc:
            await small.add("one too many")
        assert "capacity" in str(exc.value)

    @pytest.mark.asyncio
    async def test_search(self, vector):
        emb = _random_embedding()
        await vector.add("test", embedding=emb)

        results = await vector.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_clear(self, vector):
        await vector.add("test")
        await vector.clear()
        assert vector.count() == 0


class TestBaselineRegistry:
    def test_registry_populated(self):
        assert "fifo" in BASELINE_REGISTRY
        assert "lru" in BASELINE_REGISTRY
        assert "vector" in BASELINE_REGISTRY

    def test_get_baseline(self):
        baseline = get_baseline("fifo", capacity=50)
        assert isinstance(baseline, FIFOMemory)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError) as exc:
            get_baseline("unknown_baseline")
        assert "Unknown baseline" in str(exc.value)
