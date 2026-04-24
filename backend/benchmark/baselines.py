# Author: Bradley R. Kinnard
"""
Baseline memory system interfaces for comparison.
Allows benchmarking ABES against simpler memory approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class BaselineMemory:
    """Simple memory record for baselines."""
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


class BaselineMemorySystem(ABC):
    """Abstract interface for baseline memory systems."""

    @abstractmethod
    async def add(self, content: str, embedding: Optional[list[float]] = None) -> BaselineMemory:
        """Add a memory."""
        ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[BaselineMemory]:
        """Search by embedding similarity."""
        ...

    @abstractmethod
    async def get_all(self) -> list[BaselineMemory]:
        """Get all memories."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Get memory count."""
        ...


class FIFOMemory(BaselineMemorySystem):
    """
    Simple FIFO (first-in, first-out) memory.
    Oldest memories are evicted when capacity is reached.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._memories: list[BaselineMemory] = []

    async def add(self, content: str, embedding: Optional[list[float]] = None) -> BaselineMemory:
        memory = BaselineMemory(content=content, embedding=embedding)
        self._memories.append(memory)

        # evict oldest if over capacity
        while len(self._memories) > self._capacity:
            self._memories.pop(0)

        return memory

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[BaselineMemory]:
        # simple cosine similarity search
        import numpy as np

        query = np.array(query_embedding)
        scored = []

        for mem in self._memories:
            if mem.embedding:
                emb = np.array(mem.embedding)
                sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
                scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    async def get_all(self) -> list[BaselineMemory]:
        return self._memories.copy()

    async def clear(self) -> None:
        self._memories.clear()

    def count(self) -> int:
        return len(self._memories)


class LRUMemory(BaselineMemorySystem):
    """
    LRU (least recently used) memory.
    Memories accessed more recently are retained.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._memories: dict[UUID, BaselineMemory] = {}
        self._access_order: list[UUID] = []

    async def add(self, content: str, embedding: Optional[list[float]] = None) -> BaselineMemory:
        memory = BaselineMemory(content=content, embedding=embedding)
        self._memories[memory.id] = memory
        self._access_order.append(memory.id)

        # evict LRU if over capacity
        while len(self._memories) > self._capacity:
            oldest_id = self._access_order.pop(0)
            if oldest_id in self._memories:
                del self._memories[oldest_id]

        return memory

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[BaselineMemory]:
        import numpy as np

        query = np.array(query_embedding)
        scored = []

        for mem in self._memories.values():
            if mem.embedding:
                emb = np.array(mem.embedding)
                sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
                scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        # update access order for returned items
        for _, mem in scored[:top_k]:
            if mem.id in self._access_order:
                self._access_order.remove(mem.id)
            self._access_order.append(mem.id)

        return [m for _, m in scored[:top_k]]

    async def get_all(self) -> list[BaselineMemory]:
        return list(self._memories.values())

    async def clear(self) -> None:
        self._memories.clear()
        self._access_order.clear()

    def count(self) -> int:
        return len(self._memories)


class VectorStoreMemory(BaselineMemorySystem):
    """
    Simple vector store memory with no eviction.
    Pure embedding-based retrieval.
    """

    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._memories: list[BaselineMemory] = []

    async def add(self, content: str, embedding: Optional[list[float]] = None) -> BaselineMemory:
        if len(self._memories) >= self._max_size:
            raise ValueError(f"Vector store at capacity: {self._max_size}")

        memory = BaselineMemory(content=content, embedding=embedding)
        self._memories.append(memory)
        return memory

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[BaselineMemory]:
        import numpy as np

        query = np.array(query_embedding)
        scored = []

        for mem in self._memories:
            if mem.embedding:
                emb = np.array(mem.embedding)
                sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
                scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    async def get_all(self) -> list[BaselineMemory]:
        return self._memories.copy()

    async def clear(self) -> None:
        self._memories.clear()

    def count(self) -> int:
        return len(self._memories)


# registry of baseline systems
BASELINE_REGISTRY: dict[str, type[BaselineMemorySystem]] = {
    "fifo": FIFOMemory,
    "lru": LRUMemory,
    "vector": VectorStoreMemory,
}


def get_baseline(name: str, **kwargs) -> BaselineMemorySystem:
    """Get a baseline memory system by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](**kwargs)


__all__ = [
    "BaselineMemory",
    "BaselineMemorySystem",
    "FIFOMemory",
    "LRUMemory",
    "VectorStoreMemory",
    "BASELINE_REGISTRY",
    "get_baseline",
]
