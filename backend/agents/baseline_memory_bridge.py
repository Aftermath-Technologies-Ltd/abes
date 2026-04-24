# Author: Bradley R. Kinnard
"""
BaselineMemoryBridgeAgent - interfaces with non-ecological memory systems.
Enables real-time comparison between Belief Ecology and baseline approaches.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Unified result from any memory system."""

    content: str
    score: float
    source: str  # "belief_ecology", "rag", "chat_history"
    metadata: dict = field(default_factory=dict)


class MemoryBackend(Protocol):
    """Protocol for pluggable memory backends."""

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievalResult]: ...

    async def store(self, content: str, metadata: dict) -> None: ...

    def name(self) -> str: ...


class RAGBackend:
    """Simple RAG-style vector retrieval backend."""

    def __init__(self, model_name: str = settings.embedding_model):
        self._model_name = model_name
        self._model = None
        self._documents: list[tuple[str, list[float], dict]] = []

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def name(self) -> str:
        return "rag"

    async def store(self, content: str, metadata: dict) -> None:
        model = self._get_model()
        embedding = model.encode(content, normalize_embeddings=True).tolist()
        self._documents.append((content, embedding, metadata))

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        if not self._documents:
            return []

        import numpy as np

        model = self._get_model()
        query_emb = model.encode(query, normalize_embeddings=True)

        scores = []
        for content, doc_emb, meta in self._documents:
            score = float(np.dot(query_emb, doc_emb))
            scores.append((content, score, meta))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(content=c, score=s, source="rag", metadata=m)
            for c, s, m in scores[:top_k]
        ]


class ChatHistoryBackend:
    """Recency-based chat history retrieval."""

    def __init__(self, max_history: int = 100):
        self._history: list[tuple[str, datetime, dict]] = []
        self._max_history = max_history

    def name(self) -> str:
        return "chat_history"

    async def store(self, content: str, metadata: dict) -> None:
        now = datetime.now(timezone.utc)
        self._history.append((content, now, metadata))
        # trim old history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        # return most recent entries (recency-based)
        recent = self._history[-top_k:]
        recent.reverse()  # newest first

        return [
            RetrievalResult(
                content=content,
                score=1.0 - (i * 0.1),  # decay by position
                source="chat_history",
                metadata={**meta, "timestamp": ts.isoformat()},
            )
            for i, (content, ts, meta) in enumerate(recent)
        ]


class BeliefEcologyBackend:
    """Adapter to treat belief store as a memory backend."""

    def __init__(self, beliefs: Optional[list[Belief]] = None):
        self._beliefs = beliefs or []
        self._model = None
        self._model_name = settings.embedding_model

    def set_beliefs(self, beliefs: list[Belief]) -> None:
        self._beliefs = beliefs

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def name(self) -> str:
        return "belief_ecology"

    async def store(self, content: str, metadata: dict) -> None:
        # beliefs are managed externally; this is read-only
        pass

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        if not self._beliefs:
            return []

        import numpy as np

        model = self._get_model()
        query_emb = model.encode(query, normalize_embeddings=True)

        scored = []
        for belief in self._beliefs:
            belief_emb = model.encode(belief.content, normalize_embeddings=True)
            sim = float(np.dot(query_emb, belief_emb))
            scored.append((belief, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                content=b.content,
                score=s,
                source="belief_ecology",
                metadata={
                    "belief_id": str(b.id),
                    "confidence": b.confidence,
                    "tension": b.tension,
                    "status": b.status.value,
                },
            )
            for b, s in scored[:top_k]
        ]


@dataclass
class ComparisonResult:
    """Result of comparing retrieval across backends."""

    query: str
    results_by_source: dict[str, list[RetrievalResult]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def overlap_score(self, source_a: str, source_b: str) -> float:
        """Jaccard overlap of top results between two sources."""
        results_a = self.results_by_source.get(source_a, [])
        results_b = self.results_by_source.get(source_b, [])

        contents_a = {r.content for r in results_a}
        contents_b = {r.content for r in results_b}

        if not contents_a or not contents_b:
            return 0.0

        intersection = contents_a & contents_b
        union = contents_a | contents_b
        return len(intersection) / len(union)


class BaselineMemoryBridgeAgent:
    """
    Bridges Belief Ecology with baseline memory systems (spec 4.1 agent #9).
    Enables real-time comparison for experiments and benchmarks.
    """

    def __init__(self):
        self._backends: dict[str, MemoryBackend] = {}
        self._comparison_log: list[ComparisonResult] = []

    def register_backend(self, backend: MemoryBackend) -> None:
        """Register a memory backend for comparison."""
        self._backends[backend.name()] = backend
        logger.info(f"registered backend: {backend.name()}")

    def unregister_backend(self, name: str) -> None:
        """Remove a backend."""
        self._backends.pop(name, None)

    def list_backends(self) -> list[str]:
        """List registered backend names."""
        return list(self._backends.keys())

    async def retrieve_from(
        self, backend_name: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve from a specific backend."""
        backend = self._backends.get(backend_name)
        if backend is None:
            raise ValueError(f"unknown backend: {backend_name}")
        return await backend.retrieve(query, top_k)

    async def retrieve_all(
        self, query: str, top_k: int = 5
    ) -> dict[str, list[RetrievalResult]]:
        """Retrieve from all registered backends."""
        results = {}
        for name, backend in self._backends.items():
            try:
                results[name] = await backend.retrieve(query, top_k)
            except Exception as e:
                logger.warning(f"backend {name} failed: {e}")
                results[name] = []
        return results

    async def compare(self, query: str, top_k: int = 5) -> ComparisonResult:
        """
        Run retrieval across all backends and produce a comparison result.
        Logs result for later analysis.
        """
        results = await self.retrieve_all(query, top_k)
        comparison = ComparisonResult(query=query, results_by_source=results)
        self._comparison_log.append(comparison)

        # log overlap between belief ecology and each baseline
        be_name = "belief_ecology"
        if be_name in results:
            for name in results:
                if name != be_name:
                    overlap = comparison.overlap_score(be_name, name)
                    logger.debug(f"overlap {be_name} vs {name}: {overlap:.2f}")

        return comparison

    async def store_to_all(self, content: str, metadata: dict) -> None:
        """Store content to all backends that support it."""
        for name, backend in self._backends.items():
            try:
                await backend.store(content, metadata)
            except Exception as e:
                logger.warning(f"store to {name} failed: {e}")

    def get_comparison_log(self) -> list[ComparisonResult]:
        """Return logged comparisons for analysis."""
        return self._comparison_log.copy()

    def clear_comparison_log(self) -> None:
        """Clear the comparison log."""
        self._comparison_log.clear()


__all__ = [
    "BaselineMemoryBridgeAgent",
    "RetrievalResult",
    "ComparisonResult",
    "MemoryBackend",
    "RAGBackend",
    "ChatHistoryBackend",
    "BeliefEcologyBackend",
]
