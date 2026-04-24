# Author: Bradley R. Kinnard
"""
RelevanceCuratorAgent - computes relevance scores and ranks beliefs against context.
Uses embedding similarity and weighted ranking formula from spec 3.4.5 and 3.4.6.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import numpy as np

from ..core.config import settings
from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


@dataclass
class RankedBelief:
    """Belief with computed scores for ranking."""

    belief: Belief
    relevance: float
    recency: float
    rank_score: float


def _hours_since(dt: datetime) -> float:
    """Hours elapsed since a given datetime."""
    now = datetime.now(timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 3600


class RelevanceCuratorAgent:
    """
    Computes relevance and ranks beliefs against context (spec 3.4.5, 3.4.6).
    Uses sentence-transformers for embedding similarity.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        relevance_threshold: float = settings.relevance_threshold_min,
        recency_window_hours: int = settings.recency_window_hours,
        weights: Optional[dict[str, float]] = None,
    ):
        self._model_name = model_name
        self._model = None
        self._relevance_threshold = relevance_threshold
        self._recency_window = recency_window_hours

        # default weights from spec 3.4.6
        self._weights = weights or settings.ranking_weights.copy()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _compute_relevance(
        self, belief_embeddings: np.ndarray, context_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Cosine similarity between context and each belief embedding.
        Both assumed L2-normalized (dot product = cosine sim).
        """
        # context_embedding shape: (dim,) -> expand for batch dot
        return belief_embeddings @ context_embedding

    def _compute_recency(self, belief: Belief) -> float:
        """
        Recency score per spec 3.4.6:
        recency = max(0, 1 - (hours_since_reinforced / recency_window))
        """
        hours = _hours_since(belief.origin.last_reinforced)
        recency = max(0.0, 1.0 - (hours / self._recency_window))
        return recency

    def _compute_rank_score(
        self, relevance: float, confidence: float, recency: float, tension: float
    ) -> float:
        """
        Weighted combination per spec 3.4.6:
        rank = w_rel*relevance + w_conf*confidence + w_rec*recency - w_ten*tension
        """
        w = self._weights
        return (
            w.get("relevance", 0.4) * relevance
            + w.get("confidence", 0.3) * confidence
            + w.get("recency", 0.2) * recency
            - w.get("tension", 0.1) * tension
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update ranking weights. For RL tuning.
        Validates bounds from config.
        """
        lo, hi = settings.ranking_weight_bounds
        for key, val in new_weights.items():
            if key in self._weights:
                clamped = max(lo, min(hi, val))
                self._weights[key] = clamped

    async def compute_relevance_scores(
        self,
        beliefs: list[Belief],
        context: str,
        stored_embeddings: Optional[dict[UUID, list[float]]] = None,
    ) -> dict[UUID, float]:
        """
        Compute relevance scores for all beliefs against context.

        Args:
            beliefs: Beliefs to score
            context: Current context string
            stored_embeddings: Pre-computed embeddings keyed by belief ID

        Returns:
            Dict mapping belief_id -> relevance score
        """
        if not beliefs or not context.strip():
            return {}

        model = self._get_model()

        # encode context
        context_emb = model.encode(
            context, convert_to_numpy=True, normalize_embeddings=True
        )

        # get or compute belief embeddings
        belief_embs = []
        for b in beliefs:
            if stored_embeddings and b.id in stored_embeddings:
                emb = np.array(stored_embeddings[b.id])
            else:
                emb = model.encode(
                    b.content, convert_to_numpy=True, normalize_embeddings=True
                )
            belief_embs.append(emb)

        belief_embs = np.array(belief_embs)
        relevance_scores = self._compute_relevance(belief_embs, context_emb)

        return {b.id: float(relevance_scores[i]) for i, b in enumerate(beliefs)}

    async def rank_beliefs(
        self,
        beliefs: list[Belief],
        context: str,
        stored_embeddings: Optional[dict[UUID, list[float]]] = None,
        tension_map: Optional[dict[UUID, float]] = None,
    ) -> list[RankedBelief]:
        """
        Rank beliefs by weighted score formula (spec 3.4.6).
        Filters out beliefs below relevance threshold.

        Args:
            beliefs: Beliefs to rank
            context: Current context string
            stored_embeddings: Pre-computed embeddings
            tension_map: Pre-computed tensions (uses belief.tension if missing)

        Returns:
            Sorted list of RankedBelief, highest rank first
        """
        if not beliefs:
            return []

        relevance_map = await self.compute_relevance_scores(
            beliefs, context, stored_embeddings
        )

        ranked = []
        for belief in beliefs:
            relevance = relevance_map.get(belief.id, 0.0)

            # filter below threshold per spec 3.4.5
            if relevance < self._relevance_threshold:
                continue

            recency = self._compute_recency(belief)
            tension = tension_map.get(belief.id, belief.tension) if tension_map else belief.tension

            rank_score = self._compute_rank_score(
                relevance=relevance,
                confidence=belief.confidence,
                recency=recency,
                tension=tension,
            )

            ranked.append(
                RankedBelief(
                    belief=belief,
                    relevance=relevance,
                    recency=recency,
                    rank_score=rank_score,
                )
            )

        # sort descending by rank_score
        ranked.sort(key=lambda r: r.rank_score, reverse=True)

        if ranked:
            logger.debug(
                f"ranked {len(ranked)} beliefs, top score: {ranked[0].rank_score:.3f}"
            )

        return ranked

    async def get_top_beliefs(
        self,
        beliefs: list[Belief],
        context: str,
        top_k: int = 10,
        stored_embeddings: Optional[dict[UUID, list[float]]] = None,
        tension_map: Optional[dict[UUID, float]] = None,
    ) -> list[Belief]:
        """
        Convenience method: return top_k beliefs by rank.
        """
        ranked = await self.rank_beliefs(
            beliefs, context, stored_embeddings, tension_map
        )
        return [r.belief for r in ranked[:top_k]]


__all__ = ["RelevanceCuratorAgent", "RankedBelief"]
