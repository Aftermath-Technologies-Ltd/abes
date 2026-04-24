# Author: Bradley R. Kinnard
"""
Semantic similarity scorer — replaces keyword matching with embedding cosine.

Uses the same all-MiniLM-L6-v2 model already in ABES for belief similarity.
Vocabulary-agnostic: doesn't care whether the LLM says "fridge," "refrigerator,"
or "icebox" — the semantic content is what gets scored.

References:
  - BERTScore (Zhang et al. 2019): https://arxiv.org/abs/1904.09675
  - SemScore (Aynetdinov & Akbik 2024): https://arxiv.org/abs/2401.17072
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class ScoringResult:
    """Outcome of a single semantic evaluation."""
    passed: bool
    similarity: float          # cosine(response, gold_answer)
    forbidden_max: float       # max cosine to any forbidden answer, 0.0 if none
    gold_answer: str
    details: list[str] = field(default_factory=list)


class SemanticScorer:
    """Scores LLM responses against gold answers via embedding cosine similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = _load_model(model_name)

    # ---- public API ----

    def score(self, response: str, gold_answer: str) -> float:
        """Cosine similarity between response and gold answer. [0, 1]."""
        if not response.strip() or not gold_answer.strip():
            return 0.0
        vecs = self._model.encode([response, gold_answer], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))

    def score_forbidden(self, response: str, forbidden: list[str]) -> float:
        """Max cosine to any forbidden semantic. 0.0 if list is empty."""
        if not forbidden or not response.strip():
            return 0.0
        texts = [response] + forbidden
        vecs = self._model.encode(texts, normalize_embeddings=True)
        resp_vec = vecs[0]
        sims = [float(np.dot(resp_vec, vecs[i + 1])) for i in range(len(forbidden))]
        return max(sims)

    def check(
        self,
        response: str,
        gold_answer: str,
        threshold: float = 0.70,
        forbidden: list[str] | None = None,
        forbidden_threshold: float = 0.60,
    ) -> ScoringResult:
        """Full evaluation: gold similarity + forbidden check."""
        sim = self.score(response, gold_answer)
        forb_max = self.score_forbidden(response, forbidden or [])

        details: list[str] = []
        passed = True

        if sim < threshold:
            passed = False
            details.append(f"similarity {sim:.3f} < threshold {threshold}")

        if forb_max > forbidden_threshold:
            passed = False
            details.append(f"forbidden similarity {forb_max:.3f} > threshold {forbidden_threshold}")

        return ScoringResult(
            passed=passed,
            similarity=sim,
            forbidden_max=forb_max,
            gold_answer=gold_answer,
            details=details,
        )

    # ---- batch operations ----

    def batch_embeddings(self, texts: list[str]) -> np.ndarray:
        """Pre-compute embeddings for a batch. Returns (N, dim) normalized."""
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Singleton model load. Reuses across all SemanticScorer instances."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def get_scorer() -> SemanticScorer:
    """Convenience factory."""
    return SemanticScorer()
