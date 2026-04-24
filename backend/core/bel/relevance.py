"""
Relevance scoring for beliefs based on context similarity.
Uses embeddings to compute how relevant each belief is to current context.
"""

from typing import List
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer

from ...storage import Belief


def compute_relevance_scores(
    beliefs: List[Belief],
    context: str,
    embedding_model: SentenceTransformer,
) -> dict[UUID, float]:
    """Return a 0–1 relevance score for each belief given the context."""
    if not beliefs or not context.strip():
        return {b.id: 0.0 for b in beliefs}

    # embed context once
    context_embedding = embedding_model.encode(context, convert_to_numpy=True)

    # batch embed all belief contents
    contents = [b.content for b in beliefs]
    belief_embeddings = embedding_model.encode(contents, convert_to_numpy=True)

    # compute cosine similarities
    relevance_scores: dict[UUID, float] = {}
    context_norm = np.linalg.norm(context_embedding)

    for i, b in enumerate(beliefs):
        belief_norm = np.linalg.norm(belief_embeddings[i])
        denom = context_norm * belief_norm

        if denom == 0:
            relevance_scores[b.id] = 0.0
            continue

        similarity = float(np.dot(context_embedding, belief_embeddings[i]) / denom)

        # clamp cosine into [0, 1]; we ignore negative similarity
        relevance_scores[b.id] = max(0.0, min(1.0, similarity))

    return relevance_scores


__all__ = ["compute_relevance_scores"]
