"""
Contradiction detection and tension computation.
Finds semantically similar belief pairs that may conflict.
"""

import re
from typing import List
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer

from ...storage import Belief


def _contains_word(text: str, word: str) -> bool:
    """Check if word appears as a whole word in text."""
    return re.search(rf"\b{word}\b", text) is not None


def _is_likely_negation(text1: str, text2: str) -> bool:
    """
    Cheap negation check. Looks for basic "X" vs "not X" style patterns.
    Good enough for a first pass; will get replaced by LLM scoring later.
    """
    # normalize texts
    t1_lower = text1.lower()
    t2_lower = text2.lower()

    # negation keywords
    negations = [
        "not",
        "no",
        "never",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
    ]

    # count negations in each text
    neg_count_1 = sum(1 for neg in negations if f" {neg} " in f" {t1_lower} ")
    neg_count_2 = sum(1 for neg in negations if f" {neg} " in f" {t2_lower} ")

    # if one has negations and the other doesn't, possible contradiction
    if (neg_count_1 > 0) != (neg_count_2 > 0):
        return True

    # check for explicit opposites (basic cases)
    opposites = [
        ("true", "false"),
        ("yes", "no"),
        ("always", "never"),
        ("good", "bad"),
        ("like", "dislike"),
        ("love", "hate"),
    ]

    for word1, word2 in opposites:
        if (_contains_word(t1_lower, word1) and _contains_word(t2_lower, word2)) or (
            _contains_word(t1_lower, word2) and _contains_word(t2_lower, word1)
        ):
            return True

    return False


def compute_tensions(
    beliefs: List[Belief],
    embedding_model: SentenceTransformer,
    similarity_threshold: float = 0.7,
) -> dict[UUID, float]:
    """
    Compute tension by embedding beliefs, finding similar pairs,
    and bumping scores when the texts look like opposites.
    """
    if not beliefs:
        return {}

    # embed all beliefs
    contents = [b.content for b in beliefs]
    embeddings = embedding_model.encode(contents, convert_to_numpy=True)

    # init tension scores
    tension_scores: dict[UUID, float] = {b.id: 0.0 for b in beliefs}

    # pairwise comparison - O(n^2) for now; fine for small belief sets
    n = len(beliefs)
    for i in range(n):
        for j in range(i + 1, n):  # only upper triangle, avoid duplicates
            b1, b2 = beliefs[i], beliefs[j]

            # compute cosine similarity with safe denominator
            denom = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            if denom == 0:
                continue
            similarity = float(np.dot(embeddings[i], embeddings[j]) / denom)

            # if similar enough, check for contradiction
            if similarity > similarity_threshold:
                # check for negation/opposition
                is_contradiction = _is_likely_negation(b1.content, b2.content)

                if is_contradiction:
                    # high similarity + contradiction = high tension
                    tension = similarity  # could weight by confidence difference

                    # add to both beliefs' tension scores
                    tension_scores[b1.id] += tension
                    tension_scores[b2.id] += tension

    return tension_scores


__all__ = ["compute_tensions"]
