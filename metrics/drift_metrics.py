# Author: Bradley R. Kinnard
"""
Metrics calculation for drift benchmark.

Provides functions to compute belief count, contradiction count,
and entropy over time for comparison between systems.
"""

import math
from collections import Counter
from typing import Any


def compute_belief_entropy(beliefs: list[dict]) -> float:
    """
    Compute Shannon entropy of belief distribution.

    Groups beliefs by content prefix and measures distribution.
    Higher entropy = more diverse beliefs.
    Lower entropy = more concentrated/redundant beliefs.

    Args:
        beliefs: List of belief dicts with 'content' key

    Returns:
        Entropy value in bits
    """
    if not beliefs:
        return 0.0

    # Group by first 3 words (normalized)
    prefixes = []
    for b in beliefs:
        content = b.get("content", "")
        words = content.lower().split()[:3]
        prefixes.append(" ".join(words))

    counts = Counter(prefixes)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 4)


def count_potential_contradictions(beliefs: list[dict]) -> int:
    """
    Count potential contradictions in a belief set.

    Uses simple heuristics (not full semantic analysis):
    - Opposite sentiment words
    - Negation patterns
    - Known antonym pairs

    Args:
        beliefs: List of belief dicts with 'content' key

    Returns:
        Number of potential contradiction pairs detected
    """
    antonym_pairs = {
        ("warm", "cold"), ("hot", "cold"),
        ("like", "hate"), ("love", "hate"),
        ("happy", "sad"), ("good", "bad"),
        ("true", "false"), ("yes", "no"),
        ("always", "never"), ("all", "none"),
        ("big", "small"), ("large", "small"),
        ("fast", "slow"), ("old", "new"),
    }

    contradiction_count = 0
    contents = [b.get("content", "").lower() for b in beliefs]

    for i, c1 in enumerate(contents):
        words1 = set(c1.split())
        for j, c2 in enumerate(contents):
            if j <= i:
                continue
            words2 = set(c2.split())

            # Check for antonym pairs
            for w1, w2 in antonym_pairs:
                if (w1 in words1 and w2 in words2) or (w2 in words1 and w1 in words2):
                    contradiction_count += 1
                    break

            # Check for negation of same statement
            if "not" in words1 or "not" in words2:
                # Simple: if one has "not" and they share >50% words
                shared = len(words1 & words2)
                total = len(words1 | words2)
                if total > 0 and shared / total > 0.5:
                    contradiction_count += 1

    return contradiction_count


def extract_turn_metrics(state: dict) -> dict:
    """
    Extract standard metrics from a system state.

    Args:
        state: State dict from any system

    Returns:
        Dict with belief_count, contradiction_count, belief_entropy
    """
    beliefs = state.get("beliefs", [])

    return {
        "belief_count": len(beliefs),
        "contradiction_count": count_potential_contradictions(beliefs),
        "belief_entropy": compute_belief_entropy(beliefs),
    }


__all__ = [
    "compute_belief_entropy",
    "count_potential_contradictions",
    "extract_turn_metrics",
]
