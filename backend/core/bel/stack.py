# Author: Bradley R. Kinnard
"""
Belief Stack - selects the active reasoning set (~50 beliefs) from the full ecology.
Combines recency, goal relevance, salience, and graph spread to pick
the beliefs that matter RIGHT NOW for a given context.
"""

import logging
from typing import List, Optional
from uuid import UUID

import numpy as np

from ..config import settings
from ..models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)

# weights for stack scoring
_DEFAULT_STACK_WEIGHTS = {
    "salience": 0.35,
    "relevance": 0.30,
    "recency": 0.20,
    "graph_spread": 0.15,
}


def _recency_score(belief: Belief, max_age_hours: float = 168.0) -> float:
    """0-1 score, newer = higher. Caps at max_age_hours (default 1 week)."""
    from ..models.belief import utcnow

    age_seconds = (utcnow() - belief.updated_at).total_seconds()
    age_hours = max(0.0, age_seconds / 3600.0)
    if age_hours >= max_age_hours:
        return 0.0
    return 1.0 - (age_hours / max_age_hours)


def _graph_spread_score(belief: Belief, all_ids: set[UUID]) -> float:
    """How connected is this belief? More links to active beliefs = higher spread."""
    if not belief.links:
        return 0.0
    active_links = sum(1 for lnk in belief.links if lnk.target_id in all_ids)
    # normalize by a reasonable cap (10 links = max spread)
    return min(1.0, active_links / 10.0)


def select_belief_stack(
    beliefs: List[Belief],
    context_relevance: dict[UUID, float] | None = None,
    stack_size: int | None = None,
    weights: dict[str, float] | None = None,
    required_ids: set[UUID] | None = None,
) -> List[Belief]:
    """
    Select the top-N beliefs for active reasoning.

    Args:
        beliefs: full pool of candidate beliefs (should be Active/Decaying)
        context_relevance: pre-computed relevance scores keyed by belief ID
        stack_size: how many to select (defaults to settings.belief_stack_size)
        weights: scoring weights override
        required_ids: force-include these beliefs regardless of score

    Returns:
        sorted list of beliefs (highest stack-score first), capped at stack_size
    """
    size = stack_size or settings.belief_stack_size
    w = weights or _DEFAULT_STACK_WEIGHTS
    context_relevance = context_relevance or {}
    required_ids = required_ids or set()

    # only consider active/decaying beliefs for the stack
    eligible = [
        b for b in beliefs
        if b.status in (BeliefStatus.Active, BeliefStatus.Decaying)
    ]

    if not eligible:
        return []

    all_ids = {b.id for b in eligible}

    scored: list[tuple[Belief, float]] = []
    for b in eligible:
        sal = b.salience * w.get("salience", 0.0)
        rel = context_relevance.get(b.id, 0.0) * w.get("relevance", 0.0)
        rec = _recency_score(b) * w.get("recency", 0.0)
        spread = _graph_spread_score(b, all_ids) * w.get("graph_spread", 0.0)
        total = sal + rel + rec + spread
        scored.append((b, total))

    scored.sort(key=lambda x: x[1], reverse=True)

    # build result: required beliefs first, then top-scored
    result_map: dict[UUID, Belief] = {}
    for b in eligible:
        if b.id in required_ids:
            result_map[b.id] = b

    for b, _ in scored:
        if len(result_map) >= size:
            break
        if b.id not in result_map:
            result_map[b.id] = b

    # re-sort final set by score
    final_ids = set(result_map.keys())
    final_scored = [(b, s) for b, s in scored if b.id in final_ids]
    # add any required that weren't in scored
    scored_ids = {b.id for b, _ in final_scored}
    for bid, b in result_map.items():
        if bid not in scored_ids:
            final_scored.append((b, 0.0))

    final_scored.sort(key=lambda x: x[1], reverse=True)
    return [b for b, _ in final_scored]


def compete_for_attention(
    beliefs: List[Belief],
    stack_size: int | None = None,
) -> tuple[List[Belief], List[Belief]]:
    """
    Competition phase: beliefs fight for limited attention slots.
    Winners stay active, losers get hibernated.

    Returns:
        (winners, losers) - losers should be hibernated by caller
    """
    size = stack_size or settings.belief_stack_size
    # only active beliefs compete
    active = [b for b in beliefs if b.status == BeliefStatus.Active]

    if len(active) <= size:
        return active, []

    # rank by salience * confidence (energy × strength)
    ranked = sorted(active, key=lambda b: b.salience * b.confidence, reverse=True)
    winners = ranked[:size]
    losers = ranked[size:]

    logger.info(
        f"competition: {len(winners)} winners, {len(losers)} hibernated"
    )
    return winners, losers


__all__ = ["select_belief_stack", "compete_for_attention"]
