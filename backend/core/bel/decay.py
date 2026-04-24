"""
Belief confidence decay functions.
Handles time-based confidence degradation and status transitions.
"""

import math
from typing import List

from ...storage import Belief, BeliefStatus
from ..config import ABESSettings
from ..models.belief import utcnow


async def apply_decay(beliefs: List[Belief], settings: ABESSettings) -> List[Belief]:
    """
    Apply exponential confidence decay to beliefs based on time since reinforcement.

    Decay formula: confidence_new = confidence_old * exp(-λ * Δt)
    where:
    - λ (lambda) is the decay rate (default 0.05/hour)
    - Δt is hours since last reinforcement

    Status transitions:
    - confidence < 0.5 → Decaying
    - confidence < 0.1 → Deprecated
    """
    now = utcnow()
    decay_rate = 0.05  # TODO: pull from settings.decay_rate

    for b in beliefs:
        # skip already deprecated beliefs
        if b.status == BeliefStatus.Deprecated:
            continue

        # compute time delta
        delta = now - b.origin.last_reinforced
        hours = delta.total_seconds() / 3600.0

        # exponential decay
        b.confidence *= math.exp(-decay_rate * hours)

        # status transitions based on confidence thresholds
        if b.confidence < 0.1:
            b.status = BeliefStatus.Deprecated
        elif b.confidence < 0.5 and b.status == BeliefStatus.Active:
            b.status = BeliefStatus.Decaying

        # update timestamp
        b.updated_at = now

    return beliefs


__all__ = ["apply_decay"]
