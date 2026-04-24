# Author: Bradley R. Kinnard
"""
DecayControllerAgent - applies time-based confidence + salience decay, manages status transitions.
Implements spec 3.4.1 (decay formula), 3.4.2 (status thresholds), and salience half-life dynamics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class DecayEvent:
    """Record of decay applied to a belief."""

    belief_id: UUID
    old_confidence: float
    new_confidence: float
    old_salience: float
    new_salience: float
    old_status: BeliefStatus
    new_status: BeliefStatus
    hours_elapsed: float
    decay_rate_used: float
    timestamp: datetime


def _hours_since(dt: datetime) -> float:
    """Hours elapsed since a given datetime."""
    now = datetime.now(timezone.utc)
    delta = now - dt
    return max(0.0, delta.total_seconds() / 3600)


class DecayControllerAgent:
    """
    Applies exponential decay to belief confidence (spec 3.4.1).
    Manages status transitions per spec 3.4.2.
    """

    def __init__(
        self,
        decay_rate: float = settings.decay_rate,
        threshold_decaying: float = settings.confidence_threshold_decaying,
        threshold_deprecated: float = settings.confidence_threshold_deprecated,
        stale_days: int = settings.stale_days_deprecated,
        dormancy_salience_threshold: float = settings.dormancy_salience_threshold,
    ):
        self._decay_rate = decay_rate
        self._threshold_decaying = threshold_decaying
        self._threshold_deprecated = threshold_deprecated
        self._stale_days = stale_days
        self._dormancy_threshold = dormancy_salience_threshold

        # per-cluster or per-tag overrides
        self._rate_overrides: dict[str, float] = {}

    def set_decay_rate(self, rate: float) -> None:
        """Update global decay rate. For RL tuning."""
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {rate}")
        self._decay_rate = rate

    def set_override(self, key: str, rate: float) -> None:
        """Set decay rate override for a cluster ID or tag."""
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"override rate must be in (0, 1], got {rate}")
        self._rate_overrides[key] = rate

    def clear_override(self, key: str) -> None:
        """Remove a decay rate override."""
        self._rate_overrides.pop(key, None)

    def _get_effective_rate(self, belief: Belief) -> float:
        """
        Get decay rate for a belief, checking overrides.
        Priority: cluster_id override > tag override > global rate.
        """
        # cluster override
        if belief.cluster_id:
            cluster_key = str(belief.cluster_id)
            if cluster_key in self._rate_overrides:
                return self._rate_overrides[cluster_key]

        # tag override - take first matching tag
        for tag in belief.tags:
            if tag in self._rate_overrides:
                return self._rate_overrides[tag]

        return self._decay_rate

    def _compute_new_confidence(
        self, current: float, hours_elapsed: float, rate: float
    ) -> float:
        """
        Spec 3.4.1: new_confidence = confidence × (decay_rate ^ hours_elapsed)
        """
        if hours_elapsed <= 0:
            return current
        return current * (rate ** hours_elapsed)

    def _determine_status(
        self, belief: Belief, new_confidence: float
    ) -> BeliefStatus:
        """Apply status transition rules per spec 3.4.2, extended for dormancy."""
        current = belief.status

        # already deprecated stays deprecated
        if current == BeliefStatus.Deprecated:
            return BeliefStatus.Deprecated

        # mutated stays mutated
        if current == BeliefStatus.Mutated:
            return BeliefStatus.Mutated

        # dormancy check: low salience but not dead
        if (
            belief.salience < self._dormancy_threshold
            and new_confidence >= self._threshold_deprecated
            and current not in (BeliefStatus.Dormant,)
        ):
            return BeliefStatus.Dormant

        # dormant beliefs stay dormant unless reawakened externally
        if current == BeliefStatus.Dormant and belief.salience < self._dormancy_threshold:
            return BeliefStatus.Dormant

        # check stale unused beliefs
        age_days = _hours_since(belief.created_at) / 24
        if belief.use_count == 0 and age_days > self._stale_days:
            return BeliefStatus.Deprecated

        # confidence-based transitions
        if new_confidence < self._threshold_deprecated:
            return BeliefStatus.Deprecated

        if new_confidence < self._threshold_decaying:
            return BeliefStatus.Decaying

        return BeliefStatus.Active

    def apply_decay(self, belief: Belief) -> Optional[DecayEvent]:
        """Apply confidence + salience decay to a single belief. Updates in place."""
        # don't decay deprecated or mutated beliefs
        if belief.status in (BeliefStatus.Deprecated, BeliefStatus.Mutated):
            return None

        # decay clock starts from last touch (reinforce or creation, whichever is newer)
        # prevents catastrophic decay when last_reinforced is epoch (never reinforced)
        anchor = max(belief.origin.last_reinforced, belief.created_at)
        hours = _hours_since(anchor)
        if hours <= 0:
            return None

        rate = self._get_effective_rate(belief)
        old_confidence = belief.confidence
        old_salience = belief.salience
        old_status = belief.status

        new_confidence = self._compute_new_confidence(old_confidence, hours, rate)

        # salience decays independently via half-life
        belief.decay_salience(hours)
        new_salience = belief.salience

        new_status = self._determine_status(belief, new_confidence)

        # update belief
        belief.confidence = new_confidence
        belief.status = new_status
        belief.updated_at = datetime.now(timezone.utc)

        # only emit event if something changed meaningfully
        conf_changed = abs(old_confidence - new_confidence) > 0.001
        sal_changed = abs(old_salience - new_salience) > 0.001
        status_changed = old_status != new_status

        if conf_changed or sal_changed or status_changed:
            return DecayEvent(
                belief_id=belief.id,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                old_salience=old_salience,
                new_salience=new_salience,
                old_status=old_status,
                new_status=new_status,
                hours_elapsed=hours,
                decay_rate_used=rate,
                timestamp=datetime.now(timezone.utc),
            )

        return None

    async def process_beliefs(
        self, beliefs: list[Belief]
    ) -> tuple[list[DecayEvent], list[Belief]]:
        """
        Apply decay to all beliefs.

        Returns:
            (events, modified_beliefs) - events for tracking, beliefs that changed
        """
        events = []
        modified = []

        for belief in beliefs:
            event = self.apply_decay(belief)
            if event:
                events.append(event)
                modified.append(belief)

        if events:
            deprecated_count = sum(
                1 for e in events if e.new_status == BeliefStatus.Deprecated
            )
            decaying_count = sum(
                1 for e in events if e.new_status == BeliefStatus.Decaying
            )
            logger.info(
                f"decay: {len(events)} beliefs updated, "
                f"{decaying_count} now decaying, {deprecated_count} deprecated"
            )

        return events, modified

    def estimate_half_life_hours(self, rate: Optional[float] = None) -> float:
        """
        Calculate half-life in hours for a given decay rate.
        Half-life = ln(0.5) / ln(rate)
        """
        r = rate or self._decay_rate
        if r <= 0 or r >= 1:
            return float("inf")
        import math
        return math.log(0.5) / math.log(r)


__all__ = ["DecayControllerAgent", "DecayEvent"]
