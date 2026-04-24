# Author: Bradley R. Kinnard
"""Tests for updated DecayControllerAgent with salience decay and dormancy."""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.agents.decay_controller import DecayControllerAgent, DecayEvent
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.8,
    salience: float = 1.0,
    status: BeliefStatus = BeliefStatus.Active,
    last_reinforced: datetime | None = None,
    created_at: datetime | None = None,
    use_count: int = 1,
    half_life_days: float = 7.0,
    **kwargs,
) -> Belief:
    now = datetime.now(timezone.utc)
    lr = last_reinforced or now
    ca = created_at or now
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        half_life_days=half_life_days,
        status=status,
        origin=OriginMetadata(source="test", last_reinforced=lr),
        use_count=use_count,
        created_at=ca,
        **kwargs,
    )


class TestSalienceDecayInController:
    def test_salience_decays_with_confidence(self):
        """DecayEvent should now include old/new salience."""
        agent = DecayControllerAgent(decay_rate=0.99)
        # belief reinforced 48 hours ago
        past = datetime.now(timezone.utc) - timedelta(hours=48)
        b = _make_belief(
            salience=1.0,
            confidence=0.8,
            last_reinforced=past,
            created_at=past,
            half_life_days=1.0,  # fast half-life for testing
        )
        event = agent.apply_decay(b)
        assert event is not None
        assert event.old_salience == 1.0
        assert event.new_salience < 1.0
        # after 2 half-lives (48h with 1-day half-life), salience ~ 0.25
        assert 0.1 < event.new_salience < 0.35

    def test_salience_zero_hours_no_decay(self):
        """Freshly reinforced belief shouldn't lose salience."""
        agent = DecayControllerAgent()
        b = _make_belief(salience=0.8, confidence=0.7)
        event = agent.apply_decay(b)
        # might have tiny floating-point drift
        assert abs(b.salience - 0.8) < 0.01


class TestDormancyTransition:
    def test_low_salience_triggers_dormancy(self):
        """Belief with salience below threshold transitions to Dormant."""
        agent = DecayControllerAgent(
            decay_rate=0.999,  # slow confidence decay
            dormancy_salience_threshold=0.1,
        )
        past = datetime.now(timezone.utc) - timedelta(hours=200)
        b = _make_belief(
            salience=0.5,
            confidence=0.8,  # high enough to not deprecate
            last_reinforced=past,
            created_at=past,
            half_life_days=1.0,  # will decay salience rapidly
        )
        event = agent.apply_decay(b)
        assert event is not None
        # salience should have decayed below dormancy threshold
        assert b.salience < 0.1
        assert b.status == BeliefStatus.Dormant

    def test_dormant_stays_dormant_without_reawaken(self):
        """Dormant belief stays dormant through decay cycles."""
        agent = DecayControllerAgent(dormancy_salience_threshold=0.1)
        past = datetime.now(timezone.utc) - timedelta(hours=10)
        b = _make_belief(
            status=BeliefStatus.Dormant,
            salience=0.01,
            confidence=0.5,
            last_reinforced=past,
        )
        event = agent.apply_decay(b)
        # should still be dormant (dormant is not deprecated/mutated so it processes)
        assert b.status == BeliefStatus.Dormant

    def test_deprecated_and_mutated_skip_salience_decay(self):
        """Deprecated/mutated beliefs are skipped entirely."""
        agent = DecayControllerAgent()
        past = datetime.now(timezone.utc) - timedelta(hours=100)

        for status in (BeliefStatus.Deprecated, BeliefStatus.Mutated):
            b = _make_belief(salience=0.8, status=status, last_reinforced=past)
            event = agent.apply_decay(b)
            assert event is None
            assert b.salience == 0.8  # unchanged


class TestDecayEventFields:
    def test_event_has_salience_fields(self):
        agent = DecayControllerAgent(decay_rate=0.99)
        past = datetime.now(timezone.utc) - timedelta(hours=24)
        b = _make_belief(salience=0.9, confidence=0.8, last_reinforced=past, created_at=past)
        event = agent.apply_decay(b)
        assert event is not None
        assert hasattr(event, "old_salience")
        assert hasattr(event, "new_salience")
        assert event.old_salience == 0.9
        assert event.new_salience < 0.9


class TestProcessBeliefsWithSalience:
    @pytest.mark.asyncio
    async def test_batch_processes_salience(self):
        agent = DecayControllerAgent(decay_rate=0.99)
        past = datetime.now(timezone.utc) - timedelta(hours=48)
        beliefs = [
            _make_belief(
                content=f"belief {i}",
                salience=0.8,
                confidence=0.7,
                last_reinforced=past,
                created_at=past,
                half_life_days=1.0,
            )
            for i in range(5)
        ]
        events, modified = await agent.process_beliefs(beliefs)
        assert len(events) == 5
        assert all(b.salience < 0.8 for b in modified)
