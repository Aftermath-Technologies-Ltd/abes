# Author: Bradley R. Kinnard
"""Tests for DecayControllerAgent."""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.agents.decay_controller import (
    DecayControllerAgent,
    DecayEvent,
    _hours_since,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.8,
    status: BeliefStatus = BeliefStatus.Active,
    source: str = "test",
    tags: list[str] = None,
    last_reinforced: datetime = None,
    created_at: datetime = None,
    use_count: int = 1,
    cluster_id=None,
) -> Belief:
    now = datetime.now(timezone.utc)
    lr = last_reinforced or now
    ca = created_at or now
    return Belief(
        content=content,
        confidence=confidence,
        status=status,
        origin=OriginMetadata(source=source, last_reinforced=lr),
        tags=tags or [],
        use_count=use_count,
        created_at=ca,
        cluster_id=cluster_id,
    )


class TestHoursSince:
    def test_recent(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=2)
        hours = _hours_since(dt)
        assert 1.9 < hours < 2.1

    def test_future_returns_zero(self):
        dt = datetime.now(timezone.utc) + timedelta(hours=1)
        hours = _hours_since(dt)
        assert hours == 0.0


class TestDecayRate:
    def test_set_valid_rate(self):
        agent = DecayControllerAgent()
        agent.set_decay_rate(0.99)
        assert agent._decay_rate == 0.99

    def test_set_invalid_rate_raises(self):
        agent = DecayControllerAgent()
        with pytest.raises(ValueError):
            agent.set_decay_rate(1.5)
        with pytest.raises(ValueError):
            agent.set_decay_rate(0.0)


class TestRateOverrides:
    def test_set_override(self):
        agent = DecayControllerAgent()
        agent.set_override("core_memory", 0.999)
        assert agent._rate_overrides["core_memory"] == 0.999

    def test_clear_override(self):
        agent = DecayControllerAgent()
        agent.set_override("core_memory", 0.999)
        agent.clear_override("core_memory")
        assert "core_memory" not in agent._rate_overrides

    def test_clear_nonexistent_is_safe(self):
        agent = DecayControllerAgent()
        agent.clear_override("nonexistent")  # should not raise

    def test_cluster_override_takes_priority(self):
        agent = DecayControllerAgent(decay_rate=0.9)
        cluster_id = uuid4()
        agent.set_override(str(cluster_id), 0.999)

        b = _make_belief(cluster_id=cluster_id)
        rate = agent._get_effective_rate(b)
        assert rate == 0.999

    def test_tag_override(self):
        agent = DecayControllerAgent(decay_rate=0.9)
        agent.set_override("core_value", 0.998)

        b = _make_belief(tags=["core_value", "preference"])
        rate = agent._get_effective_rate(b)
        assert rate == 0.998

    def test_global_rate_fallback(self):
        agent = DecayControllerAgent(decay_rate=0.95)
        b = _make_belief()
        rate = agent._get_effective_rate(b)
        assert rate == 0.95


class TestComputeNewConfidence:
    def test_no_elapsed_time(self):
        agent = DecayControllerAgent(decay_rate=0.995)
        result = agent._compute_new_confidence(0.8, 0.0, 0.995)
        assert result == 0.8

    def test_one_hour(self):
        agent = DecayControllerAgent()
        # 0.8 * 0.995^1 = 0.796
        result = agent._compute_new_confidence(0.8, 1.0, 0.995)
        assert abs(result - 0.796) < 0.001

    def test_24_hours(self):
        agent = DecayControllerAgent()
        # 0.8 * 0.995^24 ≈ 0.71
        result = agent._compute_new_confidence(0.8, 24.0, 0.995)
        assert 0.70 < result < 0.72


class TestDetermineStatus:
    def test_stays_deprecated(self):
        agent = DecayControllerAgent()
        b = _make_belief(status=BeliefStatus.Deprecated)
        status = agent._determine_status(b, 0.9)
        assert status == BeliefStatus.Deprecated

    def test_stays_mutated(self):
        agent = DecayControllerAgent()
        b = _make_belief(status=BeliefStatus.Mutated)
        status = agent._determine_status(b, 0.9)
        assert status == BeliefStatus.Mutated

    def test_stale_unused_deprecated(self):
        agent = DecayControllerAgent(stale_days=30)
        old_date = datetime.now(timezone.utc) - timedelta(days=35)
        b = _make_belief(use_count=0, created_at=old_date)
        status = agent._determine_status(b, 0.9)
        assert status == BeliefStatus.Deprecated

    def test_below_deprecated_threshold(self):
        agent = DecayControllerAgent(threshold_deprecated=0.1)
        b = _make_belief()
        status = agent._determine_status(b, 0.05)
        assert status == BeliefStatus.Deprecated

    def test_below_decaying_threshold(self):
        agent = DecayControllerAgent(threshold_decaying=0.3, threshold_deprecated=0.1)
        b = _make_belief()
        status = agent._determine_status(b, 0.2)
        assert status == BeliefStatus.Decaying

    def test_stays_active(self):
        agent = DecayControllerAgent(threshold_decaying=0.3)
        b = _make_belief()
        status = agent._determine_status(b, 0.5)
        assert status == BeliefStatus.Active


class TestApplyDecay:
    def test_no_decay_for_deprecated(self):
        agent = DecayControllerAgent()
        b = _make_belief(status=BeliefStatus.Deprecated)
        event = agent.apply_decay(b)
        assert event is None

    def test_no_decay_for_mutated(self):
        agent = DecayControllerAgent()
        b = _make_belief(status=BeliefStatus.Mutated)
        event = agent.apply_decay(b)
        assert event is None

    def test_no_decay_if_just_reinforced(self):
        agent = DecayControllerAgent()
        b = _make_belief(last_reinforced=datetime.now(timezone.utc))
        event = agent.apply_decay(b)
        # no meaningful change
        assert event is None

    def test_decay_applied(self):
        agent = DecayControllerAgent(decay_rate=0.9)
        lr = datetime.now(timezone.utc) - timedelta(hours=10)
        b = _make_belief(confidence=0.8, last_reinforced=lr, created_at=lr)
        old_conf = b.confidence

        event = agent.apply_decay(b)

        assert event is not None
        assert b.confidence < old_conf
        assert event.old_confidence == old_conf
        assert event.new_confidence == b.confidence

    def test_status_transition_recorded(self):
        agent = DecayControllerAgent(decay_rate=0.5, threshold_decaying=0.3)
        lr = datetime.now(timezone.utc) - timedelta(hours=5)
        b = _make_belief(confidence=0.5, last_reinforced=lr, created_at=lr)

        event = agent.apply_decay(b)

        # 0.5 * 0.5^5 = 0.015625 -> deprecated
        assert event is not None
        assert b.status == BeliefStatus.Deprecated
        assert event.new_status == BeliefStatus.Deprecated


class TestProcessBeliefs:
    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = DecayControllerAgent()
        events, modified = await agent.process_beliefs([])
        assert events == []
        assert modified == []

    @pytest.mark.asyncio
    async def test_processes_multiple(self):
        agent = DecayControllerAgent(decay_rate=0.9)
        lr = datetime.now(timezone.utc) - timedelta(hours=10)
        beliefs = [
            _make_belief(content="a", confidence=0.8, last_reinforced=lr, created_at=lr),
            _make_belief(content="b", confidence=0.9, last_reinforced=lr, created_at=lr),
        ]

        events, modified = await agent.process_beliefs(beliefs)

        assert len(events) == 2
        assert len(modified) == 2

    @pytest.mark.asyncio
    async def test_skips_unchanged(self):
        agent = DecayControllerAgent()
        now = datetime.now(timezone.utc)
        beliefs = [
            _make_belief(content="fresh", last_reinforced=now),
            _make_belief(content="deprecated", status=BeliefStatus.Deprecated),
        ]

        events, modified = await agent.process_beliefs(beliefs)

        assert len(events) == 0
        assert len(modified) == 0


class TestHalfLife:
    def test_default_rate(self):
        agent = DecayControllerAgent(decay_rate=0.995)
        half_life = agent.estimate_half_life_hours()
        # ln(0.5) / ln(0.995) ≈ 138.3
        assert 137 < half_life < 140

    def test_custom_rate(self):
        agent = DecayControllerAgent()
        half_life = agent.estimate_half_life_hours(0.9)
        # ln(0.5) / ln(0.9) ≈ 6.58
        assert 6 < half_life < 7

    def test_rate_of_one(self):
        agent = DecayControllerAgent()
        half_life = agent.estimate_half_life_hours(1.0)
        assert half_life == float("inf")
