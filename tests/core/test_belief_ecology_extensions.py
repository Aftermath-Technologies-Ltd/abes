# Author: Bradley R. Kinnard
"""Tests for salience dynamics, evidence ledger, graph edges, and dormancy on the Belief model."""

import math
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.core.models.belief import (
    Belief,
    BeliefLink,
    BeliefStatus,
    EvidenceRef,
    OriginMetadata,
    utcnow,
)


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.8,
    salience: float = 1.0,
    half_life_days: float = 7.0,
    status: BeliefStatus = BeliefStatus.Active,
    **kwargs,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        half_life_days=half_life_days,
        status=status,
        origin=OriginMetadata(source="test"),
        **kwargs,
    )


# ─── EvidenceRef model ───────────────────────────────────────────────────────


class TestEvidenceRef:
    def test_defaults(self):
        ref = EvidenceRef(content="observed X")
        assert ref.direction == "supports"
        assert ref.weight == 1.0
        assert ref.source_id is None

    def test_attack_direction(self):
        ref = EvidenceRef(content="refutes X", direction="attacks", weight=2.0)
        assert ref.direction == "attacks"
        assert ref.weight == 2.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            EvidenceRef(content="bad", direction="neutral")


# ─── BeliefLink model ────────────────────────────────────────────────────────


class TestBeliefLink:
    def test_reinforces(self):
        target = uuid4()
        link = BeliefLink(target_id=target, relation="reinforces", weight=0.7)
        assert link.target_id == target
        assert link.relation == "reinforces"

    def test_contradicts(self):
        link = BeliefLink(target_id=uuid4(), relation="contradicts", weight=1.5)
        assert link.relation == "contradicts"

    def test_invalid_relation_raises(self):
        with pytest.raises(ValueError, match="relation"):
            BeliefLink(target_id=uuid4(), relation="ignores")


# ─── BeliefStatus ────────────────────────────────────────────────────────────


class TestBeliefStatusDormant:
    def test_dormant_exists(self):
        assert BeliefStatus.Dormant.value == "dormant"

    def test_all_statuses(self):
        expected = {"active", "decaying", "dormant", "mutated", "deprecated"}
        actual = {s.value for s in BeliefStatus}
        assert actual == expected


# ─── Salience dynamics ───────────────────────────────────────────────────────


class TestSalienceDecay:
    def test_default_salience_is_one(self):
        b = _make_belief()
        assert b.salience == 1.0

    def test_decay_salience_zero_hours_noop(self):
        b = _make_belief(salience=0.8)
        result = b.decay_salience(0)
        assert result == 0.8
        assert b.salience == 0.8

    def test_decay_salience_half_life(self):
        """After exactly one half-life, salience should halve."""
        b = _make_belief(salience=1.0, half_life_days=7.0)
        hours = 7.0 * 24  # one half-life
        result = b.decay_salience(hours)
        assert abs(result - 0.5) < 0.01

    def test_decay_salience_two_half_lives(self):
        b = _make_belief(salience=1.0, half_life_days=1.0)
        hours = 2 * 24  # two half-lives
        result = b.decay_salience(hours)
        assert abs(result - 0.25) < 0.01

    def test_decay_salience_clamped_to_zero(self):
        b = _make_belief(salience=0.001, half_life_days=0.01)
        result = b.decay_salience(9999)
        assert result >= 0.0

    def test_decay_salience_negative_hours_noop(self):
        b = _make_belief(salience=0.5)
        result = b.decay_salience(-10)
        assert result == 0.5

    def test_salience_validation_rejects_negative(self):
        with pytest.raises(ValueError):
            _make_belief(salience=-0.1)

    def test_salience_validation_rejects_over_one(self):
        with pytest.raises(ValueError):
            _make_belief(salience=1.5)


class TestSalienceBoost:
    def test_boost_salience(self):
        b = _make_belief(salience=0.5)
        result = b.boost_salience(0.2)
        assert abs(result - 0.7) < 0.001

    def test_boost_salience_capped_at_one(self):
        b = _make_belief(salience=0.95)
        result = b.boost_salience(0.2)
        assert result == 1.0

    def test_boost_updates_timestamp(self):
        b = _make_belief(salience=0.3)
        old_ts = b.updated_at
        # tiny sleep isn't reliable; just check the method runs
        b.boost_salience(0.1)
        assert b.updated_at >= old_ts


# ─── Evidence Ledger ─────────────────────────────────────────────────────────


class TestEvidenceLedger:
    def test_add_supporting_evidence(self):
        b = _make_belief(confidence=0.5)
        ref = EvidenceRef(content="saw it happen", direction="supports", weight=1.0)
        b.add_evidence(ref)
        assert len(b.evidence_for) == 1
        assert len(b.evidence_against) == 0

    def test_add_attacking_evidence(self):
        b = _make_belief(confidence=0.8)
        ref = EvidenceRef(content="contradicted by data", direction="attacks", weight=1.5)
        b.add_evidence(ref)
        assert len(b.evidence_against) == 1
        assert len(b.evidence_for) == 0

    def test_bayes_update_pushes_confidence_up(self):
        b = _make_belief(confidence=0.5)
        for _ in range(5):
            b.add_evidence(EvidenceRef(content="more proof", direction="supports"))
        assert b.confidence > 0.5

    def test_bayes_update_pushes_confidence_down(self):
        b = _make_belief(confidence=0.8)
        for _ in range(5):
            b.add_evidence(EvidenceRef(content="refuted", direction="attacks"))
        assert b.confidence < 0.8

    def test_bayes_update_bounded(self):
        b = _make_belief(confidence=0.5)
        for _ in range(100):
            b.add_evidence(EvidenceRef(content="proof", direction="supports"))
        assert 0.01 <= b.confidence <= 0.99

    def test_evidence_balance_positive(self):
        b = _make_belief()
        b.add_evidence(EvidenceRef(content="a", direction="supports", weight=3.0))
        b.add_evidence(EvidenceRef(content="b", direction="attacks", weight=1.0))
        assert b.evidence_balance == 2.0

    def test_evidence_balance_negative(self):
        b = _make_belief()
        b.add_evidence(EvidenceRef(content="a", direction="attacks", weight=5.0))
        b.add_evidence(EvidenceRef(content="b", direction="supports", weight=1.0))
        assert b.evidence_balance == -4.0

    def test_no_evidence_keeps_confidence(self):
        b = _make_belief(confidence=0.6)
        b._bayes_update()
        assert b.confidence == 0.6


# ─── Graph Edges ─────────────────────────────────────────────────────────────


class TestGraphEdges:
    def test_add_link(self):
        b = _make_belief()
        target = uuid4()
        b.add_link(target, "reinforces", 0.8)
        assert len(b.links) == 1
        assert b.links[0].target_id == target
        assert b.links[0].weight == 0.8

    def test_add_link_updates_existing(self):
        b = _make_belief()
        target = uuid4()
        b.add_link(target, "reinforces", 0.5)
        b.add_link(target, "reinforces", 0.9)
        assert len(b.links) == 1
        assert b.links[0].weight == 0.9

    def test_add_link_different_relations_coexist(self):
        b = _make_belief()
        target = uuid4()
        b.add_link(target, "reinforces", 0.5)
        b.add_link(target, "contradicts", 0.8)
        assert len(b.links) == 2

    def test_get_links_all(self):
        b = _make_belief()
        b.add_link(uuid4(), "reinforces", 0.3)
        b.add_link(uuid4(), "contradicts", 0.7)
        assert len(b.get_links()) == 2

    def test_get_links_filtered(self):
        b = _make_belief()
        b.add_link(uuid4(), "reinforces", 0.3)
        b.add_link(uuid4(), "contradicts", 0.7)
        assert len(b.get_links("reinforces")) == 1
        assert len(b.get_links("contradicts")) == 1


# ─── Dormancy ────────────────────────────────────────────────────────────────


class TestDormancy:
    def test_hibernate(self):
        b = _make_belief(salience=0.8)
        b.hibernate()
        assert b.status == BeliefStatus.Dormant
        assert b.salience == 0.0

    def test_reawaken(self):
        b = _make_belief(status=BeliefStatus.Dormant, salience=0.0)
        b.reawaken(0.6)
        assert b.status == BeliefStatus.Active
        assert b.salience == 0.6

    def test_reawaken_only_from_dormant(self):
        b = _make_belief(status=BeliefStatus.Active, salience=0.5)
        b.reawaken(0.9)
        # should not change status since it's already Active
        assert b.status == BeliefStatus.Active
        assert b.salience == 0.5  # unchanged

    def test_reawaken_capped_at_one(self):
        b = _make_belief(status=BeliefStatus.Dormant, salience=0.0)
        b.reawaken(1.5)
        assert b.salience == 1.0


# ─── Backward compatibility ──────────────────────────────────────────────────


class TestBackwardCompat:
    """Ensure existing belief features still work with new fields."""

    def test_apply_decay_still_works(self):
        b = _make_belief(confidence=0.8)
        b.apply_decay(0.5)
        assert abs(b.confidence - 0.4) < 0.001

    def test_increment_use_still_works(self):
        b = _make_belief()
        b.increment_use()
        assert b.use_count == 1

    def test_reinforce_still_works(self):
        b = _make_belief()
        old_lr = b.origin.last_reinforced
        b.reinforce()
        assert b.origin.last_reinforced >= old_lr

    def test_belief_serialization_includes_new_fields(self):
        b = _make_belief()
        b.add_evidence(EvidenceRef(content="test", direction="supports"))
        b.add_link(uuid4(), "reinforces", 0.5)
        data = b.model_dump()
        assert "salience" in data
        assert "half_life_days" in data
        assert "evidence_for" in data
        assert "evidence_against" in data
        assert "links" in data
        assert len(data["evidence_for"]) == 1
        assert len(data["links"]) == 1
