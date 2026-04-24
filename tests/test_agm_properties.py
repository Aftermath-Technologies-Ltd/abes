# Author: Bradley R. Kinnard
"""AGM belief revision postulate tests.

Tests verify that ABES satisfies four core AGM-like properties using real
Belief objects and actual agent methods - no mocks.

Postulates tested:
- Success: new evidence updates confidence (not ignored).
- Consistency: after contradiction detection, contradicting beliefs don't
  both stay high-confidence.
- Minimal change: recent reinforcement protects beliefs from heavy decay.
- Recovery: a deprecated belief that receives strong evidence can be shown
  to regain confidence when reactivated (acknowledged limitation documented).
"""

import asyncio
import pytest
from uuid import uuid4

from backend.core.models.belief import (
    Belief,
    BeliefStatus,
    EvidenceRef,
    OriginMetadata,
)
from backend.agents.decay_controller import DecayControllerAgent


def _belief(content: str = "test", confidence: float = 0.5, salience: float = 0.8) -> Belief:
    """Construct a minimal active Belief."""
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        origin=OriginMetadata(source="test"),
    )


def _evidence(direction: str = "supports", weight: float = 1.0) -> EvidenceRef:
    """Construct a minimal EvidenceRef."""
    return EvidenceRef(
        content="corroborating observation",
        direction=direction,
        weight=weight,
    )


class TestSuccessPostulate:
    """AGM Success: revision by new evidence must incorporate that evidence.

    When a belief receives supporting evidence via add_evidence(), its
    confidence must strictly increase above the prior value.
    """

    def test_single_supporting_evidence_increases_confidence(self) -> None:
        """add_evidence('supports') on a low-confidence belief raises confidence."""
        b = _belief(confidence=0.4)
        prior_confidence = b.confidence

        b.add_evidence(_evidence(direction="supports", weight=2.0))

        assert b.confidence > prior_confidence, (
            f"Expected confidence to increase after supporting evidence; "
            f"got {b.confidence} vs prior {prior_confidence}"
        )

    def test_multiple_supporting_evidence_converges_upward(self) -> None:
        """Repeated supporting evidence pushes confidence above the initial value."""
        b = _belief(confidence=0.3)
        initial = b.confidence

        for _ in range(5):
            b.add_evidence(_evidence(direction="supports", weight=1.0))

        assert b.confidence > initial

    def test_attacking_evidence_decreases_confidence(self) -> None:
        """add_evidence('attacks') lowers confidence below the prior."""
        b = _belief(confidence=0.8)
        prior = b.confidence

        b.add_evidence(_evidence(direction="attacks", weight=3.0))

        assert b.confidence < prior, (
            f"Expected confidence to decrease after attacking evidence; "
            f"got {b.confidence} vs prior {prior}"
        )

    def test_evidence_balance_reflects_inputs(self) -> None:
        """evidence_balance is positive when support outweighs attack."""
        b = _belief()
        b.add_evidence(_evidence(direction="supports", weight=3.0))
        b.add_evidence(_evidence(direction="attacks", weight=1.0))

        assert b.evidence_balance > 0.0


class TestConsistencyPostulate:
    """AGM Consistency: the revised belief set should be internally consistent.

    After contradicting beliefs are identified, at most one of a contradicting
    pair should remain at high confidence. We test this structurally by marking
    one as deprecated and verifying the invariant holds.
    """

    def test_contradicting_pair_resolved_by_deprecation(self) -> None:
        """After resolution, at most one of a contradicting pair retains confidence > 0.5."""
        b1 = _belief(content="birds can fly", confidence=0.85)
        b2 = _belief(content="birds cannot fly", confidence=0.80)

        # Record the contradiction via graph link
        b1.add_link(b2.id, relation="contradicts", weight=0.9)

        # Resolution: deprecate the weaker belief (b2 has less confidence)
        loser = b2 if b2.confidence <= b1.confidence else b1
        loser.deprecate(reason="resolved contradiction")

        active_high = [
            b for b in (b1, b2)
            if b.status == BeliefStatus.Active and b.confidence > 0.5
        ]
        assert len(active_high) <= 1, (
            f"After contradiction resolution, at most 1 belief should be active and "
            f"high-confidence; found {len(active_high)}"
        )

    def test_deprecated_belief_excluded_from_active_pool(self) -> None:
        """A deprecated belief is not considered active."""
        b = _belief(confidence=0.9)
        b.deprecate(reason="lost contradiction")
        assert b.status == BeliefStatus.Deprecated
        assert b.status != BeliefStatus.Active

    def test_one_of_contradicting_pair_can_remain_dominant(self) -> None:
        """The surviving belief keeps its high confidence intact."""
        b1 = _belief(content="the earth is round", confidence=0.95)
        b2 = _belief(content="the earth is flat", confidence=0.3)

        b1.add_link(b2.id, relation="contradicts", weight=1.0)
        b2.deprecate(reason="resolved contradiction")

        assert b1.status == BeliefStatus.Active
        assert b1.confidence == pytest.approx(0.95)


class TestMinimalChangePostulate:
    """AGM Minimal Change: revision should change the belief set as little as possible.

    Concretely: beliefs that received recent reinforcement (boost_salience) should
    retain higher salience than unreinforced peers after the same amount of decay.
    Decay must not flatten all beliefs to the same level.
    """

    def test_reinforced_belief_retains_higher_salience_than_unreinforced(self) -> None:
        """After equal decay, reinforced belief has strictly higher salience."""
        reinforced = _belief(confidence=0.7, salience=0.8)
        unreinforced = _belief(confidence=0.7, salience=0.8)

        # Boost the reinforced belief
        reinforced.boost_salience(0.15)

        # Apply equal decay to both
        hours = 24.0
        s_reinforced = reinforced.decay_salience(hours)
        s_unreinforced = unreinforced.decay_salience(hours)

        assert s_reinforced > s_unreinforced, (
            f"Reinforced belief salience ({s_reinforced:.4f}) should exceed "
            f"unreinforced ({s_unreinforced:.4f}) after equal decay"
        )

    def test_axiom_immune_to_confidence_decay(self) -> None:
        """Axioms are exempt from apply_decay - a form of minimal-change protection."""
        b = _belief(confidence=0.9)
        b.promote_to_axiom()
        prior = b.confidence

        b.apply_decay(0.5)

        assert b.confidence == pytest.approx(prior), (
            "Axiom confidence must not change under apply_decay"
        )

    def test_decay_does_not_flatten_different_salience_levels(self) -> None:
        """Beliefs starting at different salience levels remain ordered after decay."""
        high = _belief(salience=0.9)
        low = _belief(salience=0.3)

        high.decay_salience(48.0)
        low.decay_salience(48.0)

        assert high.salience > low.salience, (
            "Ordering of salience must be preserved after equal decay"
        )


class TestRecoveryPostulate:
    """AGM Recovery: a contracted belief can be restored by new evidence.

    Full recovery (reactivating a Deprecated belief back to Active) requires
    external logic - the Belief model itself does not auto-activate on evidence.
    This test documents the limitation and demonstrates that confidence is
    recoverable even on a deprecated belief, serving as the foundation for
    an external re-activation mechanism.
    """

    def test_deprecated_belief_confidence_updates_on_evidence(self) -> None:
        """add_evidence() still adjusts confidence even on a deprecated belief.

        Limitation: ABES does not automatically re-activate a deprecated belief
        when evidence is added. Re-activation requires explicit status mutation
        by a higher-level controller (e.g., the ReinforcementAgent). This test
        confirms the evidence-bookkeeping is correct as a prerequisite for
        external re-activation logic.
        """
        b = _belief(confidence=0.2)
        b.deprecate(reason="low confidence")
        assert b.status == BeliefStatus.Deprecated

        prior_confidence = b.confidence
        b.add_evidence(_evidence(direction="supports", weight=5.0))

        # Confidence should have changed even though the belief is deprecated
        assert b.confidence != pytest.approx(prior_confidence), (
            "Evidence must update confidence even on deprecated beliefs, "
            "so external recovery logic has an updated prior to work with"
        )

    def test_manual_reactivation_after_evidence_recovery(self) -> None:
        """A controller can reactivate a deprecated belief after sufficient evidence."""
        b = _belief(confidence=0.1)
        b.deprecate(reason="initial deprecation due to low confidence")

        # Apply strong supporting evidence
        for _ in range(10):
            b.add_evidence(_evidence(direction="supports", weight=2.0))

        # External controller checks confidence threshold and reactivates
        if b.confidence >= 0.5:
            b.status = BeliefStatus.Active

        assert b.status == BeliefStatus.Active, (
            f"Belief with sufficient post-deprecation evidence should be "
            f"re-activatable by an external controller; confidence={b.confidence:.4f}"
        )
