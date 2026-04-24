# Author: Bradley R. Kinnard
"""Tests for MutationEngineerAgent."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.agents.mutation_engineer import (
    MutationEngineerAgent,
    MutationProposal,
    _has_temporal_marker,
    _has_broad_claim,
    _count_mutation_depth,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.4,
    tension: float = 0.7,
    status: BeliefStatus = BeliefStatus.Active,
    source: str = "test",
    parent_id=None,
    tags: list[str] = None,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        status=status,
        origin=OriginMetadata(source=source),
        parent_id=parent_id,
        tags=tags or [],
    )


class TestTemporalDetection:
    def test_detects_as_of(self):
        assert _has_temporal_marker("as of January 2024")

    def test_detects_since(self):
        assert _has_temporal_marker("since last week")

    def test_detects_month_year(self):
        assert _has_temporal_marker("March 2025 data")

    def test_no_temporal_marker(self):
        assert not _has_temporal_marker("the sky is blue")


class TestBroadClaimDetection:
    def test_detects_always(self):
        assert _has_broad_claim("I always prefer tea")

    def test_detects_never(self):
        assert _has_broad_claim("Python never crashes")

    def test_detects_everyone(self):
        assert _has_broad_claim("Everyone loves pizza")

    def test_no_broad_claim(self):
        assert not _has_broad_claim("I like coffee")


class TestMutationDepth:
    def test_no_parent(self):
        b = _make_belief()
        assert _count_mutation_depth(b, {b.id: b}) == 0

    def test_one_parent(self):
        parent = _make_belief(content="parent")
        child = _make_belief(content="child", parent_id=parent.id)
        belief_map = {parent.id: parent, child.id: child}
        assert _count_mutation_depth(child, belief_map) == 1

    def test_chain_of_three(self):
        grandparent = _make_belief(content="gp")
        parent = _make_belief(content="p", parent_id=grandparent.id)
        child = _make_belief(content="c", parent_id=parent.id)
        belief_map = {grandparent.id: grandparent, parent.id: parent, child.id: child}
        assert _count_mutation_depth(child, belief_map) == 2

    def test_missing_parent_stops_chain(self):
        fake_parent_id = uuid4()
        child = _make_belief(content="orphan", parent_id=fake_parent_id)
        belief_map = {child.id: child}
        assert _count_mutation_depth(child, belief_map) == 1


class TestShouldMutate:
    def test_triggers_on_high_tension_low_confidence(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(tension=0.7, confidence=0.4)
        assert agent._should_mutate(b)

    def test_no_trigger_low_tension(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(tension=0.3, confidence=0.4)
        assert not agent._should_mutate(b)

    def test_no_trigger_high_confidence(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(tension=0.7, confidence=0.8)
        assert not agent._should_mutate(b)

    def test_no_trigger_non_active_status(self):
        agent = MutationEngineerAgent()
        b = _make_belief(tension=0.7, confidence=0.4, status=BeliefStatus.Deprecated)
        assert not agent._should_mutate(b)


class TestStrategySelection:
    def test_hedge_default(self):
        agent = MutationEngineerAgent()
        b = _make_belief(content="coffee is good", confidence=0.4)
        assert agent._select_strategy(b, None) == "hedge"

    def test_condition_when_contradicting_has_temporal(self):
        agent = MutationEngineerAgent()
        b = _make_belief(content="coffee is good")
        contra = _make_belief(content="as of 2024, tea is better")
        assert agent._select_strategy(b, contra) == "condition"

    def test_scope_narrow_for_broad_claim(self):
        agent = MutationEngineerAgent()
        b = _make_belief(content="coffee is always the best")
        contra = _make_belief(content="tea can be better")
        assert agent._select_strategy(b, contra) == "scope_narrow"

    def test_source_attribute_for_very_low_confidence(self):
        agent = MutationEngineerAgent()
        b = _make_belief(content="coffee helps focus", confidence=0.2)
        assert agent._select_strategy(b, None) == "source_attribute"


class TestMutationTemplates:
    def test_hedge_adds_prefix(self):
        agent = MutationEngineerAgent()
        result = agent._apply_hedge("Coffee is good")
        assert result.startswith("It may be that")
        assert "coffee is good" in result.lower()

    def test_hedge_avoids_double_hedging(self):
        agent = MutationEngineerAgent()
        result = agent._apply_hedge("It may be that coffee is good")
        assert result == "It may be that coffee is good"

    def test_condition_appends_date(self):
        agent = MutationEngineerAgent()
        result = agent._apply_condition("Coffee is popular", "January 2024")
        assert "at least as of January 2024" in result

    def test_scope_narrow_replaces_always(self):
        agent = MutationEngineerAgent()
        result = agent._apply_scope_narrow("Coffee always tastes good")
        assert "usually" in result.lower()
        assert "always" not in result.lower()

    def test_scope_narrow_replaces_never(self):
        agent = MutationEngineerAgent()
        result = agent._apply_scope_narrow("Coffee never disappoints")
        assert "rarely" in result.lower()

    def test_scope_narrow_adds_qualifier_if_no_match(self):
        agent = MutationEngineerAgent()
        result = agent._apply_scope_narrow("Coffee is the best drink")
        assert "in most cases" in result

    def test_source_attribute_adds_prefix(self):
        agent = MutationEngineerAgent()
        result = agent._apply_source_attribute("Coffee helps focus", "user_input")
        assert result.startswith("According to user_input")


class TestProposeMutation:
    def test_returns_none_if_not_triggered(self):
        agent = MutationEngineerAgent()
        b = _make_belief(tension=0.1, confidence=0.9)
        assert agent.propose_mutation(b) is None

    def test_returns_proposal_when_triggered(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(content="Coffee is great", tension=0.7, confidence=0.4)
        proposal = agent.propose_mutation(b)
        assert proposal is not None
        assert isinstance(proposal, MutationProposal)
        assert proposal.original_id == b.id
        assert proposal.mutated_belief.parent_id == b.id

    def test_mutated_belief_has_reduced_confidence(self):
        """Mutated beliefs get reduced confidence based on tension penalty."""
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(tension=0.7, confidence=0.4)
        proposal = agent.propose_mutation(b)
        # Confidence is reduced with penalty: original - 0.1 - (tension * 0.1)
        # 0.4 - 0.1 - 0.07 = 0.23, clamped to 0.3 minimum
        assert proposal.mutated_belief.confidence <= b.confidence
        assert proposal.mutated_belief.confidence >= 0.3

    def test_mutated_belief_inherits_tags(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(tension=0.7, confidence=0.4, tags=["preference", "food"])
        proposal = agent.propose_mutation(b)
        assert proposal.mutated_belief.tags == ["preference", "food"]

    def test_blocks_at_max_depth(self):
        agent = MutationEngineerAgent(
            tension_threshold=0.6, confidence_threshold=0.5, max_depth=2
        )
        gp = _make_belief(content="gp")
        p = _make_belief(content="p", parent_id=gp.id, tension=0.7, confidence=0.4)
        c = _make_belief(content="c", parent_id=p.id, tension=0.7, confidence=0.4)
        all_beliefs = [gp, p, c]

        # child is at depth 2, should be blocked
        proposal = agent.propose_mutation(c, all_beliefs=all_beliefs)
        assert proposal is None


class TestProcessBeliefs:
    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = MutationEngineerAgent()
        result = await agent.process_beliefs([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_mutations_when_none_triggered(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        beliefs = [
            _make_belief(tension=0.1, confidence=0.9),
            _make_belief(tension=0.2, confidence=0.8),
        ]
        result = await agent.process_beliefs(beliefs)
        assert result == []

    @pytest.mark.asyncio
    async def test_proposes_mutations_for_qualifying_beliefs(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        beliefs = [
            _make_belief(content="Coffee is great", tension=0.7, confidence=0.4),
            _make_belief(content="Tea is fine", tension=0.2, confidence=0.8),
        ]
        result = await agent.process_beliefs(beliefs)
        assert len(result) == 1
        assert result[0].mutated_belief.parent_id == beliefs[0].id

    @pytest.mark.asyncio
    async def test_uses_tension_map_if_provided(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b = _make_belief(content="Test", tension=0.1, confidence=0.4)
        # tension_map overrides belief.tension
        result = await agent.process_beliefs([b], tension_map={b.id: 0.8})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_finds_contradicting_belief_from_pairs(self):
        agent = MutationEngineerAgent(tension_threshold=0.6, confidence_threshold=0.5)
        b1 = _make_belief(content="Coffee is great", tension=0.7, confidence=0.4)
        b2 = _make_belief(content="As of 2024, tea is better", tension=0.7, confidence=0.4)
        pairs = [(b1.id, b2.id, 0.8)]

        result = await agent.process_beliefs([b1, b2], contradiction_pairs=pairs)
        # both should trigger; b1 should use "condition" strategy due to b2's temporal marker
        strategies = {p.strategy for p in result}
        assert "condition" in strategies
