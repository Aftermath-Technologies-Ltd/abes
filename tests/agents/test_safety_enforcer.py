# Author: Bradley R. Kinnard
"""Tests for SafetyLimitEnforcer."""

import pytest
from uuid import uuid4

from backend.agents.safety_enforcer import (
    SafetyLimitEnforcer,
    SafetyLimitError,
    get_safety_enforcer,
    reset_safety_enforcer,
)
from backend.agents.safety_sanity import SafetySanityAgent
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


@pytest.fixture
def enforcer():
    reset_safety_enforcer()
    return SafetyLimitEnforcer()


@pytest.fixture
def sample_belief():
    return Belief(
        content="Test belief",
        confidence=0.8,
        origin=OriginMetadata(source="test"),
    )


class TestContentValidation:
    def test_valid_content(self, enforcer):
        result = enforcer.validate_belief_content("Valid content")
        assert result == "Valid content"

    def test_empty_content_raises(self, enforcer):
        with pytest.raises(SafetyLimitError) as exc:
            enforcer.validate_belief_content("")
        assert exc.value.limit_type == "content_empty"

    def test_whitespace_only_raises(self, enforcer):
        with pytest.raises(SafetyLimitError):
            enforcer.validate_belief_content("   ")

    def test_long_content_truncated(self, enforcer):
        long_content = "x" * 3000
        result = enforcer.validate_belief_content(long_content)
        assert len(result) <= 2000
        assert "[truncated]" in result


class TestBeliefCreation:
    def test_normal_creation(self, enforcer, sample_belief):
        # should not raise
        enforcer.validate_belief_creation(sample_belief, 100)

    def test_at_limit_raises(self, enforcer, sample_belief):
        with pytest.raises(SafetyLimitError) as exc:
            enforcer.validate_belief_creation(sample_belief, 10000)
        assert exc.value.limit_type == "max_active_beliefs"

    def test_truncates_long_content(self, enforcer):
        belief = Belief(
            content="x" * 3000,
            confidence=0.8,
            origin=OriginMetadata(source="test"),
        )
        enforcer.validate_belief_creation(belief, 100)
        assert len(belief.content) <= 2000


class TestMutationValidation:
    def test_shallow_mutation_allowed(self, enforcer):
        parent = Belief(
            content="Parent",
            confidence=0.8,
            origin=OriginMetadata(source="test"),
        )
        # should not raise
        enforcer.validate_mutation(parent, [parent])

    def test_deep_mutation_blocked(self, enforcer):
        # create chain of 6 beliefs
        beliefs = []
        parent_id = None
        for i in range(6):
            b = Belief(
                content=f"Belief {i}",
                confidence=0.8,
                origin=OriginMetadata(source="test"),
                parent_id=parent_id,
            )
            beliefs.append(b)
            parent_id = b.id

        # last belief has depth 5, at limit
        with pytest.raises(SafetyLimitError) as exc:
            enforcer.validate_mutation(beliefs[-1], beliefs)
        assert exc.value.limit_type == "max_mutation_depth"


class TestClusterValidation:
    def test_normal_assignment(self, enforcer):
        # should not raise
        enforcer.validate_cluster_assignment(uuid4(), 100)

    def test_at_limit_raises(self, enforcer):
        with pytest.raises(SafetyLimitError) as exc:
            enforcer.validate_cluster_assignment(uuid4(), 500)
        assert exc.value.limit_type == "max_beliefs_per_cluster"


class TestContradictionPairs:
    def test_under_limit(self, enforcer):
        pairs = [(i, i + 1) for i in range(1000)]
        result = enforcer.limit_contradiction_pairs(pairs)
        assert len(result) == 1000

    def test_over_limit_truncated(self, enforcer):
        pairs = [(i, i + 1) for i in range(60000)]
        result = enforcer.limit_contradiction_pairs(pairs)
        assert len(result) == 50000


class TestSnapshotSize:
    def test_normal_size(self, enforcer):
        # 10MB should be fine
        enforcer.validate_snapshot_size(10 * 1024 * 1024)

    def test_over_limit_raises(self, enforcer):
        # 60MB exceeds 50MB limit
        with pytest.raises(SafetyLimitError) as exc:
            enforcer.validate_snapshot_size(60 * 1024 * 1024)
        assert exc.value.limit_type == "max_snapshot_size"


class TestSingleton:
    def test_get_singleton(self):
        reset_safety_enforcer()
        e1 = get_safety_enforcer()
        e2 = get_safety_enforcer()
        assert e1 is e2

    def test_reset_creates_new(self):
        e1 = get_safety_enforcer()
        reset_safety_enforcer()
        e2 = get_safety_enforcer()
        assert e1 is not e2


class TestSafetyAgent:
    def test_get_agent(self, enforcer):
        agent = enforcer.get_safety_agent()
        assert isinstance(agent, SafetySanityAgent)
