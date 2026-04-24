# Author: Bradley R. Kinnard
"""Tests for SafetySanityAgent."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.agents.safety_sanity import (
    SafetySanityAgent,
    SafetyViolation,
    SafetyMetrics,
    ViolationType,
    ActionType,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "test",
    confidence: float = 0.8,
    status: BeliefStatus = BeliefStatus.Active,
    tags: list[str] = None,
    parent_id=None,
    cluster_id=None,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        status=status,
        origin=OriginMetadata(source="test"),
        tags=tags or [],
        parent_id=parent_id,
        cluster_id=cluster_id,
    )


class TestLowConfidenceUsage:
    def test_flags_low_confidence(self):
        agent = SafetySanityAgent(min_confidence_for_usage=0.3)
        b = _make_belief(confidence=0.1)
        beliefs = [b]

        violations = agent.check_low_confidence_usage(beliefs, [b.id])

        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.LowConfidenceUsage

    def test_allows_high_confidence(self):
        agent = SafetySanityAgent(min_confidence_for_usage=0.3)
        b = _make_belief(confidence=0.8)
        beliefs = [b]

        violations = agent.check_low_confidence_usage(beliefs, [b.id])

        assert len(violations) == 0


class TestMutationDepth:
    def test_blocks_deep_mutations(self):
        agent = SafetySanityAgent(max_mutation_depth=2)

        gp = _make_belief(content="grandparent")
        p = _make_belief(content="parent", parent_id=gp.id)
        c = _make_belief(content="child", parent_id=p.id)
        all_beliefs = [gp, p, c]

        violation = agent.check_mutation_depth(c, all_beliefs)

        assert violation is not None
        assert violation.violation_type == ViolationType.RunawayMutation
        assert violation.action_taken == ActionType.Block

    def test_allows_shallow_mutations(self):
        agent = SafetySanityAgent(max_mutation_depth=5)

        parent = _make_belief(content="parent")
        child = _make_belief(content="child", parent_id=parent.id)
        all_beliefs = [parent, child]

        violation = agent.check_mutation_depth(child, all_beliefs)

        assert violation is None

    def test_vetoes_recorded(self):
        agent = SafetySanityAgent(max_mutation_depth=1)

        parent = _make_belief(content="parent")
        child = _make_belief(content="child", parent_id=parent.id)
        all_beliefs = [parent, child]

        agent.check_mutation_depth(child, all_beliefs)

        assert agent.is_mutation_vetoed(child.id)


class TestBeliefCount:
    def test_blocks_at_limit(self):
        agent = SafetySanityAgent(max_active_beliefs=5)
        beliefs = [_make_belief(content=f"b{i}") for i in range(5)]

        violation = agent.check_belief_count(beliefs)

        assert violation is not None
        assert violation.violation_type == ViolationType.BeliefProliferation
        assert violation.action_taken == ActionType.Block

    def test_warns_near_limit(self):
        agent = SafetySanityAgent(max_active_beliefs=10)
        beliefs = [_make_belief(content=f"b{i}") for i in range(9)]

        violation = agent.check_belief_count(beliefs)

        assert violation is not None
        assert violation.action_taken == ActionType.Warn

    def test_allows_under_limit(self):
        agent = SafetySanityAgent(max_active_beliefs=100)
        beliefs = [_make_belief(content=f"b{i}") for i in range(10)]

        violation = agent.check_belief_count(beliefs)

        assert violation is None


class TestClusterSizes:
    def test_flags_large_clusters(self):
        agent = SafetySanityAgent(max_beliefs_per_cluster=3)
        cluster_id = uuid4()
        beliefs = [_make_belief(content=f"b{i}", cluster_id=cluster_id) for i in range(4)]

        violations = agent.check_cluster_sizes(beliefs)

        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.ClusterOverflow

    def test_allows_small_clusters(self):
        agent = SafetySanityAgent(max_beliefs_per_cluster=10)
        cluster_id = uuid4()
        beliefs = [_make_belief(content=f"b{i}", cluster_id=cluster_id) for i in range(5)]

        violations = agent.check_cluster_sizes(beliefs)

        assert len(violations) == 0


class TestContentLength:
    def test_flags_long_content(self):
        agent = SafetySanityAgent(max_content_length=50)
        b = _make_belief(content="x" * 100)

        violation = agent.check_content_length(b)

        assert violation is not None
        assert violation.violation_type == ViolationType.ContentTooLong
        assert violation.action_taken == ActionType.Override

    def test_allows_short_content(self):
        agent = SafetySanityAgent(max_content_length=100)
        b = _make_belief(content="short content")

        violation = agent.check_content_length(b)

        assert violation is None


class TestCoreBeliefDeprecation:
    def test_blocks_core_deprecation(self):
        agent = SafetySanityAgent(core_tags=["core_value"])
        b = _make_belief(tags=["core_value", "other"])

        violations = agent.check_core_belief_deprecation([b])

        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.CoreBeliefForgotten
        assert violations[0].action_taken == ActionType.Block

    def test_allows_non_core_deprecation(self):
        agent = SafetySanityAgent(core_tags=["core_value"])
        b = _make_belief(tags=["preference"])

        violations = agent.check_core_belief_deprecation([b])

        assert len(violations) == 0

    def test_vetoes_recorded(self):
        agent = SafetySanityAgent(core_tags=["essential"])
        b = _make_belief(tags=["essential"])

        agent.check_core_belief_deprecation([b])

        assert agent.is_deprecation_vetoed(b.id)


class TestDeprecationSpike:
    def test_flags_large_deprecation(self):
        agent = SafetySanityAgent(deprecation_spike_threshold=0.2)

        violation = agent.check_deprecation_spike(total_active=100, to_deprecate_count=25)

        assert violation is not None
        assert violation.violation_type == ViolationType.DeprecationSpike

    def test_allows_normal_deprecation(self):
        agent = SafetySanityAgent(deprecation_spike_threshold=0.3)

        violation = agent.check_deprecation_spike(total_active=100, to_deprecate_count=10)

        assert violation is None

    def test_handles_zero_active(self):
        agent = SafetySanityAgent()

        violation = agent.check_deprecation_spike(total_active=0, to_deprecate_count=5)

        assert violation is None


class TestTruncateContent:
    def test_truncates_long_content(self):
        agent = SafetySanityAgent(max_content_length=50)

        result = agent.truncate_content("x" * 100)

        assert len(result) == 50
        assert result.endswith("[truncated]")

    def test_preserves_short_content(self):
        agent = SafetySanityAgent(max_content_length=100)

        result = agent.truncate_content("short")

        assert result == "short"


class TestRunAllChecks:
    @pytest.mark.asyncio
    async def test_runs_multiple_checks(self):
        agent = SafetySanityAgent(
            max_active_beliefs=5,
            max_content_length=20,
        )
        beliefs = [_make_belief(content="x" * 30) for _ in range(5)]

        violations = await agent.run_all_checks(beliefs)

        # should have content length + belief count violations
        assert len(violations) >= 2


class TestMetrics:
    def test_computes_correctly(self):
        agent = SafetySanityAgent()

        # create some violations
        agent._record_violation(ViolationType.LowConfidenceUsage, "low", "msg1")
        agent._record_violation(
            ViolationType.RunawayMutation, "high", "msg2", action=ActionType.Block
        )
        agent._record_violation(
            ViolationType.ContentTooLong, "low", "msg3", action=ActionType.Override
        )

        metrics = agent.get_metrics()

        assert metrics.total_violations == 3
        assert metrics.warnings_issued == 1
        assert metrics.blocks_issued == 1
        assert metrics.overrides_issued == 1


class TestGetViolations:
    def test_returns_all(self):
        agent = SafetySanityAgent()
        agent._record_violation(ViolationType.LowConfidenceUsage, "low", "msg")

        violations = agent.get_violations()
        assert len(violations) == 1

    def test_filters_by_time(self):
        agent = SafetySanityAgent()
        agent._record_violation(ViolationType.LowConfidenceUsage, "low", "msg")

        # filter to future time
        future = datetime(2030, 1, 1, tzinfo=timezone.utc)
        violations = agent.get_violations(since=future)

        assert len(violations) == 0


class TestClearVetoes:
    def test_clears_vetoes(self):
        agent = SafetySanityAgent(max_mutation_depth=1, core_tags=["core"])

        parent = _make_belief(content="parent")
        child = _make_belief(content="child", parent_id=parent.id)
        core = _make_belief(tags=["core"])

        agent.check_mutation_depth(child, [parent, child])
        agent.check_core_belief_deprecation([core])

        assert agent.is_mutation_vetoed(child.id)
        assert agent.is_deprecation_vetoed(core.id)

        agent.clear_vetoes()

        assert not agent.is_mutation_vetoed(child.id)
        assert not agent.is_deprecation_vetoed(core.id)
