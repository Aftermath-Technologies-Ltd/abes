# Author: Bradley R. Kinnard
"""Tests for RLPolicyAgent."""

import pytest

from backend.agents.rl_policy import (
    RLPolicyAgent,
    EcologyState,
    PolicyAction,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "test",
    confidence: float = 0.8,
    tension: float = 0.0,
    status: BeliefStatus = BeliefStatus.Active,
    cluster_id=None,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        status=status,
        origin=OriginMetadata(source="test"),
        cluster_id=cluster_id,
    )


class TestEcologyState:
    def test_empty_beliefs(self):
        state = EcologyState.from_beliefs([])
        assert state.total_beliefs == 0
        assert state.confidence_mean == 0.0

    def test_computes_stats(self):
        beliefs = [
            _make_belief(confidence=0.6, tension=0.2),
            _make_belief(confidence=0.8, tension=0.4),
        ]
        state = EcologyState.from_beliefs(beliefs)

        assert state.total_beliefs == 2
        assert abs(state.confidence_mean - 0.7) < 0.001
        assert state.confidence_min == 0.6
        assert state.confidence_max == 0.8
        assert abs(state.tension_mean - 0.3) < 0.001
        assert state.tension_max == 0.4

    def test_counts_by_status(self):
        beliefs = [
            _make_belief(status=BeliefStatus.Active),
            _make_belief(status=BeliefStatus.Active),
            _make_belief(status=BeliefStatus.Decaying),
            _make_belief(status=BeliefStatus.Deprecated),
        ]
        state = EcologyState.from_beliefs(beliefs)

        assert state.active_count == 2
        assert state.decaying_count == 1
        assert state.deprecated_count == 1

    def test_high_tension_count(self):
        beliefs = [
            _make_belief(tension=0.2),
            _make_belief(tension=0.7),
            _make_belief(tension=0.9),
        ]
        state = EcologyState.from_beliefs(beliefs)

        assert state.high_tension_count == 2

    def test_cluster_count(self):
        from uuid import uuid4

        c1, c2 = uuid4(), uuid4()
        beliefs = [
            _make_belief(cluster_id=c1),
            _make_belief(cluster_id=c1),
            _make_belief(cluster_id=c2),
            _make_belief(cluster_id=None),
        ]
        state = EcologyState.from_beliefs(beliefs)

        assert state.cluster_count == 2

    def test_task_context_passed(self):
        state = EcologyState.from_beliefs(
            [], episode_step=5, task_type="qa"
        )
        assert state.episode_step == 5
        assert state.task_type == "qa"


class TestPolicyAction:
    def test_default_values(self):
        action = PolicyAction()
        assert action.beliefs_to_surface == 5
        assert action.ranking_weights == {}


class TestRLPolicyAgent:
    def test_get_action_returns_policy_action(self):
        agent = RLPolicyAgent()
        state = EcologyState.from_beliefs([_make_belief()])

        action = agent.get_action(state)

        assert isinstance(action, PolicyAction)

    def test_high_tension_adjusts_thresholds(self):
        agent = RLPolicyAgent()
        beliefs = [_make_belief(tension=0.9) for _ in range(6)]
        state = EcologyState.from_beliefs(beliefs)

        action = agent.get_action(state)

        assert action.mutation_threshold <= 0.5
        assert action.resolution_threshold <= 0.6

    def test_action_history_recorded(self):
        agent = RLPolicyAgent()
        state = EcologyState.from_beliefs([_make_belief()])

        agent.get_action(state)
        agent.get_action(state)

        history = agent.get_action_history()
        assert len(history) == 2

    def test_reset_clears_history(self):
        agent = RLPolicyAgent()
        state = EcologyState.from_beliefs([])
        agent.get_action(state)

        agent.reset()

        assert len(agent.get_action_history()) == 0

    def test_set_policy_weights(self):
        agent = RLPolicyAgent()
        agent.set_policy_weights({"layer1": [1, 2, 3]})

        assert agent._policy_weights is not None
