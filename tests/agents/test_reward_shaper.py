# Author: Bradley R. Kinnard
"""Tests for RewardShaperAgent."""

import pytest

from backend.agents.reward_shaper import (
    RewardShaperAgent,
    RewardSignal,
    RewardComponents,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(status: BeliefStatus = BeliefStatus.Active) -> Belief:
    return Belief(
        content="test",
        confidence=0.8,
        status=status,
        origin=OriginMetadata(source="test"),
    )


class TestRewardComponents:
    def test_total(self):
        components = RewardComponents(
            task_success=1.0,
            consistency=0.3,
            memory_efficiency=0.2,
            stability=0.2,
            contradiction_penalty=0.1,
            forgetting_penalty=0.1,
        )
        assert abs(components.total - 1.5) < 0.001


class TestComputeReward:
    def test_basic_reward(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        signal = agent.compute_reward(
            task_success=0.8,
            beliefs=beliefs,
        )

        assert isinstance(signal, RewardSignal)
        assert signal.total_reward > 0
        assert signal.step == 1

    def test_perfect_task_success(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief() for _ in range(500)]  # target count

        signal = agent.compute_reward(
            task_success=1.0,
            beliefs=beliefs,
            consistency_score=1.0,
        )

        assert signal.components.task_success == 1.0
        assert signal.components.consistency > 0

    def test_contradiction_penalty(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        signal = agent.compute_reward(
            task_success=1.0,
            beliefs=beliefs,
            contradiction_errors=5,
        )

        assert signal.components.contradiction_penalty > 0

    def test_forgetting_penalty(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        signal = agent.compute_reward(
            task_success=1.0,
            beliefs=beliefs,
            core_beliefs_lost=2,
        )

        assert signal.components.forgetting_penalty > 0

    def test_stability_bonus(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief() for _ in range(100)]

        # stable count
        signal = agent.compute_reward(
            task_success=1.0,
            beliefs=beliefs,
            previous_belief_count=100,
        )

        assert signal.components.stability > 0

    def test_stability_penalty_for_swing(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief() for _ in range(100)]

        # big swing
        signal = agent.compute_reward(
            task_success=1.0,
            beliefs=beliefs,
            previous_belief_count=50,  # doubled
        )

        # stability should be lower
        assert signal.components.stability < 0.2


class TestEpisodeMetrics:
    def test_episode_return(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        agent.compute_reward(task_success=0.5, beliefs=beliefs)
        agent.compute_reward(task_success=0.5, beliefs=beliefs)

        total = agent.get_episode_return()
        assert total > 0

    def test_average_reward(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        agent.compute_reward(task_success=1.0, beliefs=beliefs)
        agent.compute_reward(task_success=0.0, beliefs=beliefs)

        avg = agent.get_average_reward()
        assert avg > 0

    def test_component_averages(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        agent.compute_reward(task_success=1.0, beliefs=beliefs)
        agent.compute_reward(task_success=0.5, beliefs=beliefs)

        avgs = agent.get_component_averages()
        assert "task_success" in avgs
        assert avgs["task_success"] == 0.75


class TestHistory:
    def test_reward_history(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]

        for _ in range(5):
            agent.compute_reward(task_success=0.5, beliefs=beliefs)

        history = agent.get_reward_history(limit=3)
        assert len(history) == 3

    def test_reset(self):
        agent = RewardShaperAgent()
        beliefs = [_make_belief()]
        agent.compute_reward(task_success=0.5, beliefs=beliefs)

        agent.reset()

        assert len(agent.get_reward_history()) == 0
        assert agent._step == 0
