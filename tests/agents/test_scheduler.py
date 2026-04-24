# Author: Bradley R. Kinnard
"""Tests for Agent Scheduler."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from backend.agents.scheduler import (
    AgentScheduler,
    AgentPhase,
    AgentResult,
    SchedulerContext,
    DEFAULT_SCHEDULE,
)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.run_count = 0

    async def run(self, ctx: SchedulerContext) -> AgentResult:
        self.run_count += 1
        return AgentResult(
            phase=AgentPhase.Perception,
            success=True,
            duration_ms=1.0,
        )


class TestSchedulerContext:
    def test_default_values(self):
        ctx = SchedulerContext()
        assert ctx.raw_input == ""
        assert ctx.beliefs == []
        assert ctx.iteration == 0

    def test_custom_values(self):
        ctx = SchedulerContext(raw_input="hello", iteration=5)
        assert ctx.raw_input == "hello"
        assert ctx.iteration == 5


class TestAgentResult:
    def test_default_values(self):
        result = AgentResult(
            phase=AgentPhase.Perception,
            success=True,
            duration_ms=10.0,
        )
        assert result.phase == AgentPhase.Perception
        assert result.success is True
        assert result.beliefs_modified == 0

    def test_with_error(self):
        result = AgentResult(
            phase=AgentPhase.Safety,
            success=False,
            duration_ms=5.0,
            error="test error",
        )
        assert result.error == "test error"


class TestAgentScheduler:
    def test_init_default_schedule(self):
        scheduler = AgentScheduler()
        assert scheduler.get_schedule() == DEFAULT_SCHEDULE

    def test_custom_schedule(self):
        custom = [AgentPhase.Perception, AgentPhase.Safety]
        scheduler = AgentScheduler(schedule=custom)
        assert scheduler.get_schedule() == custom

    def test_register_agent(self):
        scheduler = AgentScheduler()
        agent = MockAgent()

        scheduler.register(AgentPhase.Perception, agent)

        assert AgentPhase.Perception in scheduler._agents

    def test_unregister_agent(self):
        scheduler = AgentScheduler()
        agent = MockAgent()
        scheduler.register(AgentPhase.Perception, agent)

        scheduler.unregister(AgentPhase.Perception)

        assert AgentPhase.Perception not in scheduler._agents

    def test_enable_disable(self):
        scheduler = AgentScheduler()
        agent = MockAgent()
        scheduler.register(AgentPhase.Perception, agent, enabled=True)

        scheduler.disable(AgentPhase.Perception)
        assert scheduler._agents[AgentPhase.Perception].enabled is False

        scheduler.enable(AgentPhase.Perception)
        assert scheduler._agents[AgentPhase.Perception].enabled is True


class TestRunIteration:
    @pytest.mark.asyncio
    async def test_runs_registered_agents(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])
        agent = MockAgent()
        scheduler.register(AgentPhase.Perception, agent)

        ctx = await scheduler.run_iteration()

        assert agent.run_count == 1
        assert len(ctx.agent_results) == 1

    @pytest.mark.asyncio
    async def test_skips_unregistered_phases(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception, AgentPhase.Safety])
        agent = MockAgent()
        scheduler.register(AgentPhase.Perception, agent)
        # Safety not registered

        ctx = await scheduler.run_iteration()

        assert len(ctx.agent_results) == 1

    @pytest.mark.asyncio
    async def test_skips_disabled_agents(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])
        agent = MockAgent()
        scheduler.register(AgentPhase.Perception, agent, enabled=False)

        ctx = await scheduler.run_iteration()

        assert agent.run_count == 0
        assert len(ctx.agent_results) == 0

    @pytest.mark.asyncio
    async def test_increments_iteration(self):
        scheduler = AgentScheduler(schedule=[])

        await scheduler.run_iteration()
        await scheduler.run_iteration()

        assert scheduler.get_iteration_count() == 2

    @pytest.mark.asyncio
    async def test_run_every_n(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])
        agent = MockAgent()
        scheduler.register(AgentPhase.Perception, agent, run_every_n=2)

        await scheduler.run_iteration()  # iteration 1, skip
        await scheduler.run_iteration()  # iteration 2, run
        await scheduler.run_iteration()  # iteration 3, skip
        await scheduler.run_iteration()  # iteration 4, run

        assert agent.run_count == 2

    @pytest.mark.asyncio
    async def test_conditional_execution(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])
        agent = MockAgent()

        # only run when there's input
        scheduler.register(
            AgentPhase.Perception,
            agent,
            condition=lambda ctx: len(ctx.raw_input) > 0,
        )

        ctx1 = SchedulerContext(raw_input="")
        await scheduler.run_iteration(ctx1)
        assert agent.run_count == 0

        scheduler.reset()
        ctx2 = SchedulerContext(raw_input="hello")
        await scheduler.run_iteration(ctx2)
        assert agent.run_count == 1

    @pytest.mark.asyncio
    async def test_handles_agent_exception(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])

        class FailingAgent:
            async def run(self, ctx):
                raise RuntimeError("boom")

        scheduler.register(AgentPhase.Perception, FailingAgent())

        ctx = await scheduler.run_iteration()

        assert len(ctx.agent_results) == 1
        assert ctx.agent_results[0].success is False
        assert "boom" in ctx.agent_results[0].error

    @pytest.mark.asyncio
    async def test_context_passed_through(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])

        class ContextModifyingAgent:
            async def run(self, ctx):
                ctx.candidate_beliefs = ["belief1", "belief2"]
                return AgentResult(
                    phase=AgentPhase.Perception,
                    success=True,
                    duration_ms=1.0,
                )

        scheduler.register(AgentPhase.Perception, ContextModifyingAgent())

        ctx = SchedulerContext()
        result = await scheduler.run_iteration(ctx)

        assert result.candidate_beliefs == ["belief1", "belief2"]


class TestDispatchAgent:
    @pytest.mark.asyncio
    async def test_dispatch_perception(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Perception])

        class PerceptionLikeAgent:
            async def ingest(self, raw_input, context):
                return ["extracted1", "extracted2"]

        scheduler.register(AgentPhase.Perception, PerceptionLikeAgent())

        ctx = SchedulerContext(raw_input="test input")
        result = await scheduler.run_iteration(ctx)

        assert result.candidate_beliefs == ["extracted1", "extracted2"]
        assert result.agent_results[0].data["extracted_count"] == 2

    @pytest.mark.asyncio
    async def test_dispatch_contradiction(self):
        scheduler = AgentScheduler(schedule=[AgentPhase.Contradiction])

        class ContradictionEvent:
            def __init__(self, bid, tension):
                self.belief_id = bid
                self.tension = tension

        class AuditorLikeAgent:
            async def audit(self, beliefs):
                from uuid import uuid4
                return [ContradictionEvent(uuid4(), 0.8)]

        scheduler.register(AgentPhase.Contradiction, AuditorLikeAgent())

        ctx = SchedulerContext()
        result = await scheduler.run_iteration(ctx)

        assert len(result.events) == 1
        assert result.agent_results[0].events_emitted == 1


class TestReset:
    def test_reset_iteration_counter(self):
        scheduler = AgentScheduler()
        scheduler._iteration = 10

        scheduler.reset()

        assert scheduler.get_iteration_count() == 0
