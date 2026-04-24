# Author: Bradley R. Kinnard
"""Tests for scheduler Consolidation phase and updated schedule."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from backend.agents.scheduler import (
    AgentPhase,
    AgentScheduler,
    AgentResult,
    SchedulerContext,
    DEFAULT_SCHEDULE,
)


class TestConsolidationPhase:
    def test_consolidation_in_default_schedule(self):
        assert AgentPhase.Consolidation in DEFAULT_SCHEDULE

    def test_consolidation_is_last_phase(self):
        assert DEFAULT_SCHEDULE[-1] == AgentPhase.Consolidation

    def test_fifteen_phases_total(self):
        assert len(DEFAULT_SCHEDULE) == 15

    @pytest.mark.asyncio
    async def test_consolidation_agent_dispatched(self):
        """Consolidation agent's consolidate() should be called."""
        mock_agent = MagicMock(spec=[])
        mock_agent.consolidate = AsyncMock(return_value=([], [], []))

        scheduler = AgentScheduler()
        scheduler.register(AgentPhase.Consolidation, mock_agent)

        ctx = SchedulerContext(beliefs=[])
        result_ctx = await scheduler.run_iteration(ctx)

        mock_agent.consolidate.assert_called_once()
        consolidation_results = [
            r for r in result_ctx.agent_results
            if r.phase == AgentPhase.Consolidation
        ]
        assert len(consolidation_results) == 1
        assert consolidation_results[0].success is True

    @pytest.mark.asyncio
    async def test_consolidation_reports_stats(self):
        from backend.agents.consolidation import ConsolidationEvent
        from uuid import uuid4

        mock_events = [
            ConsolidationEvent(
                event_type="merged",
                affected_ids=[uuid4(), uuid4()],
                result_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
            ),
            ConsolidationEvent(
                event_type="compressed",
                affected_ids=[uuid4(), uuid4(), uuid4()],
                result_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        mock_agent = MagicMock(spec=[])
        mock_agent.consolidate = AsyncMock(return_value=(mock_events, [], []))

        scheduler = AgentScheduler()
        scheduler.register(AgentPhase.Consolidation, mock_agent)

        ctx = SchedulerContext(beliefs=[])
        result_ctx = await scheduler.run_iteration(ctx)

        consolidation_result = [
            r for r in result_ctx.agent_results
            if r.phase == AgentPhase.Consolidation
        ][0]
        assert consolidation_result.data["merged"] == 1
        assert consolidation_result.data["compressed"] == 1


class TestSchedulerPhaseEnum:
    def test_all_expected_phases_exist(self):
        expected = [
            "perception", "creation", "reinforcement", "decay",
            "contradiction", "mutation", "resolution", "relevance",
            "rl_policy", "consistency", "safety", "baseline",
            "narrative", "experiment", "consolidation",
        ]
        actual = [p.value for p in AgentPhase]
        for phase_name in expected:
            assert phase_name in actual, f"Missing phase: {phase_name}"
