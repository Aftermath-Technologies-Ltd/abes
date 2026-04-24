# Author: Bradley R. Kinnard
"""
Agent Scheduler - orchestrates agent execution per spec 4.2.
Defines execution order and handles conditional agent activation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol
from uuid import UUID

from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


class AgentPhase(str, Enum):
    """Phases of the agent execution pipeline."""

    Perception = "perception"
    Creation = "creation"
    Reinforcement = "reinforcement"
    Decay = "decay"
    Contradiction = "contradiction"
    Mutation = "mutation"
    Resolution = "resolution"
    Relevance = "relevance"
    RLPolicy = "rl_policy"
    Consistency = "consistency"
    Safety = "safety"
    Baseline = "baseline"
    Narrative = "narrative"
    Experiment = "experiment"
    Consolidation = "consolidation"


# Default execution order per spec 4.2, extended with Consolidation
DEFAULT_SCHEDULE = [
    AgentPhase.Perception,
    AgentPhase.Creation,
    AgentPhase.Reinforcement,
    AgentPhase.Decay,
    AgentPhase.Contradiction,
    AgentPhase.Mutation,
    AgentPhase.Resolution,
    AgentPhase.Relevance,
    AgentPhase.RLPolicy,
    AgentPhase.Consistency,
    AgentPhase.Safety,
    AgentPhase.Baseline,
    AgentPhase.Narrative,
    AgentPhase.Experiment,
    AgentPhase.Consolidation,
]


class AgentProtocol(Protocol):
    """Protocol all agents should implement for scheduling."""

    async def run(self, context: "SchedulerContext") -> "AgentResult": ...


@dataclass
class AgentResult:
    """Result from running an agent."""

    phase: AgentPhase
    success: bool
    duration_ms: float
    beliefs_modified: int = 0
    events_emitted: int = 0
    error: Optional[str] = None
    data: dict = field(default_factory=dict)


@dataclass
class SchedulerContext:
    """Shared context passed through the agent pipeline."""

    # input
    raw_input: str = ""
    input_source: str = "chat"

    # belief state
    beliefs: list[Belief] = field(default_factory=list)
    candidate_beliefs: list[str] = field(default_factory=list)

    # tension and contradiction
    tension_map: dict[UUID, float] = field(default_factory=dict)
    contradiction_pairs: list[tuple[UUID, UUID, float]] = field(default_factory=list)

    # RL state
    rl_action: Optional[dict] = None
    rl_state_action: Optional[dict] = None

    # results accumulator
    agent_results: list[AgentResult] = field(default_factory=list)
    events: list = field(default_factory=list)

    # metadata
    iteration: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentEntry:
    """Registration entry for an agent."""

    phase: AgentPhase
    agent: Any  # the actual agent instance
    enabled: bool = True
    run_every_n: int = 1  # run every N iterations (1 = always)
    condition: Optional[Callable[[SchedulerContext], bool]] = None


class AgentScheduler:
    """
    Orchestrates agent execution in defined order (spec 4.2).
    Supports conditional execution and frequency control.
    """

    def __init__(self, schedule: Optional[list[AgentPhase]] = None):
        self._schedule = schedule or DEFAULT_SCHEDULE.copy()
        self._agents: dict[AgentPhase, AgentEntry] = {}
        self._iteration = 0

    def register(
        self,
        phase: AgentPhase,
        agent: Any,
        enabled: bool = True,
        run_every_n: int = 1,
        condition: Optional[Callable[[SchedulerContext], bool]] = None,
    ) -> None:
        """Register an agent for a phase."""
        self._agents[phase] = AgentEntry(
            phase=phase,
            agent=agent,
            enabled=enabled,
            run_every_n=run_every_n,
            condition=condition,
        )
        logger.debug(f"registered agent for {phase.value}")

    def unregister(self, phase: AgentPhase) -> None:
        """Remove an agent from the schedule."""
        if phase in self._agents:
            del self._agents[phase]

    def enable(self, phase: AgentPhase) -> None:
        """Enable an agent."""
        if phase in self._agents:
            self._agents[phase].enabled = True

    def disable(self, phase: AgentPhase) -> None:
        """Disable an agent."""
        if phase in self._agents:
            self._agents[phase].enabled = False

    def set_schedule(self, schedule: list[AgentPhase]) -> None:
        """Override the execution schedule."""
        self._schedule = schedule

    def get_schedule(self) -> list[AgentPhase]:
        """Get current schedule."""
        return self._schedule.copy()

    async def run_iteration(
        self,
        context: Optional[SchedulerContext] = None,
    ) -> SchedulerContext:
        """
        Run one full iteration through all scheduled agents.
        Returns the context with accumulated results.
        """
        self._iteration += 1
        ctx = context or SchedulerContext()
        ctx.iteration = self._iteration

        for phase in self._schedule:
            entry = self._agents.get(phase)
            if entry is None:
                continue

            if not self._should_run(entry, ctx):
                continue

            result = await self._run_agent(entry, ctx)
            ctx.agent_results.append(result)

            if not result.success and result.error:
                logger.warning(f"agent {phase.value} failed: {result.error}")

        return ctx

    def _should_run(self, entry: AgentEntry, ctx: SchedulerContext) -> bool:
        """Check if agent should run this iteration."""
        if not entry.enabled:
            return False

        # frequency check
        if self._iteration % entry.run_every_n != 0:
            return False

        # custom condition
        if entry.condition and not entry.condition(ctx):
            return False

        return True

    async def _run_agent(
        self, entry: AgentEntry, ctx: SchedulerContext
    ) -> AgentResult:
        """Execute a single agent and time it."""
        start = datetime.now(timezone.utc)

        try:
            # check if agent has a standard run method
            if hasattr(entry.agent, "run"):
                result = await entry.agent.run(ctx)
                if isinstance(result, AgentResult):
                    return result

            # fallback: call agent-specific methods based on phase
            result = await self._dispatch_agent(entry, ctx)

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            result.duration_ms = duration
            return result

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            logger.exception(f"agent {entry.phase.value} raised exception")
            return AgentResult(
                phase=entry.phase,
                success=False,
                duration_ms=duration,
                error=str(e),
            )

    async def _dispatch_agent(
        self, entry: AgentEntry, ctx: SchedulerContext
    ) -> AgentResult:
        """Dispatch to agent-specific methods based on phase."""
        agent = entry.agent
        phase = entry.phase

        beliefs_before = len(ctx.beliefs)
        events_before = len(ctx.events)

        if phase == AgentPhase.Perception:
            if hasattr(agent, "ingest"):
                extracted = await agent.ingest(ctx.raw_input, {"source_type": ctx.input_source})
                ctx.candidate_beliefs = extracted
                return AgentResult(
                    phase=phase,
                    success=True,
                    duration_ms=0,
                    data={"extracted_count": len(extracted)},
                )

        elif phase == AgentPhase.Contradiction:
            if hasattr(agent, "audit"):
                events = await agent.audit(ctx.beliefs)
                ctx.events.extend(events)
                # build tension map
                for e in events:
                    if hasattr(e, "belief_id") and hasattr(e, "tension"):
                        ctx.tension_map[e.belief_id] = max(
                            ctx.tension_map.get(e.belief_id, 0.0), e.tension
                        )
                return AgentResult(
                    phase=phase,
                    success=True,
                    duration_ms=0,
                    events_emitted=len(events),
                )

        elif phase == AgentPhase.Decay:
            if hasattr(agent, "process_beliefs"):
                events, modified = await agent.process_beliefs(ctx.beliefs)
                ctx.events.extend(events)
                return AgentResult(
                    phase=phase,
                    success=True,
                    duration_ms=0,
                    beliefs_modified=len(modified),
                    events_emitted=len(events),
                )

        elif phase == AgentPhase.Safety:
            if hasattr(agent, "run_all_checks"):
                violations = await agent.run_all_checks(ctx.beliefs)
                return AgentResult(
                    phase=phase,
                    success=True,
                    duration_ms=0,
                    data={"violations": len(violations)},
                )

        elif phase == AgentPhase.Consolidation:
            if hasattr(agent, "consolidate"):
                events, new_beliefs, deprecated = await agent.consolidate(ctx.beliefs)
                ctx.events.extend(events)
                return AgentResult(
                    phase=phase,
                    success=True,
                    duration_ms=0,
                    beliefs_modified=len(new_beliefs) + len(deprecated),
                    events_emitted=len(events),
                    data={
                        "merged": sum(1 for e in events if e.event_type == "merged"),
                        "compressed": sum(1 for e in events if e.event_type == "compressed"),
                    },
                )

        # generic fallback - just mark success
        return AgentResult(
            phase=phase,
            success=True,
            duration_ms=0,
            beliefs_modified=len(ctx.beliefs) - beliefs_before,
            events_emitted=len(ctx.events) - events_before,
        )

    def get_iteration_count(self) -> int:
        return self._iteration

    def reset(self) -> None:
        """Reset iteration counter."""
        self._iteration = 0


__all__ = [
    "AgentScheduler",
    "AgentPhase",
    "AgentEntry",
    "AgentResult",
    "SchedulerContext",
    "DEFAULT_SCHEDULE",
]
