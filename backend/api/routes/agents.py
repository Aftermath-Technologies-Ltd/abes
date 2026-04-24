# Author: Bradley R. Kinnard
"""
Agent API routes.
"""

from fastapi import APIRouter, HTTPException

from ..schemas import AgentStatus, AgentListResponse, AgentToggle
from ...agents.scheduler import AgentPhase, DEFAULT_SCHEDULE
from ...core.deps import get_scheduler

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=AgentListResponse)
async def list_agents():
    """List all agents with their status."""
    scheduler = get_scheduler()

    agents = []
    for phase in DEFAULT_SCHEDULE:
        entry = scheduler._agents.get(phase)
        agents.append(
            AgentStatus(
                name=phase.value,
                phase=phase.value,
                enabled=entry.enabled if entry else False,
                last_run=None,
                run_count=0,
            )
        )

    return AgentListResponse(agents=agents)


@router.get("/schedule", response_model=list[str])
async def get_schedule():
    """Get current agent execution order."""
    scheduler = get_scheduler()
    return [p.value for p in scheduler.get_schedule()]


@router.get("/{agent_name}", response_model=AgentStatus)
async def get_agent(agent_name: str):
    """Get status of a specific agent."""
    try:
        phase = AgentPhase(agent_name)
    except ValueError:
        raise HTTPException(404, f"Unknown agent: {agent_name}")

    scheduler = get_scheduler()
    entry = scheduler._agents.get(phase)

    return AgentStatus(
        name=phase.value,
        phase=phase.value,
        enabled=entry.enabled if entry else False,
        last_run=None,
        run_count=0,
    )


@router.patch("/{agent_name}", response_model=AgentStatus)
async def toggle_agent(agent_name: str, req: AgentToggle):
    """Enable or disable an agent."""
    try:
        phase = AgentPhase(agent_name)
    except ValueError:
        raise HTTPException(404, f"Unknown agent: {agent_name}")

    scheduler = get_scheduler()

    if req.enabled:
        scheduler.enable(phase)
    else:
        scheduler.disable(phase)

    entry = scheduler._agents.get(phase)

    return AgentStatus(
        name=phase.value,
        phase=phase.value,
        enabled=entry.enabled if entry else req.enabled,
        last_run=None,
        run_count=0,
    )
    return [p.value for p in scheduler.get_schedule()]
