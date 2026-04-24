# Author: Bradley R. Kinnard
"""
ExperimentOrchestratorAgent - runs scripted scenarios and manages experimental runs.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    Pending = "pending"
    Running = "running"
    Completed = "completed"
    Failed = "failed"
    Cancelled = "cancelled"


class SystemConfig(str, Enum):
    BeliefEcology = "belief_ecology"
    RAGBaseline = "rag_baseline"
    ChatHistoryBaseline = "chat_history"
    StaticMemory = "static_memory"


@dataclass
class ScenarioStep:
    """One step in a scenario."""

    action: str  # "input", "query", "wait", "inject_belief", etc.
    params: dict = field(default_factory=dict)
    expected_outcome: Optional[dict] = None


@dataclass
class Scenario:
    """A scripted experimental scenario."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    steps: list[ScenarioStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    config_overrides: dict = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Record of a single experiment run."""

    id: UUID = field(default_factory=uuid4)
    scenario_id: UUID = field(default_factory=uuid4)
    system_config: SystemConfig = SystemConfig.BeliefEcology
    status: ExperimentStatus = ExperimentStatus.Pending
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    step_results: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExperimentSummary:
    """Summary across multiple experiment runs."""

    total_runs: int
    completed_runs: int
    failed_runs: int
    avg_duration_seconds: float
    metrics_by_config: dict[str, dict]


class ExperimentOrchestratorAgent:
    """
    Experiment Orchestrator Agent (spec 4.1 agent #12).
    Runs scripted scenarios and manages experimental comparisons.
    """

    def __init__(self):
        self._scenarios: dict[UUID, Scenario] = {}
        self._runs: dict[UUID, ExperimentRun] = {}
        self._current_run: Optional[ExperimentRun] = None
        self._step_handlers: dict[str, Callable] = {}

    def register_scenario(self, scenario: Scenario) -> UUID:
        """Register a scenario for later execution."""
        self._scenarios[scenario.id] = scenario
        logger.info(f"registered scenario: {scenario.name} ({scenario.id})")
        return scenario.id

    def get_scenario(self, scenario_id: UUID) -> Optional[Scenario]:
        """Get a registered scenario."""
        return self._scenarios.get(scenario_id)

    def list_scenarios(self, tag: Optional[str] = None) -> list[Scenario]:
        """List scenarios, optionally filtered by tag."""
        scenarios = list(self._scenarios.values())
        if tag:
            scenarios = [s for s in scenarios if tag in s.tags]
        return scenarios

    def register_step_handler(
        self, action: str, handler: Callable[[dict], dict]
    ) -> None:
        """
        Register a handler for a step action type.
        Handler signature: (params: dict) -> result: dict
        """
        self._step_handlers[action] = handler

    async def run_experiment(
        self,
        scenario_id: UUID,
        system_config: SystemConfig = SystemConfig.BeliefEcology,
    ) -> ExperimentRun:
        """
        Execute a scenario with the given system configuration.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"unknown scenario: {scenario_id}")

        run = ExperimentRun(
            scenario_id=scenario_id,
            system_config=system_config,
            status=ExperimentStatus.Running,
            started_at=datetime.now(timezone.utc),
        )
        self._runs[run.id] = run
        self._current_run = run

        logger.info(f"starting experiment: {scenario.name} with {system_config.value}")

        try:
            for i, step in enumerate(scenario.steps):
                result = await self._execute_step(step, i)
                run.step_results.append(result)

                # check for expected outcome
                if step.expected_outcome:
                    self._check_outcome(result, step.expected_outcome, i)

            run.status = ExperimentStatus.Completed
            run.completed_at = datetime.now(timezone.utc)

            # compute metrics
            run.metrics = self._compute_run_metrics(run)

            logger.info(f"experiment completed: {run.id}")

        except Exception as e:
            run.status = ExperimentStatus.Failed
            run.error = str(e)
            run.completed_at = datetime.now(timezone.utc)
            logger.error(f"experiment failed: {e}")

        self._current_run = None
        return run

    async def _execute_step(self, step: ScenarioStep, index: int) -> dict:
        """Execute a single scenario step."""
        handler = self._step_handlers.get(step.action)

        if handler is None:
            logger.warning(f"no handler for action: {step.action}")
            return {"step": index, "action": step.action, "skipped": True}

        try:
            result = handler(step.params)
            return {"step": index, "action": step.action, "result": result}
        except Exception as e:
            return {"step": index, "action": step.action, "error": str(e)}

    def _check_outcome(
        self, result: dict, expected: dict, step_index: int
    ) -> None:
        """Check if step result matches expected outcome."""
        for key, expected_value in expected.items():
            actual = result.get("result", {}).get(key)
            if actual != expected_value:
                logger.warning(
                    f"step {step_index}: expected {key}={expected_value}, got {actual}"
                )

    def _compute_run_metrics(self, run: ExperimentRun) -> dict:
        """Compute metrics for a completed run."""
        duration = 0.0
        if run.started_at and run.completed_at:
            duration = (run.completed_at - run.started_at).total_seconds()

        successful_steps = sum(
            1 for r in run.step_results if "error" not in r and not r.get("skipped")
        )

        return {
            "duration_seconds": duration,
            "total_steps": len(run.step_results),
            "successful_steps": successful_steps,
            "step_success_rate": successful_steps / len(run.step_results) if run.step_results else 0.0,
        }

    def get_run(self, run_id: UUID) -> Optional[ExperimentRun]:
        """Get a specific run."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        scenario_id: Optional[UUID] = None,
        status: Optional[ExperimentStatus] = None,
    ) -> list[ExperimentRun]:
        """List runs, optionally filtered."""
        runs = list(self._runs.values())
        if scenario_id:
            runs = [r for r in runs if r.scenario_id == scenario_id]
        if status:
            runs = [r for r in runs if r.status == status]
        return runs

    async def run_comparison(
        self,
        scenario_id: UUID,
        configs: list[SystemConfig],
    ) -> dict[str, ExperimentRun]:
        """
        Run the same scenario with multiple system configurations.
        Returns mapping of config name to run result.
        """
        results = {}
        for config in configs:
            run = await self.run_experiment(scenario_id, config)
            results[config.value] = run
        return results

    def get_summary(self) -> ExperimentSummary:
        """Get summary statistics across all runs."""
        runs = list(self._runs.values())
        completed = [r for r in runs if r.status == ExperimentStatus.Completed]
        failed = [r for r in runs if r.status == ExperimentStatus.Failed]

        durations = [
            r.metrics.get("duration_seconds", 0) for r in completed if r.metrics
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # group metrics by config
        metrics_by_config: dict[str, dict] = {}
        for run in completed:
            config = run.system_config.value
            if config not in metrics_by_config:
                metrics_by_config[config] = {"runs": 0, "total_steps": 0, "successful_steps": 0}
            metrics_by_config[config]["runs"] += 1
            metrics_by_config[config]["total_steps"] += run.metrics.get("total_steps", 0)
            metrics_by_config[config]["successful_steps"] += run.metrics.get("successful_steps", 0)

        return ExperimentSummary(
            total_runs=len(runs),
            completed_runs=len(completed),
            failed_runs=len(failed),
            avg_duration_seconds=avg_duration,
            metrics_by_config=metrics_by_config,
        )

    def cancel_current(self) -> bool:
        """Cancel the currently running experiment."""
        if self._current_run is None:
            return False

        self._current_run.status = ExperimentStatus.Cancelled
        self._current_run.completed_at = datetime.now(timezone.utc)
        self._current_run = None
        return True


__all__ = [
    "ExperimentOrchestratorAgent",
    "Scenario",
    "ScenarioStep",
    "ExperimentRun",
    "ExperimentStatus",
    "ExperimentSummary",
    "SystemConfig",
]
