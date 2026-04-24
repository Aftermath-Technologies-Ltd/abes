# Author: Bradley R. Kinnard
"""
Benchmark runner for executing scenarios and collecting results.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import UUID

from .scenarios import (
    BenchmarkScenario,
    ScenarioResult,
    get_scenario,
)
from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the benchmark runner."""
    save_snapshots: bool = True
    log_progress: bool = True
    progress_interval: int = 10  # log every N iterations
    timeout_seconds: Optional[float] = None


@dataclass
class BenchmarkReport:
    """Full benchmark report with all scenario results."""
    run_id: UUID
    started_at: datetime
    completed_at: datetime
    total_duration_seconds: float
    scenarios_run: int
    scenarios_passed: int
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.scenarios_run == 0:
            return 0.0
        return self.scenarios_passed / self.scenarios_run


class BenchmarkRunner:
    """
    Runs benchmark scenarios against the belief ecology.
    """

    def __init__(
        self,
        bel=None,
        belief_store=None,
        snapshot_store=None,
        config: Optional[RunnerConfig] = None,
    ):
        self._bel = bel
        self._belief_store = belief_store
        self._snapshot_store = snapshot_store
        self._config = config or RunnerConfig()

        self._on_progress: Optional[Callable[[int, int, str], None]] = None
        self._on_iteration: Optional[Callable[[int, float], None]] = None

    def set_progress_callback(
        self, callback: Callable[[int, int, str], None]
    ) -> None:
        """Set callback for progress updates: (current, total, message)."""
        self._on_progress = callback

    def set_iteration_callback(
        self, callback: Callable[[int, float], None]
    ) -> None:
        """Set callback for iteration updates: (iteration, duration_ms)."""
        self._on_iteration = callback

    async def run_scenario(
        self, scenario: BenchmarkScenario
    ) -> ScenarioResult:
        """Run a single benchmark scenario."""
        config = scenario.config
        started_at = datetime.now(timezone.utc)

        if self._config.log_progress:
            logger.info(f"starting scenario: {config.name}")

        # generate and load initial beliefs
        initial_beliefs = scenario.generate_initial_beliefs()
        for belief in initial_beliefs:
            await self._belief_store.create(belief)

        initial_count = len(initial_beliefs)

        # track metrics
        snapshot_ids: list[UUID] = []
        beliefs_created = 0
        beliefs_mutated = 0

        # run iterations
        for i in range(config.iterations):
            iter_start = time.perf_counter()

            # generate input
            input_text = scenario.generate_input(i)

            # run BEL iteration (if we have one)
            if self._bel is not None and input_text:
                snapshot = await self._bel.run_iteration(context=input_text)
                if self._config.save_snapshots and self._snapshot_store:
                    await self._snapshot_store.save(snapshot)
                    snapshot_ids.append(snapshot.id)

            iter_duration = (time.perf_counter() - iter_start) * 1000

            if self._on_iteration:
                self._on_iteration(i, iter_duration)

            if self._config.log_progress and (i + 1) % self._config.progress_interval == 0:
                logger.info(f"  iteration {i + 1}/{config.iterations}")

        # collect final stats
        all_beliefs = await self._belief_store.list(limit=50000)
        active = [b for b in all_beliefs if b.status == BeliefStatus.Active]
        deprecated = [b for b in all_beliefs if b.status == BeliefStatus.Deprecated]
        mutated = [b for b in all_beliefs if b.status == BeliefStatus.Mutated]

        # compute averages
        avg_confidence = (
            sum(b.confidence for b in active) / len(active) if active else 0.0
        )
        avg_tension = (
            sum(b.tension for b in active) / len(active) if active else 0.0
        )

        # count clusters
        cluster_ids = {b.cluster_id for b in all_beliefs if b.cluster_id}

        completed_at = datetime.now(timezone.utc)
        duration = (completed_at - started_at).total_seconds()

        result = ScenarioResult(
            scenario_name=config.name,
            scenario_type=config.type.value,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            initial_belief_count=initial_count,
            final_belief_count=len(all_beliefs),
            beliefs_created=len(all_beliefs) - initial_count,
            beliefs_deprecated=len(deprecated),
            beliefs_mutated=len(mutated),
            final_avg_confidence=avg_confidence,
            final_avg_tension=avg_tension,
            final_cluster_count=len(cluster_ids),
            snapshot_ids=snapshot_ids,
        )

        # evaluate against targets
        scores = scenario.evaluate(result)
        for metric, score in scores.items():
            target = config.target_metrics.get(metric)
            result.targets_met[metric] = score >= config.success_threshold

        met_count = sum(1 for v in result.targets_met.values() if v)
        result.success_rate = met_count / len(result.targets_met) if result.targets_met else 0.0

        if self._config.log_progress:
            logger.info(f"completed scenario: {config.name} (success: {result.success_rate:.0%})")

        return result

    async def run_scenarios(
        self, scenario_names: list[str], **kwargs
    ) -> BenchmarkReport:
        """Run multiple scenarios and generate a report."""
        from uuid import uuid4

        run_id = uuid4()
        started_at = datetime.now(timezone.utc)
        results: list[ScenarioResult] = []
        passed = 0

        for i, name in enumerate(scenario_names):
            if self._on_progress:
                self._on_progress(i + 1, len(scenario_names), f"Running {name}")

            scenario = get_scenario(name, **kwargs.get(name, {}))
            result = await self.run_scenario(scenario)
            results.append(result)

            if result.success_rate >= scenario.config.success_threshold:
                passed += 1

        completed_at = datetime.now(timezone.utc)
        duration = (completed_at - started_at).total_seconds()

        return BenchmarkReport(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=duration,
            scenarios_run=len(scenario_names),
            scenarios_passed=passed,
            results=results,
        )

    async def run_all(self, **kwargs) -> BenchmarkReport:
        """Run all registered scenarios."""
        from .scenarios import SCENARIO_REGISTRY
        return await self.run_scenarios(list(SCENARIO_REGISTRY.keys()), **kwargs)


__all__ = [
    "RunnerConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
]
