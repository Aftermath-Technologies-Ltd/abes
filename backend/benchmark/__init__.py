# Author: Bradley R. Kinnard
"""Benchmark system for evaluating belief ecology performance."""

from .baselines import (
    BaselineMemory,
    BaselineMemorySystem,
    FIFOMemory,
    LRUMemory,
    VectorStoreMemory,
    BASELINE_REGISTRY,
    get_baseline,
)
from .runner import BenchmarkRunner, BenchmarkReport, RunnerConfig
from .scenarios import (
    ScenarioType,
    ScenarioConfig,
    ScenarioResult,
    BenchmarkScenario,
    ContradictionScenario,
    DecayScenario,
    ScaleScenario,
    SCENARIO_REGISTRY,
    get_scenario,
)

__all__ = [
    # baselines
    "BaselineMemory",
    "BaselineMemorySystem",
    "FIFOMemory",
    "LRUMemory",
    "VectorStoreMemory",
    "BASELINE_REGISTRY",
    "get_baseline",
    # runner
    "BenchmarkRunner",
    "BenchmarkReport",
    "RunnerConfig",
    # scenarios
    "ScenarioType",
    "ScenarioConfig",
    "ScenarioResult",
    "BenchmarkScenario",
    "ContradictionScenario",
    "DecayScenario",
    "ScaleScenario",
    "SCENARIO_REGISTRY",
    "get_scenario",
]
