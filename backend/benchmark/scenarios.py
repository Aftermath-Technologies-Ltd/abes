# Author: Bradley R. Kinnard
"""
Benchmark scenario definitions per spec 5.1.
Each scenario tests the belief ecology under specific conditions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable
from uuid import UUID, uuid4

from ..core.models.belief import Belief


class ScenarioType(str, Enum):
    """Categories of benchmark scenarios."""
    Contradiction = "contradiction"
    Decay = "decay"
    Reinforcement = "reinforcement"
    Clustering = "clustering"
    Mutation = "mutation"
    Scale = "scale"
    Mixed = "mixed"


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""
    name: str
    type: ScenarioType
    description: str

    # belief generation
    initial_belief_count: int = 100
    belief_content_template: str = "Belief {i}: {topic}"
    topics: list[str] = field(default_factory=lambda: ["general"])

    # iteration params
    iterations: int = 50
    inputs_per_iteration: int = 5

    # expected behavior
    target_metrics: dict[str, float] = field(default_factory=dict)
    success_threshold: float = 0.8  # % of targets to hit

    # random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class ScenarioResult:
    """Result from running a benchmark scenario."""
    scenario_name: str
    scenario_type: str

    # timing
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

    # belief stats
    initial_belief_count: int
    final_belief_count: int
    beliefs_created: int
    beliefs_deprecated: int
    beliefs_mutated: int

    # metrics
    final_avg_confidence: float
    final_avg_tension: float
    final_cluster_count: int

    # target comparison
    targets_met: dict[str, bool] = field(default_factory=dict)
    success_rate: float = 0.0

    # snapshots
    snapshot_ids: list[UUID] = field(default_factory=list)


class BenchmarkScenario(ABC):
    """Base class for benchmark scenarios."""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self._id = uuid4()

    @property
    def id(self) -> UUID:
        return self._id

    @abstractmethod
    def generate_initial_beliefs(self) -> list[Belief]:
        """Generate initial beliefs for the scenario."""
        ...

    @abstractmethod
    def generate_input(self, iteration: int) -> str:
        """Generate input for an iteration."""
        ...

    @abstractmethod
    def evaluate(self, result: ScenarioResult) -> dict[str, float]:
        """Evaluate scenario results against targets."""
        ...


class ContradictionScenario(BenchmarkScenario):
    """Tests contradiction detection and resolution."""

    def __init__(
        self,
        contradiction_ratio: float = 0.3,  # % of beliefs that contradict
        **kwargs,
    ):
        config = ScenarioConfig(
            name="contradiction_stress",
            type=ScenarioType.Contradiction,
            description="Tests contradiction detection under high tension",
            initial_belief_count=200,
            iterations=30,
            target_metrics={
                "avg_tension": 0.4,  # should stabilize around this
                "resolution_rate": 0.8,  # % contradictions resolved
            },
            **kwargs,
        )
        super().__init__(config)
        self._ratio = contradiction_ratio

    def generate_initial_beliefs(self) -> list[Belief]:
        from ..core.models.belief import OriginMetadata

        beliefs = []
        n = self.config.initial_belief_count
        contradicting = int(n * self._ratio)

        # base beliefs
        for i in range(n - contradicting):
            beliefs.append(Belief(
                content=f"Fact {i}: The value of X is {i}",
                confidence=0.7,
                origin=OriginMetadata(source="benchmark"),
                tags=["fact"],
            ))

        # contradicting beliefs
        for i in range(contradicting):
            base_i = i % (n - contradicting)
            beliefs.append(Belief(
                content=f"Counter {i}: The value of X is NOT {base_i}",
                confidence=0.6,
                origin=OriginMetadata(source="benchmark"),
                tags=["counter"],
            ))

        return beliefs

    def generate_input(self, iteration: int) -> str:
        return f"What is the value of X at iteration {iteration}?"

    def evaluate(self, result: ScenarioResult) -> dict[str, float]:
        scores = {}

        # tension should be moderate, not extreme
        tension_target = self.config.target_metrics.get("avg_tension", 0.4)
        tension_diff = abs(result.final_avg_tension - tension_target)
        scores["tension_stability"] = max(0, 1 - tension_diff / tension_target)

        # resolution rate
        if result.initial_belief_count > 0:
            resolution = result.beliefs_deprecated / (result.initial_belief_count * self._ratio)
            scores["resolution_rate"] = min(1.0, resolution)
        else:
            scores["resolution_rate"] = 0.0

        return scores


class DecayScenario(BenchmarkScenario):
    """Tests decay behavior over time."""

    def __init__(self, decay_rate: float = 0.995, **kwargs):
        config = ScenarioConfig(
            name="decay_over_time",
            type=ScenarioType.Decay,
            description="Tests confidence decay without reinforcement",
            initial_belief_count=100,
            iterations=100,
            inputs_per_iteration=0,  # no input, just decay
            target_metrics={
                "deprecated_ratio": 0.3,  # expect ~30% deprecated
                "avg_confidence_drop": 0.3,  # expect ~30% confidence drop
            },
            **kwargs,
        )
        super().__init__(config)
        self._decay_rate = decay_rate

    def generate_initial_beliefs(self) -> list[Belief]:
        from ..core.models.belief import OriginMetadata

        return [
            Belief(
                content=f"Memory {i}: Something from the past",
                confidence=0.8,
                origin=OriginMetadata(source="benchmark"),
                tags=["memory"],
            )
            for i in range(self.config.initial_belief_count)
        ]

    def generate_input(self, iteration: int) -> str:
        return ""  # no input for decay scenario

    def evaluate(self, result: ScenarioResult) -> dict[str, float]:
        scores = {}

        # deprecated ratio
        if result.initial_belief_count > 0:
            deprecated_ratio = result.beliefs_deprecated / result.initial_belief_count
            target = self.config.target_metrics.get("deprecated_ratio", 0.3)
            scores["deprecated_ratio"] = 1 - abs(deprecated_ratio - target) / target
        else:
            scores["deprecated_ratio"] = 0.0

        # confidence drop
        initial_confidence = 0.8  # we set this in generation
        confidence_drop = initial_confidence - result.final_avg_confidence
        target_drop = self.config.target_metrics.get("avg_confidence_drop", 0.3)
        scores["confidence_decay"] = 1 - abs(confidence_drop - target_drop) / target_drop

        return scores


class ScaleScenario(BenchmarkScenario):
    """Tests system under high belief count."""

    def __init__(self, max_beliefs: int = 5000, **kwargs):
        config = ScenarioConfig(
            name="scale_test",
            type=ScenarioType.Scale,
            description="Tests performance under high belief count",
            initial_belief_count=1000,
            iterations=50,
            inputs_per_iteration=10,
            target_metrics={
                "max_iteration_time_ms": 1000,  # 1 second max per iteration
                "memory_growth_factor": 2.0,  # should not grow > 2x
            },
            **kwargs,
        )
        super().__init__(config)
        self._max_beliefs = max_beliefs

    def generate_initial_beliefs(self) -> list[Belief]:
        from ..core.models.belief import OriginMetadata

        return [
            Belief(
                content=f"Data point {i}: Observation about topic {i % 50}",
                confidence=0.6 + (i % 40) / 100,
                origin=OriginMetadata(source="benchmark"),
                tags=[f"topic_{i % 50}"],
            )
            for i in range(self.config.initial_belief_count)
        ]

    def generate_input(self, iteration: int) -> str:
        topic = iteration % 50
        return f"Tell me about topic {topic}"

    def evaluate(self, result: ScenarioResult) -> dict[str, float]:
        scores = {}

        # performance based on iteration time
        avg_time_ms = (result.duration_seconds * 1000) / max(1, self.config.iterations)
        target_time = self.config.target_metrics.get("max_iteration_time_ms", 1000)
        scores["performance"] = min(1.0, target_time / max(1, avg_time_ms))

        # belief count management
        if result.final_belief_count <= self._max_beliefs:
            scores["scale_management"] = 1.0
        else:
            overage = result.final_belief_count - self._max_beliefs
            scores["scale_management"] = max(0, 1 - overage / self._max_beliefs)

        return scores


# registry of available scenarios
SCENARIO_REGISTRY: dict[str, type[BenchmarkScenario]] = {
    "contradiction": ContradictionScenario,
    "decay": DecayScenario,
    "scale": ScaleScenario,
}


def get_scenario(name: str, **kwargs) -> BenchmarkScenario:
    """Get a scenario by name."""
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_REGISTRY.keys())}")
    return SCENARIO_REGISTRY[name](**kwargs)


__all__ = [
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
