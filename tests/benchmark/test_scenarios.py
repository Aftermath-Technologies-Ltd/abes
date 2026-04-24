# Author: Bradley R. Kinnard
"""Tests for benchmark scenarios."""

import pytest
from uuid import uuid4

from backend.benchmark.scenarios import (
    ScenarioType,
    ScenarioConfig,
    ScenarioResult,
    ContradictionScenario,
    DecayScenario,
    ScaleScenario,
    get_scenario,
    SCENARIO_REGISTRY,
)


class TestScenarioConfig:
    def test_default_values(self):
        config = ScenarioConfig(
            name="test",
            type=ScenarioType.Mixed,
            description="A test scenario",
        )
        assert config.initial_belief_count == 100
        assert config.iterations == 50
        assert config.success_threshold == 0.8

    def test_custom_values(self):
        config = ScenarioConfig(
            name="custom",
            type=ScenarioType.Scale,
            description="Custom scenario",
            initial_belief_count=500,
            iterations=200,
        )
        assert config.initial_belief_count == 500
        assert config.iterations == 200


class TestContradictionScenario:
    def test_init(self):
        scenario = ContradictionScenario()
        assert scenario.config.name == "contradiction_stress"
        assert scenario.config.type == ScenarioType.Contradiction

    def test_generate_initial_beliefs(self):
        scenario = ContradictionScenario(contradiction_ratio=0.3)
        beliefs = scenario.generate_initial_beliefs()

        assert len(beliefs) == 200  # default count
        facts = [b for b in beliefs if "fact" in b.tags]
        counters = [b for b in beliefs if "counter" in b.tags]

        assert len(counters) == 60  # 30% of 200
        assert len(facts) == 140

    def test_generate_input(self):
        scenario = ContradictionScenario()
        input_text = scenario.generate_input(5)
        assert "iteration 5" in input_text

    def test_evaluate(self):
        scenario = ContradictionScenario()
        result = ScenarioResult(
            scenario_name="test",
            scenario_type="contradiction",
            started_at=None,
            completed_at=None,
            duration_seconds=10.0,
            initial_belief_count=200,
            final_belief_count=180,
            beliefs_created=0,
            beliefs_deprecated=30,
            beliefs_mutated=5,
            final_avg_confidence=0.6,
            final_avg_tension=0.4,
            final_cluster_count=10,
        )

        scores = scenario.evaluate(result)
        assert "tension_stability" in scores
        assert "resolution_rate" in scores


class TestDecayScenario:
    def test_init(self):
        scenario = DecayScenario()
        assert scenario.config.name == "decay_over_time"
        assert scenario.config.type == ScenarioType.Decay

    def test_generate_initial_beliefs(self):
        scenario = DecayScenario()
        beliefs = scenario.generate_initial_beliefs()

        assert len(beliefs) == 100
        assert all(b.confidence == 0.8 for b in beliefs)
        assert all("memory" in b.tags for b in beliefs)

    def test_no_input(self):
        scenario = DecayScenario()
        assert scenario.generate_input(0) == ""

    def test_evaluate(self):
        scenario = DecayScenario()
        result = ScenarioResult(
            scenario_name="test",
            scenario_type="decay",
            started_at=None,
            completed_at=None,
            duration_seconds=10.0,
            initial_belief_count=100,
            final_belief_count=100,
            beliefs_created=0,
            beliefs_deprecated=30,
            beliefs_mutated=0,
            final_avg_confidence=0.5,
            final_avg_tension=0.0,
            final_cluster_count=5,
        )

        scores = scenario.evaluate(result)
        assert "deprecated_ratio" in scores
        assert "confidence_decay" in scores


class TestScaleScenario:
    def test_init(self):
        scenario = ScaleScenario()
        assert scenario.config.name == "scale_test"
        assert scenario.config.type == ScenarioType.Scale

    def test_generate_initial_beliefs(self):
        scenario = ScaleScenario()
        beliefs = scenario.generate_initial_beliefs()

        assert len(beliefs) == 1000
        # check varied confidence
        confidences = {b.confidence for b in beliefs}
        assert len(confidences) > 1

    def test_generate_input(self):
        scenario = ScaleScenario()
        input_text = scenario.generate_input(25)
        assert "topic 25" in input_text

    def test_evaluate(self):
        scenario = ScaleScenario()
        result = ScenarioResult(
            scenario_name="test",
            scenario_type="scale",
            started_at=None,
            completed_at=None,
            duration_seconds=5.0,  # 5 seconds for 50 iterations = 100ms each
            initial_belief_count=1000,
            final_belief_count=2000,
            beliefs_created=1000,
            beliefs_deprecated=0,
            beliefs_mutated=0,
            final_avg_confidence=0.6,
            final_avg_tension=0.1,
            final_cluster_count=50,
        )

        scores = scenario.evaluate(result)
        assert "performance" in scores
        assert "scale_management" in scores


class TestScenarioRegistry:
    def test_registry_populated(self):
        assert "contradiction" in SCENARIO_REGISTRY
        assert "decay" in SCENARIO_REGISTRY
        assert "scale" in SCENARIO_REGISTRY

    def test_get_scenario(self):
        scenario = get_scenario("contradiction")
        assert isinstance(scenario, ContradictionScenario)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError) as exc:
            get_scenario("unknown_scenario")
        assert "Unknown scenario" in str(exc.value)
