# Author: Bradley R. Kinnard
"""Tests for ExperimentOrchestratorAgent."""

import pytest
from uuid import uuid4

from backend.agents.experiment_orchestrator import (
    ExperimentOrchestratorAgent,
    Scenario,
    ScenarioStep,
    ExperimentRun,
    ExperimentStatus,
    SystemConfig,
)


class TestScenario:
    def test_create_scenario(self):
        scenario = Scenario(
            name="Test Scenario",
            description="A test",
            steps=[
                ScenarioStep(action="input", params={"text": "hello"}),
                ScenarioStep(action="query", params={"text": "world"}),
            ],
            tags=["test", "simple"],
        )

        assert scenario.name == "Test Scenario"
        assert len(scenario.steps) == 2
        assert "test" in scenario.tags


class TestScenarioRegistration:
    def test_register_scenario(self):
        agent = ExperimentOrchestratorAgent()
        scenario = Scenario(name="Test")

        sid = agent.register_scenario(scenario)

        assert agent.get_scenario(sid) == scenario

    def test_list_scenarios(self):
        agent = ExperimentOrchestratorAgent()
        agent.register_scenario(Scenario(name="A", tags=["qa"]))
        agent.register_scenario(Scenario(name="B", tags=["factual"]))

        all_scenarios = agent.list_scenarios()
        assert len(all_scenarios) == 2

        qa_scenarios = agent.list_scenarios(tag="qa")
        assert len(qa_scenarios) == 1
        assert qa_scenarios[0].name == "A"


class TestStepHandlers:
    def test_register_handler(self):
        agent = ExperimentOrchestratorAgent()

        def handler(params):
            return {"result": params.get("x", 0) * 2}

        agent.register_step_handler("double", handler)

        assert "double" in agent._step_handlers


class TestRunExperiment:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        agent = ExperimentOrchestratorAgent()

        def echo_handler(params):
            return {"echo": params.get("text", "")}

        agent.register_step_handler("echo", echo_handler)

        scenario = Scenario(
            name="Echo Test",
            steps=[
                ScenarioStep(action="echo", params={"text": "hello"}),
                ScenarioStep(action="echo", params={"text": "world"}),
            ],
        )
        agent.register_scenario(scenario)

        run = await agent.run_experiment(scenario.id)

        assert run.status == ExperimentStatus.Completed
        assert len(run.step_results) == 2
        assert run.step_results[0]["result"]["echo"] == "hello"

    @pytest.mark.asyncio
    async def test_unknown_scenario_raises(self):
        agent = ExperimentOrchestratorAgent()

        with pytest.raises(ValueError):
            await agent.run_experiment(uuid4())

    @pytest.mark.asyncio
    async def test_missing_handler_skips(self):
        agent = ExperimentOrchestratorAgent()
        scenario = Scenario(
            name="Missing Handler",
            steps=[ScenarioStep(action="nonexistent", params={})],
        )
        agent.register_scenario(scenario)

        run = await agent.run_experiment(scenario.id)

        assert run.status == ExperimentStatus.Completed
        assert run.step_results[0].get("skipped")

    @pytest.mark.asyncio
    async def test_handler_error_recorded(self):
        agent = ExperimentOrchestratorAgent()

        def failing_handler(params):
            raise ValueError("oops")

        agent.register_step_handler("fail", failing_handler)

        scenario = Scenario(
            name="Failing",
            steps=[ScenarioStep(action="fail", params={})],
        )
        agent.register_scenario(scenario)

        run = await agent.run_experiment(scenario.id)

        assert "error" in run.step_results[0]


class TestMetrics:
    @pytest.mark.asyncio
    async def test_run_metrics(self):
        agent = ExperimentOrchestratorAgent()

        def noop(params):
            return {}

        agent.register_step_handler("noop", noop)

        scenario = Scenario(
            name="Metrics Test",
            steps=[ScenarioStep(action="noop", params={}) for _ in range(5)],
        )
        agent.register_scenario(scenario)

        run = await agent.run_experiment(scenario.id)

        assert run.metrics["total_steps"] == 5
        assert run.metrics["successful_steps"] == 5
        assert run.metrics["step_success_rate"] == 1.0


class TestRunComparison:
    @pytest.mark.asyncio
    async def test_compare_configs(self):
        agent = ExperimentOrchestratorAgent()

        def noop(params):
            return {}

        agent.register_step_handler("noop", noop)

        scenario = Scenario(
            name="Compare",
            steps=[ScenarioStep(action="noop", params={})],
        )
        agent.register_scenario(scenario)

        results = await agent.run_comparison(
            scenario.id,
            [SystemConfig.BeliefEcology, SystemConfig.RAGBaseline],
        )

        assert "belief_ecology" in results
        assert "rag_baseline" in results


class TestListRuns:
    @pytest.mark.asyncio
    async def test_list_all_runs(self):
        agent = ExperimentOrchestratorAgent()
        agent.register_step_handler("noop", lambda p: {})

        scenario = Scenario(name="A", steps=[ScenarioStep(action="noop", params={})])
        agent.register_scenario(scenario)

        await agent.run_experiment(scenario.id)
        await agent.run_experiment(scenario.id)

        runs = agent.list_runs()
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_filter_by_status(self):
        agent = ExperimentOrchestratorAgent()
        agent.register_step_handler("noop", lambda p: {})

        scenario = Scenario(name="A", steps=[ScenarioStep(action="noop", params={})])
        agent.register_scenario(scenario)

        await agent.run_experiment(scenario.id)

        completed = agent.list_runs(status=ExperimentStatus.Completed)
        pending = agent.list_runs(status=ExperimentStatus.Pending)

        assert len(completed) == 1
        assert len(pending) == 0


class TestSummary:
    @pytest.mark.asyncio
    async def test_get_summary(self):
        agent = ExperimentOrchestratorAgent()
        agent.register_step_handler("noop", lambda p: {})

        scenario = Scenario(name="A", steps=[ScenarioStep(action="noop", params={})])
        agent.register_scenario(scenario)

        await agent.run_experiment(scenario.id, SystemConfig.BeliefEcology)
        await agent.run_experiment(scenario.id, SystemConfig.RAGBaseline)

        summary = agent.get_summary()

        assert summary.total_runs == 2
        assert summary.completed_runs == 2
        assert "belief_ecology" in summary.metrics_by_config


class TestCancel:
    def test_cancel_no_current(self):
        agent = ExperimentOrchestratorAgent()
        assert not agent.cancel_current()
