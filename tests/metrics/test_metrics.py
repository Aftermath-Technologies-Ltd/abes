# Author: Bradley R. Kinnard
"""Tests for metrics system."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.metrics import (
    EcologyMetrics,
    AgentMetrics,
    IterationMetrics,
    MetricsComputer,
    MetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)
from backend.metrics.export import MetricsExporter
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


@pytest.fixture
def sample_beliefs():
    """Create sample beliefs for testing."""
    cluster_id = uuid4()
    return [
        Belief(
            content="Active belief 1",
            confidence=0.8,
            origin=OriginMetadata(source="test"),
            tension=0.2,
            status=BeliefStatus.Active,
            cluster_id=cluster_id,
        ),
        Belief(
            content="Active belief 2",
            confidence=0.6,
            origin=OriginMetadata(source="test"),
            tension=0.4,
            status=BeliefStatus.Active,
            cluster_id=cluster_id,
        ),
        Belief(
            content="Decaying belief",
            confidence=0.25,
            origin=OriginMetadata(source="test"),
            tension=0.3,
            status=BeliefStatus.Decaying,
        ),
        Belief(
            content="Deprecated belief",
            confidence=0.05,
            origin=OriginMetadata(source="test"),
            tension=0.0,
            status=BeliefStatus.Deprecated,
        ),
    ]


@pytest.fixture
def collector():
    reset_metrics_collector()
    return MetricsCollector()


class TestMetricsComputer:
    def test_empty_beliefs(self):
        computer = MetricsComputer()
        metrics = computer.compute_ecology_metrics([])

        assert metrics.total_beliefs == 0
        assert metrics.avg_confidence == 0.0

    def test_counts(self, sample_beliefs):
        computer = MetricsComputer()
        metrics = computer.compute_ecology_metrics(sample_beliefs)

        assert metrics.total_beliefs == 4
        assert metrics.active_beliefs == 2
        assert metrics.decaying_beliefs == 1
        assert metrics.deprecated_beliefs == 1

    def test_averages(self, sample_beliefs):
        computer = MetricsComputer()
        metrics = computer.compute_ecology_metrics(sample_beliefs)

        expected_confidence = (0.8 + 0.6 + 0.25 + 0.05) / 4
        assert abs(metrics.avg_confidence - expected_confidence) < 0.001

        expected_tension = (0.2 + 0.4 + 0.3 + 0.0) / 4
        assert abs(metrics.avg_tension - expected_tension) < 0.001

    def test_extremes(self, sample_beliefs):
        computer = MetricsComputer()
        metrics = computer.compute_ecology_metrics(sample_beliefs)

        assert metrics.min_confidence == 0.05
        assert metrics.max_confidence == 0.8
        assert metrics.min_tension == 0.0
        assert metrics.max_tension == 0.4

    def test_clustering(self, sample_beliefs):
        computer = MetricsComputer()
        metrics = computer.compute_ecology_metrics(sample_beliefs)

        assert metrics.cluster_count == 1
        assert metrics.orphan_beliefs == 2  # decaying and deprecated have no cluster
        assert metrics.avg_cluster_size == 2.0  # 2 beliefs in 1 cluster

    def test_mutation_depth(self):
        computer = MetricsComputer()

        # create chain: root -> child -> grandchild
        root = Belief(
            content="Root",
            confidence=0.8,
            origin=OriginMetadata(source="test"),
        )
        child = Belief(
            content="Child",
            confidence=0.7,
            origin=OriginMetadata(source="test"),
            parent_id=root.id,
        )
        grandchild = Belief(
            content="Grandchild",
            confidence=0.6,
            origin=OriginMetadata(source="test"),
            parent_id=child.id,
        )

        metrics = computer.compute_ecology_metrics([root, child, grandchild])

        assert metrics.max_mutation_depth == 2
        assert metrics.beliefs_with_parents == 2


class TestMetricsCollector:
    def test_record_ecology(self, collector, sample_beliefs):
        metrics = collector.record_ecology(sample_beliefs)

        assert metrics.total_beliefs == 4
        assert len(collector.get_ecology_history()) == 1

    def test_record_iteration(self, collector):
        iteration = IterationMetrics(
            iteration=1,
            timestamp=datetime.now(timezone.utc),
            duration_ms=100.5,
            beliefs_created=5,
        )
        collector.record_iteration(iteration)

        history = collector.get_iteration_history()
        assert len(history) == 1
        assert history[0].beliefs_created == 5

    def test_record_agent_run(self, collector):
        collector.record_agent_run(
            "perception",
            duration_ms=50.0,
            beliefs_processed=10,
            actions_taken=3,
        )
        collector.record_agent_run(
            "perception",
            duration_ms=70.0,
            beliefs_processed=15,
        )

        agents = collector.get_agent_metrics()
        assert "perception" in agents

        am = agents["perception"]
        assert am.run_count == 2
        assert am.avg_duration_ms == 60.0  # (50 + 70) / 2
        assert am.beliefs_processed == 25
        assert am.actions_taken == 3

    def test_agent_errors(self, collector):
        collector.record_agent_run("decay", duration_ms=10, error=True)
        collector.record_agent_run("decay", duration_ms=10, error=False)

        am = collector.get_agent_metrics()["decay"]
        assert am.errors == 1
        assert am.run_count == 2

    def test_history_limit(self, collector, sample_beliefs):
        small_collector = MetricsCollector(max_history=5)

        for _ in range(10):
            small_collector.record_ecology(sample_beliefs)

        assert len(small_collector.get_ecology_history()) == 5

    def test_get_latest(self, collector, sample_beliefs):
        assert collector.get_latest_ecology() is None

        collector.record_ecology(sample_beliefs)
        latest = collector.get_latest_ecology()

        assert latest is not None
        assert latest.total_beliefs == 4

    def test_clear(self, collector, sample_beliefs):
        collector.record_ecology(sample_beliefs)
        collector.record_agent_run("test", 10)

        collector.clear()

        assert len(collector.get_ecology_history()) == 0
        assert len(collector.get_agent_metrics()) == 0


class TestMetricsSingleton:
    def test_get_singleton(self):
        reset_metrics_collector()
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2

    def test_reset(self):
        c1 = get_metrics_collector()
        reset_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is not c2


class TestMetricsExporter:
    @pytest.fixture
    def exporter(self, collector, sample_beliefs):
        collector.record_ecology(sample_beliefs)
        collector.record_agent_run("perception", 50.0, 10, 3)
        return MetricsExporter(collector)

    def test_export_ecology_json(self, exporter):
        json_str = exporter.export_ecology_json()

        import json
        data = json.loads(json_str)

        assert len(data) == 1
        assert data[0]["total_beliefs"] == 4

    def test_export_agents_json(self, exporter):
        json_str = exporter.export_agents_json()

        import json
        data = json.loads(json_str)

        assert "perception" in data
        assert data["perception"]["run_count"] == 1

    def test_export_ecology_csv(self, exporter):
        csv_str = exporter.export_ecology_csv()

        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "total_beliefs" in lines[0]

    def test_export_prometheus(self, exporter):
        prom = exporter.export_prometheus()

        assert "abes_beliefs_total 4" in prom
        assert 'abes_agent_runs_total{agent="perception"} 1' in prom

    def test_export_summary(self, exporter):
        summary = exporter.export_summary()

        assert "ecology" in summary
        assert summary["ecology"]["total_beliefs"] == 4
        assert "perception" in summary["agents"]
