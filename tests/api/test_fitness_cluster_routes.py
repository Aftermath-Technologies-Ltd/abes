# Author: Bradley R. Kinnard
"""Tests for /beliefs/fitness, /beliefs/clusters, and /rl/status endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.core.deps import reset_singletons, get_rl_state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset singletons and inject a fresh in-memory store before each test."""
    from backend.storage import InMemoryBeliefStore
    import backend.core.deps as deps
    reset_singletons()
    deps._belief_store = InMemoryBeliefStore()
    yield
    reset_singletons()


@pytest.fixture
def client():
    return TestClient(app)


def _seed_beliefs(n: int = 5) -> list[dict]:
    """Create n beliefs via the API and return their response dicts."""
    c = TestClient(app)
    created = []
    for i in range(n):
        resp = c.post("/beliefs", json={
            "content": f"belief number {i} about alpha beta gamma",
            "confidence": round(0.3 + i * 0.1, 1),
        })
        assert resp.status_code == 201, resp.text
        created.append(resp.json())
    return created


class TestFitnessEndpoint:
    """GET /beliefs/fitness schema and sort order."""

    def test_returns_200(self, client):
        response = client.get("/beliefs/fitness")
        assert response.status_code == 200

    def test_response_has_required_keys(self, client):
        data = client.get("/beliefs/fitness").json()
        assert "beliefs" in data
        assert "stats" in data
        assert "total" in data["stats"]
        assert "mean_fitness" in data["stats"]

    def test_empty_store_returns_empty_beliefs(self, client):
        data = client.get("/beliefs/fitness").json()
        assert data["beliefs"] == []
        assert data["stats"]["total"] == 0
        assert data["stats"]["mean_fitness"] == 0.0

    def test_each_belief_entry_has_required_fields(self, client):
        _seed_beliefs(3)
        data = client.get("/beliefs/fitness").json()
        assert len(data["beliefs"]) == 3
        for entry in data["beliefs"]:
            assert "id" in entry
            assert "content_summary" in entry
            assert "fitness" in entry
            assert "tier" in entry
            assert "tension" in entry

    def test_sorted_fitness_descending(self, client):
        _seed_beliefs(5)
        data = client.get("/beliefs/fitness").json()
        fitnesses = [e["fitness"] for e in data["beliefs"]]
        assert fitnesses == sorted(fitnesses, reverse=True), \
            "Beliefs must be sorted by fitness descending"

    def test_content_summary_max_100_chars(self, client):
        long_content = "word " * 50  # 250 chars
        client.post("/beliefs", json={"content": long_content, "confidence": 0.8})
        data = client.get("/beliefs/fitness").json()
        for entry in data["beliefs"]:
            assert len(entry["content_summary"]) <= 100

    def test_stats_total_matches_belief_count(self, client):
        _seed_beliefs(4)
        data = client.get("/beliefs/fitness").json()
        assert data["stats"]["total"] == len(data["beliefs"])


class TestClustersEndpoint:
    """GET /beliefs/clusters schema and grouping."""

    def test_returns_200(self, client):
        response = client.get("/beliefs/clusters")
        assert response.status_code == 200

    def test_response_has_clusters_key(self, client):
        data = client.get("/beliefs/clusters").json()
        assert "clusters" in data
        assert isinstance(data["clusters"], list)

    def test_empty_store_returns_empty_clusters(self, client):
        data = client.get("/beliefs/clusters").json()
        assert data["clusters"] == []

    def test_each_cluster_entry_has_required_fields(self, client):
        # Seed beliefs with shared tokens so the clustering threshold triggers
        for i in range(4):
            client.post("/beliefs", json={
                "content": f"the sky is blue today number {i}",
                "confidence": 0.7,
            })

        data = client.get("/beliefs/clusters").json()
        if data["clusters"]:
            for cluster in data["clusters"]:
                assert "cluster_id" in cluster
                assert "size" in cluster
                assert "mean_tension" in cluster
                assert "mean_fitness" in cluster
                assert "top_belief_summary" in cluster
                assert "belief_ids" in cluster


class TestRLStatusEndpoint:
    """GET /rl/status schema and value reflection."""

    def test_returns_200(self, client):
        response = client.get("/rl/status")
        assert response.status_code == 200

    def test_response_has_required_keys(self, client):
        data = client.get("/rl/status").json()
        required = {
            "policy_trained",
            "total_trajectories",
            "last_training_epoch_loss",
            "current_action_means",
            "training_interval",
            "iterations_until_next_training",
        }
        assert required.issubset(data.keys())

    def test_default_state_untrained(self, client):
        data = client.get("/rl/status").json()
        assert data["policy_trained"] is False
        assert data["total_trajectories"] == 0
        assert data["last_training_epoch_loss"] is None

    def test_action_means_is_list_of_7(self, client):
        data = client.get("/rl/status").json()
        assert isinstance(data["current_action_means"], list)
        assert len(data["current_action_means"]) == 7

    def test_state_mutation_reflected_in_response(self, client):
        state = get_rl_state()
        state.policy_trained = True
        state.total_trajectories = 250
        state.last_training_epoch_loss = 0.023

        data = client.get("/rl/status").json()
        assert data["policy_trained"] is True
        assert data["total_trajectories"] == 250
        assert abs(data["last_training_epoch_loss"] - 0.023) < 1e-6
