# Author: Bradley R. Kinnard
"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from backend.api.app import app
from backend.core.deps import reset_singletons, get_belief_store
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset singletons and force in-memory store for test isolation."""
    from backend.storage import InMemoryBeliefStore
    import backend.core.deps as deps
    reset_singletons()
    # Directly inject fresh in-memory store, bypassing settings.storage_backend
    deps._belief_store = InMemoryBeliefStore()
    yield
    reset_singletons()


@pytest.fixture
def client():
    return TestClient(app)


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ABES API"
        assert "version" in data


class TestBeliefsAPI:
    def test_list_empty(self, client):
        response = client.get("/beliefs")
        assert response.status_code == 200
        data = response.json()
        assert data["beliefs"] == []
        assert data["total"] == 0

    def test_create_belief(self, client):
        response = client.post("/beliefs", json={
            "content": "The sky is blue",
            "confidence": 0.9,
            "source": "test",
            "tags": ["fact"],
        })
        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "The sky is blue"
        assert data["confidence"] == 0.9
        assert "id" in data

    def test_get_belief(self, client):
        # create first
        create_resp = client.post("/beliefs", json={
            "content": "Water is wet",
            "confidence": 0.8,
        })
        belief_id = create_resp.json()["id"]

        # get it
        response = client.get(f"/beliefs/{belief_id}")
        assert response.status_code == 200
        assert response.json()["content"] == "Water is wet"

    def test_get_nonexistent(self, client):
        response = client.get(f"/beliefs/{uuid4()}")
        assert response.status_code == 404

    def test_update_belief(self, client):
        # create
        create_resp = client.post("/beliefs", json={
            "content": "Test belief",
            "confidence": 0.5,
        })
        belief_id = create_resp.json()["id"]

        # update
        response = client.patch(f"/beliefs/{belief_id}", json={
            "confidence": 0.9,
            "tags": ["updated"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == 0.9
        assert "updated" in data["tags"]

    def test_delete_belief(self, client):
        # create
        create_resp = client.post("/beliefs", json={
            "content": "To be deleted",
        })
        belief_id = create_resp.json()["id"]

        # delete
        response = client.delete(f"/beliefs/{belief_id}")
        assert response.status_code == 204

        # verify deprecated
        get_resp = client.get(f"/beliefs/{belief_id}")
        assert get_resp.json()["status"] == "deprecated"

    def test_reinforce_belief(self, client):
        # create
        create_resp = client.post("/beliefs", json={
            "content": "Reinforce me",
            "confidence": 0.5,
        })
        belief_id = create_resp.json()["id"]

        # reinforce
        response = client.post(f"/beliefs/{belief_id}/reinforce?boost=0.2")
        assert response.status_code == 200
        assert response.json()["confidence"] == 0.7

    def test_list_with_pagination(self, client):
        # create multiple beliefs
        for i in range(5):
            client.post("/beliefs", json={"content": f"Belief {i}"})

        response = client.get("/beliefs?page=1&page_size=2")
        data = response.json()
        assert len(data["beliefs"]) == 2
        assert data["total"] == 5

    def test_simulate_time(self, client):
        create_resp = client.post("/beliefs", json={
            "content": "Time shift test",
            "confidence": 0.8,
        })
        assert create_resp.status_code == 201
        belief_id = create_resp.json()["id"]

        before = client.get(f"/beliefs/{belief_id}")
        assert before.status_code == 200
        before_updated = before.json()["updated_at"]

        sim_resp = client.post("/beliefs/simulate-time?hours=24")
        assert sim_resp.status_code == 200
        payload = sim_resp.json()
        assert payload["adjusted"] >= 1
        assert payload["hours"] == 24

        after = client.get(f"/beliefs/{belief_id}")
        assert after.status_code == 200
        after_updated = after.json()["updated_at"]
        assert after_updated < before_updated


class TestAgentsAPI:
    def test_list_agents(self, client):
        response = client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 15  # default schedule (14 + consolidation)

    def test_get_agent(self, client):
        response = client.get("/agents/perception")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "perception"

    def test_get_unknown_agent(self, client):
        response = client.get("/agents/unknown")
        assert response.status_code == 404

    def test_get_schedule(self, client):
        response = client.get("/agents/schedule")
        assert response.status_code == 200
        schedule = response.json()
        assert "perception" in schedule


class TestBELAPI:
    def test_health(self, client):
        response = client.get("/bel/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_stats(self, client):
        response = client.get("/bel/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_beliefs" in data
        assert "cluster_count" in data


class TestClustersAPI:
    def test_list_clusters_empty(self, client):
        response = client.get("/clusters")
        assert response.status_code == 200
        data = response.json()
        assert data["clusters"] == []
        assert data["total"] == 0

    def test_cluster_stats(self, client):
        response = client.get("/clusters/stats")
        assert response.status_code == 200
        data = response.json()
        assert "cluster_count" in data

    def test_maintenance(self, client):
        response = client.post("/clusters/maintenance")
        assert response.status_code == 200


class TestSnapshotsAPI:
    def test_list_snapshots_empty(self, client):
        response = client.get("/snapshots")
        assert response.status_code == 200
        data = response.json()
        assert data["snapshots"] == []

    def test_get_nonexistent_snapshot(self, client):
        response = client.get(f"/snapshots/{uuid4()}")
        assert response.status_code == 404

    def test_latest_snapshot_none(self, client):
        response = client.get("/snapshots/latest")
        assert response.status_code == 200
        assert response.json() is None
