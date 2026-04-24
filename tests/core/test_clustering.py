# Author: Bradley R. Kinnard
"""Tests for Belief Clustering."""

import pytest
import numpy as np
from uuid import uuid4

from backend.core.bel.clustering import (
    BeliefClusterManager,
    Cluster,
    ClusteringConfig,
)


def _random_embedding(dim: int = 384) -> np.ndarray:
    """Generate a random normalized embedding."""
    vec = np.random.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _similar_embedding(base: np.ndarray, noise: float = 0.1) -> np.ndarray:
    """Generate an embedding similar to base."""
    noisy = base + np.random.randn(*base.shape).astype(np.float32) * noise
    return noisy / np.linalg.norm(noisy)


class TestCluster:
    def test_default_values(self):
        cluster = Cluster()
        assert cluster.size == 0
        assert cluster.centroid.shape == (384,)

    def test_to_dict(self):
        cluster = Cluster()
        d = cluster.to_dict()
        assert "id" in d
        assert d["size"] == 0


class TestClusteringConfig:
    def test_default_values(self):
        cfg = ClusteringConfig()
        assert cfg.similarity_threshold == 0.7
        assert cfg.merge_threshold == 0.85
        assert cfg.max_cluster_size == 500


class TestBeliefClusterManager:
    def test_init(self):
        manager = BeliefClusterManager()
        assert len(manager.get_all_clusters()) == 0

    def test_assign_creates_cluster(self):
        manager = BeliefClusterManager()
        belief_id = uuid4()
        embedding = _random_embedding()

        cluster_id = manager.assign_cluster(belief_id, embedding)

        assert cluster_id is not None
        assert manager.get_cluster_id(belief_id) == cluster_id
        assert len(manager.get_all_clusters()) == 1

    def test_assign_joins_similar_cluster(self):
        manager = BeliefClusterManager()

        # first belief creates cluster
        b1 = uuid4()
        emb1 = np.ones(384, dtype=np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        c1 = manager.assign_cluster(b1, emb1)

        # second belief with same embedding should join same cluster
        b2 = uuid4()
        c2 = manager.assign_cluster(b2, emb1)  # identical embedding

        assert c1 == c2
        assert len(manager.get_all_clusters()) == 1
        assert manager.get_cluster(c1).size == 2

    def test_assign_creates_new_for_dissimilar(self):
        manager = BeliefClusterManager()

        b1 = uuid4()
        emb1 = _random_embedding()
        c1 = manager.assign_cluster(b1, emb1)

        # very different embedding
        b2 = uuid4()
        emb2 = -emb1  # opposite direction
        c2 = manager.assign_cluster(b2, emb2)

        assert c1 != c2
        assert len(manager.get_all_clusters()) == 2

    def test_remove_belief(self):
        manager = BeliefClusterManager()

        b1 = uuid4()
        emb1 = _random_embedding()
        c1 = manager.assign_cluster(b1, emb1)

        manager.remove_belief(b1)

        assert manager.get_cluster_id(b1) is None
        assert manager.get_cluster(c1).size == 0

    def test_get_cluster_members(self):
        manager = BeliefClusterManager()

        b1, b2 = uuid4(), uuid4()
        emb = np.ones(384, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        manager.assign_cluster(b1, emb)
        manager.assign_cluster(b2, emb)  # identical embedding, same cluster

        cluster_id = manager.get_cluster_id(b1)
        members = manager.get_cluster_members(cluster_id)

        assert b1 in members
        assert b2 in members

    def test_get_stats(self):
        manager = BeliefClusterManager()

        for _ in range(5):
            manager.assign_cluster(uuid4(), _random_embedding())

        stats = manager.get_stats()

        assert stats["total_beliefs"] == 5
        assert stats["cluster_count"] >= 1


class TestMaintenance:
    def test_removes_empty_clusters(self):
        manager = BeliefClusterManager()

        b1 = uuid4()
        c1 = manager.assign_cluster(b1, _random_embedding())
        manager.remove_belief(b1)

        stats = manager.run_maintenance(force=True)

        assert stats["removed"] == 1
        assert len(manager.get_all_clusters()) == 0

    def test_merges_similar_clusters(self):
        # manually create two separate clusters, then merge
        config = ClusteringConfig(merge_threshold=0.5)
        manager = BeliefClusterManager(config)

        # create base embedding
        emb = np.ones(384, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        # manually create two clusters with same centroid
        from backend.core.bel.clustering import Cluster
        c1 = Cluster(centroid=emb.copy(), member_ids=[uuid4()])
        c2 = Cluster(centroid=emb.copy(), member_ids=[uuid4()])
        manager._clusters[c1.id] = c1
        manager._clusters[c2.id] = c2
        for bid in c1.member_ids:
            manager._belief_to_cluster[bid] = c1.id
            manager._embeddings[bid] = emb.copy()
        for bid in c2.member_ids:
            manager._belief_to_cluster[bid] = c2.id
            manager._embeddings[bid] = emb.copy()

        assert len(manager.get_all_clusters()) == 2

        stats = manager.run_maintenance(force=True)

        assert stats["merged"] >= 1
        assert len(manager.get_all_clusters()) == 1

    def test_splits_large_clusters(self):
        config = ClusteringConfig(max_cluster_size=5)
        manager = BeliefClusterManager(config)

        # create cluster with 10 beliefs
        base_emb = _random_embedding()
        for _ in range(10):
            manager.assign_cluster(uuid4(), _similar_embedding(base_emb, 0.02))

        assert len(manager.get_all_clusters()) == 1

        stats = manager.run_maintenance(force=True)

        assert stats["split"] >= 1
        assert len(manager.get_all_clusters()) >= 2

    def test_maintenance_interval(self):
        config = ClusteringConfig(maintenance_interval=5)
        manager = BeliefClusterManager(config)

        # shouldn't run on iterations 1-4
        for _ in range(4):
            result = manager.run_maintenance()
            assert result.get("skipped") is True

        # should run on iteration 5
        result = manager.run_maintenance()
        assert "skipped" not in result


class TestCentroidUpdate:
    def test_centroid_updates_on_add(self):
        manager = BeliefClusterManager()

        # first embedding
        b1 = uuid4()
        emb1 = np.ones(384, dtype=np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        c1 = manager.assign_cluster(b1, emb1)

        # verify cluster has 1 member initially
        assert manager.get_cluster(c1).size == 1

        # add another belief with same embedding - should join and update centroid
        b2 = uuid4()
        manager.assign_cluster(b2, emb1)

        # verify cluster now has 2 members
        assert manager.get_cluster(c1).size == 2

        # centroid should still be valid (normalized mean)
        centroid = manager.get_cluster(c1).centroid
        assert np.linalg.norm(centroid) > 0.99  # still normalized


class TestKMeans:
    def test_kmeans_splits_data(self):
        manager = BeliefClusterManager()

        # two clearly distinct groups
        # group1: positive first half, negative second half
        group1 = np.zeros((10, 384), dtype=np.float32)
        group1[:, :192] = 1.0
        group1[:, 192:] = -1.0
        group1 = group1 / np.linalg.norm(group1, axis=1, keepdims=True)

        # group2: negative first half, positive second half
        group2 = np.zeros((10, 384), dtype=np.float32)
        group2[:, :192] = -1.0
        group2[:, 192:] = 1.0
        group2 = group2 / np.linalg.norm(group2, axis=1, keepdims=True)

        all_data = np.vstack([group1, group2])
        labels = manager._kmeans_2(all_data)

        # should split into two groups
        assert set(labels) == {0, 1}
        # first 10 should have same label
        assert len(set(labels[:10])) == 1
        # last 10 should have same label
        assert len(set(labels[10:])) == 1
        # groups should have different labels
        assert labels[0] != labels[10]
