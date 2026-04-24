# Author: Bradley R. Kinnard
"""
Incremental Clustering for Belief Ecology (spec 3.6).
Assigns beliefs to semantic clusters and maintains cluster health.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """A semantic cluster of beliefs."""

    id: UUID = field(default_factory=uuid4)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(384, dtype=np.float32))
    member_ids: list[UUID] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "size": self.size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithm."""

    similarity_threshold: float = 0.7  # min similarity to join cluster
    merge_threshold: float = 0.85  # centroid similarity to merge clusters
    max_cluster_size: int = 500  # split if exceeded
    embedding_dim: int = 384
    maintenance_interval: int = 100  # run maintenance every N iterations


class BeliefClusterManager:
    """
    Manages incremental clustering of beliefs per spec 3.6.
    - Assigns new beliefs to nearest cluster or creates new one
    - Maintains cluster health (merge/split)
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self._clusters: dict[UUID, Cluster] = {}
        self._belief_to_cluster: dict[UUID, UUID] = {}
        self._embeddings: dict[UUID, np.ndarray] = {}
        self._iteration_count = 0

    def assign_cluster(
        self,
        belief_id: UUID,
        embedding: np.ndarray,
    ) -> Optional[UUID]:
        """
        Assign a belief to the nearest cluster or create a new one.
        Returns the assigned cluster ID.
        """
        embedding = self._normalize(embedding)
        self._embeddings[belief_id] = embedding

        # find nearest cluster
        best_cluster_id = None
        best_similarity = -1.0

        for cluster_id, cluster in self._clusters.items():
            sim = self._cosine_similarity(embedding, cluster.centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_cluster_id = cluster_id

        # join existing cluster if similarity >= threshold
        if best_similarity >= self.config.similarity_threshold and best_cluster_id:
            self._add_to_cluster(belief_id, best_cluster_id, embedding)
            return best_cluster_id

        # create new cluster with this belief as seed
        new_cluster = Cluster(
            centroid=embedding.copy(),
            member_ids=[belief_id],
        )
        self._clusters[new_cluster.id] = new_cluster
        self._belief_to_cluster[belief_id] = new_cluster.id
        logger.debug(f"created new cluster {new_cluster.id} for belief {belief_id}")
        return new_cluster.id

    def remove_belief(self, belief_id: UUID) -> None:
        """Remove a belief from its cluster."""
        cluster_id = self._belief_to_cluster.pop(belief_id, None)
        if cluster_id and cluster_id in self._clusters:
            cluster = self._clusters[cluster_id]
            if belief_id in cluster.member_ids:
                cluster.member_ids.remove(belief_id)
                self._update_centroid(cluster_id)

        self._embeddings.pop(belief_id, None)

    def get_cluster_id(self, belief_id: UUID) -> Optional[UUID]:
        """Get the cluster ID for a belief."""
        return self._belief_to_cluster.get(belief_id)

    def get_cluster(self, cluster_id: UUID) -> Optional[Cluster]:
        """Get a cluster by ID."""
        return self._clusters.get(cluster_id)

    def get_cluster_members(self, cluster_id: UUID) -> list[UUID]:
        """Get all belief IDs in a cluster."""
        cluster = self._clusters.get(cluster_id)
        return cluster.member_ids.copy() if cluster else []

    def get_all_clusters(self) -> list[Cluster]:
        """Get all clusters."""
        return list(self._clusters.values())

    def run_maintenance(self, force: bool = False) -> dict:
        """
        Run cluster maintenance per spec 3.6.2:
        - Merge clusters with centroid similarity >= 0.85
        - Split clusters exceeding max size
        - Remove empty clusters

        Returns maintenance stats.
        """
        self._iteration_count += 1

        if not force and self._iteration_count % self.config.maintenance_interval != 0:
            return {"skipped": True}

        stats = {"merged": 0, "split": 0, "removed": 0}

        # remove empty clusters
        empty_ids = [cid for cid, c in self._clusters.items() if c.size == 0]
        for cid in empty_ids:
            del self._clusters[cid]
            stats["removed"] += 1

        # merge similar clusters
        stats["merged"] = self._merge_similar_clusters()

        # split oversized clusters
        stats["split"] = self._split_large_clusters()

        if stats["merged"] or stats["split"] or stats["removed"]:
            logger.info(f"cluster maintenance: {stats}")

        return stats

    def _add_to_cluster(
        self,
        belief_id: UUID,
        cluster_id: UUID,
        embedding: np.ndarray,
    ) -> None:
        """Add belief to existing cluster and update centroid."""
        cluster = self._clusters[cluster_id]
        cluster.member_ids.append(belief_id)
        cluster.updated_at = datetime.now(timezone.utc)
        self._belief_to_cluster[belief_id] = cluster_id
        self._update_centroid(cluster_id)

    def _update_centroid(self, cluster_id: UUID) -> None:
        """Recompute cluster centroid as mean of member embeddings."""
        cluster = self._clusters.get(cluster_id)
        if not cluster or cluster.size == 0:
            return

        embeddings = []
        for bid in cluster.member_ids:
            emb = self._embeddings.get(bid)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            cluster.centroid = self._normalize(np.mean(embeddings, axis=0))

    def _merge_similar_clusters(self) -> int:
        """Merge clusters with centroid similarity >= merge_threshold."""
        merged_count = 0
        cluster_ids = list(self._clusters.keys())
        merged_into: set[UUID] = set()

        for i, cid_a in enumerate(cluster_ids):
            if cid_a in merged_into or cid_a not in self._clusters:
                continue

            for cid_b in cluster_ids[i + 1:]:
                if cid_b in merged_into or cid_b not in self._clusters:
                    continue

                cluster_a = self._clusters[cid_a]
                cluster_b = self._clusters[cid_b]

                sim = self._cosine_similarity(cluster_a.centroid, cluster_b.centroid)
                if sim >= self.config.merge_threshold:
                    # merge b into a
                    for bid in cluster_b.member_ids:
                        cluster_a.member_ids.append(bid)
                        self._belief_to_cluster[bid] = cid_a

                    del self._clusters[cid_b]
                    merged_into.add(cid_b)
                    self._update_centroid(cid_a)
                    merged_count += 1

        return merged_count

    def _split_large_clusters(self) -> int:
        """Split clusters exceeding max size using k-means with k=2."""
        split_count = 0
        oversized = [
            cid for cid, c in self._clusters.items()
            if c.size > self.config.max_cluster_size
        ]

        for cluster_id in oversized:
            cluster = self._clusters[cluster_id]
            embeddings = []
            belief_ids = []

            for bid in cluster.member_ids:
                emb = self._embeddings.get(bid)
                if emb is not None:
                    embeddings.append(emb)
                    belief_ids.append(bid)

            if len(embeddings) < 2:
                continue

            # simple k-means with k=2
            emb_array = np.array(embeddings)
            labels = self._kmeans_2(emb_array)

            # create new cluster for label=1
            new_cluster = Cluster()
            self._clusters[new_cluster.id] = new_cluster

            new_member_ids = []
            old_member_ids = []

            for bid, label in zip(belief_ids, labels):
                if label == 1:
                    new_member_ids.append(bid)
                    self._belief_to_cluster[bid] = new_cluster.id
                else:
                    old_member_ids.append(bid)

            cluster.member_ids = old_member_ids
            new_cluster.member_ids = new_member_ids

            self._update_centroid(cluster_id)
            self._update_centroid(new_cluster.id)
            split_count += 1

        return split_count

    def _kmeans_2(self, embeddings: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """Simple k-means with k=2."""
        n = len(embeddings)
        if n < 2:
            return np.zeros(n, dtype=int)

        # init: pick two random points
        idx = np.random.choice(n, 2, replace=False)
        centroids = embeddings[idx].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # assign
            for i, emb in enumerate(embeddings):
                d0 = np.linalg.norm(emb - centroids[0])
                d1 = np.linalg.norm(emb - centroids[1])
                labels[i] = 0 if d0 <= d1 else 1

            # update centroids
            for k in range(2):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = embeddings[mask].mean(axis=0)

        return labels

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b))

    def get_stats(self) -> dict:
        """Get clustering statistics."""
        sizes = [c.size for c in self._clusters.values()]
        return {
            "cluster_count": len(self._clusters),
            "total_beliefs": sum(sizes),
            "avg_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "min_cluster_size": min(sizes) if sizes else 0,
        }


__all__ = [
    "BeliefClusterManager",
    "Cluster",
    "ClusteringConfig",
]
