"""
Abstract base classes for belief and snapshot storage.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus
from ..core.models.snapshot import Snapshot


class SnapshotDiff(TypedDict):
    """What changed between two snapshots."""

    beliefs_added: List[UUID]
    beliefs_removed: List[UUID]
    beliefs_modified: List[UUID]
    tension_delta: float
    iteration_delta: int


class BeliefStoreABC(ABC):
    """Async storage for beliefs. Concrete implementations talk to the actual DB."""

    @abstractmethod
    async def create(self, belief: Belief) -> Belief: ...

    @abstractmethod
    async def get(self, belief_id: UUID) -> Optional[Belief]: ...

    @abstractmethod
    async def list(
        self,
        status: Optional[BeliefStatus] = None,
        cluster_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
        user_id: Optional[UUID] = None,
    ) -> List[Belief]:
        """
        List beliefs with filters. By convention, implementations sort by
        updated_at desc, but interface does not enforce this.

        user_id is the ceiling - NEVER return beliefs from other users.
        """
        ...

    @abstractmethod
    async def update(self, belief: Belief) -> Belief: ...

    @abstractmethod
    async def delete(self, belief_id: UUID) -> bool: ...

    @abstractmethod
    async def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        status: Optional[BeliefStatus] = None,
    ) -> List[Belief]:
        """
        Embedding-based search. Used when the ecology loop computes relevance
        against current context.
        """
        ...

    @abstractmethod
    async def bulk_update(self, beliefs: List[Belief]) -> int:
        # returns count of successfully updated beliefs
        ...


class SnapshotStoreABC(ABC):
    """Snapshot storage for time travel and replay functionality."""

    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> Snapshot: ...

    @abstractmethod
    async def get_snapshot(self, snapshot_id: UUID) -> Optional[Snapshot]: ...

    @abstractmethod
    async def list_snapshots(
        self,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Snapshot]: ...

    @abstractmethod
    async def compare_snapshots(
        self, snapshot_id_1: UUID, snapshot_id_2: UUID
    ) -> SnapshotDiff:
        """
        Diff two snapshots for analysis. Returns structured comparison
        of what changed between iterations.
        """
        ...


__all__ = ["BeliefStoreABC", "SnapshotStoreABC", "SnapshotDiff"]
