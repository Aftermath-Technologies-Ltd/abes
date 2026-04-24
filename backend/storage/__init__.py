"""
Storage layer exports: models, interfaces, and implementations.
"""

# Core models
from ..core.models.belief import Belief, BeliefStatus, OriginMetadata
from ..core.models.snapshot import (
    BeliefSnapshot,
    Snapshot,
    SnapshotDiff,
    SnapshotMetadata,
)

# Events
from ..core.events import (
    BaseEvent,
    BeliefCreatedEvent,
    BeliefDecayedEvent,
    BeliefReinforcedEvent,
    ContradictionDetectedEvent,
    EventLog,
    MutationProposedEvent,
)

# Snapshot compression
from ..core.bel.snapshot_compression import compress_snapshot, decompress_snapshot

# Abstract interfaces
from .base import BeliefStoreABC, SnapshotStoreABC

# Concrete implementations
from .in_memory import InMemoryBeliefStore, InMemorySnapshotStore
from .sqlite import SQLiteBeliefStore

# Query functions
from .snapshot_queries import (
    compare_snapshots,
    get_latest_snapshot,
    get_snapshot_by_iteration,
    get_snapshot_range,
    list_snapshots,
)

__all__ = [
    # models
    "Belief",
    "BeliefStatus",
    "OriginMetadata",
    "Snapshot",
    "SnapshotMetadata",
    "BeliefSnapshot",
    "SnapshotDiff",
    # events
    "BaseEvent",
    "BeliefCreatedEvent",
    "BeliefReinforcedEvent",
    "BeliefDecayedEvent",
    "ContradictionDetectedEvent",
    "MutationProposedEvent",
    "EventLog",
    # snapshot operations
    "compress_snapshot",
    "decompress_snapshot",
    "list_snapshots",
    "get_snapshot_by_iteration",
    "get_latest_snapshot",
    "get_snapshot_range",
    "compare_snapshots",
    # interfaces
    "BeliefStoreABC",
    "SnapshotStoreABC",
    # implementations
    "InMemoryBeliefStore",
    "InMemorySnapshotStore",
    "SQLiteBeliefStore",
]
