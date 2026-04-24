"""
Core exports: models, config, and dependencies.
"""

# models
from .models.belief import Belief, BeliefStatus, OriginMetadata, EvidenceRef, BeliefLink
from .models.snapshot import Snapshot, SnapshotMetadata

# config and dependencies
from .config import ABESSettings, settings
from .deps import get_belief_store, get_settings, get_snapshot_store

__all__ = [
    # belief models
    "Belief",
    "BeliefStatus",
    "OriginMetadata",
    "EvidenceRef",
    "BeliefLink",
    # snapshot models
    "Snapshot",
    "SnapshotMetadata",
    # config
    "ABESSettings",
    "settings",
    # dependencies
    "get_belief_store",
    "get_snapshot_store",
    "get_settings",
]
