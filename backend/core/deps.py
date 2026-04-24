"""
Dependency injection for FastAPI and other components.
Provides singletons for stores and config.
"""

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field

from .config import ABESSettings, settings

# Lazy imports to avoid circular dependency
if TYPE_CHECKING:
    from ..storage import InMemoryBeliefStore, InMemorySnapshotStore, SQLiteBeliefStore
    from ..storage.base import BeliefStoreABC, SnapshotStoreABC

# singleton stores
_belief_store = None
_snapshot_store: Optional["InMemorySnapshotStore"] = None
_bel: Optional["BeliefEcologyLoop"] = None
_cluster_manager: Optional["BeliefClusterManager"] = None
_scheduler: Optional["AgentScheduler"] = None


@dataclass
class RLState:
    """Mutable singleton tracking RL training state for API exposure."""

    policy_trained: bool = False
    total_trajectories: int = 0
    last_training_epoch_loss: Optional[float] = None
    current_action_means: list[float] = field(default_factory=lambda: [0.0] * 7)
    training_interval: int = 10
    iterations_until_next_training: int = 10


_rl_state: Optional[RLState] = None


def get_belief_store():
    """Get the belief store singleton based on storage_backend setting."""
    global _belief_store
    if _belief_store is None:
        if settings.storage_backend == "sqlite":
            from ..storage import SQLiteBeliefStore
            # Extract path from database_url (sqlite+aiosqlite:///./data/abes.db -> ./data/abes.db)
            db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
            _belief_store = SQLiteBeliefStore(db_path=db_path)
        else:
            from ..storage import InMemoryBeliefStore
            _belief_store = InMemoryBeliefStore()
    return _belief_store


def get_snapshot_store():
    """Get the snapshot store singleton."""
    from ..storage import InMemorySnapshotStore

    global _snapshot_store
    if _snapshot_store is None:
        _snapshot_store = InMemorySnapshotStore()
    return _snapshot_store


def get_bel() -> "BeliefEcologyLoop":
    """Get the BEL singleton."""
    global _bel
    if _bel is None:
        from .bel.loop import BeliefEcologyLoop
        _bel = BeliefEcologyLoop(
            belief_store=get_belief_store(),
            snapshot_store=get_snapshot_store(),
        )
    return _bel


def get_cluster_manager() -> "BeliefClusterManager":
    """Get the cluster manager singleton."""
    global _cluster_manager
    if _cluster_manager is None:
        from .bel.clustering import BeliefClusterManager
        _cluster_manager = BeliefClusterManager()
    return _cluster_manager


def get_scheduler() -> "AgentScheduler":
    """Get the agent scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        from ..agents.scheduler import AgentScheduler
        _scheduler = AgentScheduler()
    return _scheduler


def get_settings() -> ABESSettings:
    """Get global settings."""
    return settings


def get_rl_state() -> RLState:
    """Get the RL training state singleton."""
    global _rl_state
    if _rl_state is None:
        _rl_state = RLState()
    return _rl_state


def reset_singletons() -> None:
    """Reset all singletons. For testing."""
    global _belief_store, _snapshot_store, _bel, _cluster_manager, _scheduler, _rl_state
    _belief_store = None
    _snapshot_store = None
    _bel = None
    _cluster_manager = None
    _scheduler = None
    _rl_state = None


__all__ = [
    "get_belief_store",
    "get_snapshot_store",
    "get_bel",
    "get_cluster_manager",
    "get_scheduler",
    "get_settings",
    "get_rl_state",
    "reset_singletons",
    "RLState",
]
