"""
Belief Ecology Loop (BEL) implementation.
"""

from .clustering import BeliefClusterManager, Cluster, ClusteringConfig
from .contradiction import compute_tensions
from .decay import apply_decay
from .loop import BeliefEcologyLoop
from .ranking import rank_beliefs
from .relevance import compute_relevance_scores
from .stack import select_belief_stack, compete_for_attention
from .semantic_contradiction import (
    check_contradiction,
    ContradictionResult,
    Proposition,
    RuleBasedContradictionDetector,
)
from .nli_detector import (
    check_contradiction_nli,
    classify_nli,
    is_nli_available,
    NLIResult,
)
from .snapshot_compression import compress_snapshot, decompress_snapshot
from .snapshot_logger import log_snapshot

__all__ = [
    "BeliefEcologyLoop",
    "BeliefClusterManager",
    "Cluster",
    "ClusteringConfig",
    "apply_decay",
    "compute_tensions",
    "compute_relevance_scores",
    "rank_beliefs",
    "select_belief_stack",
    "compete_for_attention",
    "log_snapshot",
    "compress_snapshot",
    "decompress_snapshot",
    "check_contradiction",
    "ContradictionResult",
    "Proposition",
    "RuleBasedContradictionDetector",
    "check_contradiction_nli",
    "classify_nli",
    "is_nli_available",
    "NLIResult",
]
