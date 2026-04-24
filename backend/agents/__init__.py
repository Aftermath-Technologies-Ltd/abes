# Author: Bradley R. Kinnard
"""ABES Multi-Agent Layer - 16 specialized agents per spec 4.1 (extended with Consolidation)."""

# Input & Creation agents
from .perception import PerceptionAgent
from .belief_creator import BeliefCreatorAgent

# Core loop agents
from .reinforcement import ReinforcementAgent
from .contradiction_auditor import ContradictionAuditorAgent
from .decay_controller import DecayControllerAgent
from .mutation_engineer import MutationEngineerAgent, MutationProposal
from .resolution_strategist import (
    ResolutionStrategistAgent,
    ResolutionResult,
    ResolutionStrategy,
)
from .relevance_curator import RelevanceCuratorAgent, RankedBelief
from .consolidation import ConsolidationAgent, ConsolidationEvent

# RL agents
from .rl_policy import RLPolicyAgent, EcologyState, PolicyAction
from .reward_shaper import RewardShaperAgent, RewardSignal, RewardComponents
from .experiment_orchestrator import (
    ExperimentOrchestratorAgent,
    Scenario,
    ScenarioStep,
    ExperimentRun,
    ExperimentStatus,
    SystemConfig,
)

# Support agents
from .baseline_memory_bridge import (
    BaselineMemoryBridgeAgent,
    RetrievalResult,
    ComparisonResult,
    RAGBackend,
    ChatHistoryBackend,
    BeliefEcologyBackend,
)
from .consistency_checker import (
    ConsistencyCheckerAgent,
    ConsistencyProbe,
    ConsistencyResult,
    ConsistencyMetrics,
)
from .safety_sanity import (
    SafetySanityAgent,
    SafetyViolation,
    SafetyMetrics,
    ViolationType,
    ActionType,
)
from .narrative_explainer import (
    NarrativeExplainerAgent,
    Explanation,
    ExplanationContext,
)


__all__ = [
    # Input & Creation (agents 1-2)
    "PerceptionAgent",
    "BeliefCreatorAgent",
    # Core loop (agents 3-8)
    "ReinforcementAgent",
    "ContradictionAuditorAgent",
    "DecayControllerAgent",
    "MutationEngineerAgent",
    "MutationProposal",
    "ResolutionStrategistAgent",
    "ResolutionResult",
    "ResolutionStrategy",
    "RelevanceCuratorAgent",
    "RankedBelief",
    # RL (agents 10-12)
    "RLPolicyAgent",
    "EcologyState",
    "PolicyAction",
    "RewardShaperAgent",
    "RewardSignal",
    "RewardComponents",
    "ExperimentOrchestratorAgent",
    "Scenario",
    "ScenarioStep",
    "ExperimentRun",
    "ExperimentStatus",
    "SystemConfig",
    # Support (agents 9, 13-15)
    "BaselineMemoryBridgeAgent",
    "RetrievalResult",
    "ComparisonResult",
    "RAGBackend",
    "ChatHistoryBackend",
    "BeliefEcologyBackend",
    "ConsistencyCheckerAgent",
    "ConsistencyProbe",
    "ConsistencyResult",
    "ConsistencyMetrics",
    "SafetySanityAgent",
    "SafetyViolation",
    "SafetyMetrics",
    "ViolationType",
    "ActionType",
    "NarrativeExplainerAgent",
    "Explanation",
    "ExplanationContext",
    # Consolidation (agent 16)
    "ConsolidationAgent",
    "ConsolidationEvent",
]
