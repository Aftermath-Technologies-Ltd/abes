# Author: Bradley R. Kinnard
"""
RL-BEL Integration - connects RL policy to the Belief Ecology Loop.
Provides the step callback for the RL environment and populates rl_state_action.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ...agents.rl_policy import EcologyState, PolicyAction, RLPolicyAgent
from ...agents.consistency_checker import ConsistencyCheckerAgent
from ...agents.contradiction_auditor import ContradictionAuditorAgent
from ...agents.decay_controller import DecayControllerAgent
from ...agents.mutation_engineer import MutationEngineerAgent
from ...agents.resolution_strategist import ResolutionStrategistAgent
from ...agents.safety_sanity import SafetySanityAgent
from ...storage.base import BeliefStoreABC
from ..config import settings
from ..models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class StepContext:
    """Context passed to step function."""

    context_text: str = ""
    task_type: str = ""
    episode_step: int = 0


@dataclass
class StepResult:
    """Result from running one BEL step with RL action."""

    beliefs: list[Belief]
    task_success: float
    consistency_score: float
    contradiction_errors: int
    core_beliefs_lost: int
    mutations_applied: int
    resolutions_applied: int
    rl_state_action: dict


class RLBELIntegration:
    """
    Integrates RL policy actions with Belief Ecology Loop execution.
    Provides callbacks for the RL environment to drive actual BEL iterations.
    """

    def __init__(
        self,
        belief_store: BeliefStoreABC,
        decay_controller: Optional[DecayControllerAgent] = None,
        contradiction_auditor: Optional[ContradictionAuditorAgent] = None,
        mutation_engineer: Optional[MutationEngineerAgent] = None,
        resolution_strategist: Optional[ResolutionStrategistAgent] = None,
        consistency_checker: Optional[ConsistencyCheckerAgent] = None,
        safety_agent: Optional[SafetySanityAgent] = None,
    ):
        self.store = belief_store

        # agents (lazy init if not provided)
        self._decay = decay_controller or DecayControllerAgent()
        self._auditor = contradiction_auditor or ContradictionAuditorAgent()
        self._mutation = mutation_engineer or MutationEngineerAgent()
        self._resolution = resolution_strategist or ResolutionStrategistAgent()
        self._consistency = consistency_checker or ConsistencyCheckerAgent()
        self._safety = safety_agent or SafetySanityAgent()

        # state tracking
        self._beliefs: list[Belief] = []
        self._step_count = 0
        self._mutations_total = 0
        self._deprecations_total = 0
        self._reinforcements_total = 0

    async def reset(self) -> list[Belief]:
        """Reset for new episode. Returns initial beliefs."""
        self._step_count = 0
        self._mutations_total = 0
        self._deprecations_total = 0
        self._reinforcements_total = 0

        # load active beliefs
        self._beliefs = await self.store.list(status=BeliefStatus.Active, limit=1000)
        return self._beliefs

    async def step(
        self,
        action: PolicyAction,
        context: Optional[StepContext] = None,
    ) -> StepResult:
        """
        Execute one BEL step with RL-controlled parameters.
        Returns step result with metrics for reward computation.
        """
        context = context or StepContext()
        self._step_count += 1

        # capture pre-step state for state-action logging
        pre_state = EcologyState.from_beliefs(
            self._beliefs,
            episode_step=context.episode_step,
            task_type=context.task_type,
            recent_mutations=self._mutations_total,
            recent_deprecations=self._deprecations_total,
            recent_reinforcements=self._reinforcements_total,
        )

        # apply RL action to agents
        self._apply_action(action)

        # step 1: apply decay
        decay_events, modified = await self._decay.process_beliefs(self._beliefs)
        if modified:
            await self.store.bulk_update(modified)

        # count deprecations
        new_deprecations = sum(
            1 for e in decay_events if e.new_status == BeliefStatus.Deprecated
        )
        self._deprecations_total += new_deprecations

        # step 2: compute tensions
        contradiction_events = await self._auditor.audit(self._beliefs)
        contradiction_errors = len(contradiction_events)

        # update belief tensions from auditor
        tension_map = self._auditor._compute_tensions_from_cache(
            self._beliefs, settings.similarity_threshold_contradiction
        )
        for b in self._beliefs:
            b.tension = tension_map.get(b.id, 0.0)

        # step 3: mutations for high-tension, low-confidence
        mutations_applied = 0
        proposals = await self._mutation.process_beliefs(
            self._beliefs,
            tension_map=tension_map,
        )
        for proposal in proposals:
            if not self._safety.is_mutation_vetoed(proposal.mutated_belief.id):
                await self.store.create(proposal.mutated_belief)
                self._beliefs.append(proposal.mutated_belief)
                mutations_applied += 1
                self._mutations_total += 1

        # step 4: resolutions for high-tension, high-confidence
        resolutions_applied = 0
        contradiction_pairs = self._extract_contradiction_pairs(tension_map)
        results = await self._resolution.process_pairs(self._beliefs, contradiction_pairs)

        for result in results:
            if result.deprecated_id:
                # mark loser as deprecated
                for b in self._beliefs:
                    if b.id == result.deprecated_id:
                        if not self._safety.is_deprecation_vetoed(b.id):
                            b.status = BeliefStatus.Deprecated
                            await self.store.update(b)
                            resolutions_applied += 1
                            break

            if result.merged_belief:
                await self.store.create(result.merged_belief)
                self._beliefs.append(result.merged_belief)
                resolutions_applied += 1

        # step 5: check consistency
        consistency_score = 1.0
        if context.context_text:
            probe = self._consistency.get_probe(context.context_text)
            if probe:
                # check against current beliefs
                current_ids = [b.id for b in self._beliefs[:action.beliefs_to_surface]]
                result = self._consistency.check_consistency(
                    probe, context.context_text, current_ids
                )
                consistency_score = result.similarity_score

        # step 6: safety checks
        violations = await self._safety.run_all_checks(self._beliefs)
        core_beliefs_lost = sum(
            1 for v in violations if v.violation_type.value == "core_belief_forgotten"
        )

        # task success heuristic: based on belief quality
        task_success = self._compute_task_success()

        # refresh active beliefs list
        self._beliefs = [b for b in self._beliefs if b.status == BeliefStatus.Active]

        # build rl_state_action for snapshot
        rl_state_action = {
            "state": pre_state.__dict__,
            "action": {
                "global_decay_rate": action.global_decay_rate,
                "mutation_threshold": action.mutation_threshold,
                "resolution_threshold": action.resolution_threshold,
                "deprecation_threshold": action.deprecation_threshold,
                "ranking_weights": action.ranking_weights,
                "beliefs_to_surface": action.beliefs_to_surface,
            },
            "step": self._step_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return StepResult(
            beliefs=self._beliefs,
            task_success=task_success,
            consistency_score=consistency_score,
            contradiction_errors=contradiction_errors,
            core_beliefs_lost=core_beliefs_lost,
            mutations_applied=mutations_applied,
            resolutions_applied=resolutions_applied,
            rl_state_action=rl_state_action,
        )

    def _apply_action(self, action: PolicyAction) -> None:
        """Apply RL action to agent parameters."""
        # decay controller
        self._decay.set_decay_rate(action.global_decay_rate)
        self._decay._threshold_deprecated = action.deprecation_threshold

        # mutation engineer
        self._mutation._tension_threshold = action.mutation_threshold

        # resolution strategist
        self._resolution._tension_threshold = action.resolution_threshold

    def _extract_contradiction_pairs(
        self, tension_map: dict[UUID, float]
    ) -> list[tuple[UUID, UUID, float]]:
        """Extract belief pairs with high contradiction scores."""
        pairs = []
        threshold = self._resolution._tension_threshold

        # pairs are embedded in auditor's cache - extract from tension values
        for b in self._beliefs:
            if tension_map.get(b.id, 0.0) >= threshold:
                # find the belief this one contradicts most
                for other in self._beliefs:
                    if other.id != b.id:
                        other_tension = tension_map.get(other.id, 0.0)
                        if other_tension >= threshold:
                            # approximate score as average of their tensions
                            score = (tension_map[b.id] + other_tension) / 2
                            pairs.append((b.id, other.id, score))
                            break

        return pairs

    def _compute_task_success(self) -> float:
        """Heuristic task success based on belief ecology health."""
        if not self._beliefs:
            return 0.0

        active = [b for b in self._beliefs if b.status == BeliefStatus.Active]
        if not active:
            return 0.0

        # factors: avg confidence, low tension, reasonable count
        avg_conf = sum(b.confidence for b in active) / len(active)
        avg_tension = sum(b.tension for b in active) / len(active)

        # target ~500 beliefs
        count_score = 1.0 - abs(len(active) - 500) / 500
        count_score = max(0.0, count_score)

        # combine: high confidence good, high tension bad
        success = (
            0.4 * avg_conf
            + 0.3 * (1.0 - min(1.0, avg_tension))
            + 0.3 * count_score
        )

        return float(success)

    def get_current_beliefs(self) -> list[Belief]:
        """Get current belief list."""
        return self._beliefs

    def get_step_count(self) -> int:
        return self._step_count


__all__ = [
    "RLBELIntegration",
    "StepContext",
    "StepResult",
]
