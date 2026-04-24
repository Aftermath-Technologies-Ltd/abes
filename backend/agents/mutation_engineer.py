# Author: Bradley R. Kinnard
"""
MutationEngineerAgent - proposes mutated beliefs for high-tension, low-confidence entries.
Uses rule-based templates by default; LLM-based mutation is opt-in.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus, OriginMetadata

logger = logging.getLogger(__name__)


# temporal markers for condition-based mutation
_TEMPORAL_PATTERNS = [
    re.compile(r"\b(as of|since|until|before|after|in \d{4})\b", re.I),
    re.compile(r"\b(yesterday|today|tomorrow|last week|next month)\b", re.I),
    re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d", re.I),
]

# broad claim indicators for scope narrowing
_BROAD_CLAIM_PATTERNS = [
    re.compile(r"\b(always|never|all|every|none|no one|everyone|everything)\b", re.I),
]


@dataclass
class MutationProposal:
    """Output from mutation attempt."""

    original_id: UUID
    mutated_belief: Belief
    strategy: str  # "hedge", "condition", "scope_narrow", "source_attribute"
    timestamp: datetime


def _has_temporal_marker(text: str) -> bool:
    return any(pat.search(text) for pat in _TEMPORAL_PATTERNS)


def _has_broad_claim(text: str) -> bool:
    return any(pat.search(text) for pat in _BROAD_CLAIM_PATTERNS)


def _extract_date_hint(text: str) -> Optional[str]:
    """Pull a rough date from temporal patterns. Returns None if nothing obvious."""
    for pat in _TEMPORAL_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(0)
    return None


def _count_mutation_depth(belief: Belief, belief_map: dict[UUID, Belief]) -> int:
    """Walk parent chain to count mutation depth."""
    depth = 0
    current = belief
    seen: set[UUID] = set()

    while current.parent_id and current.parent_id not in seen:
        seen.add(current.id)
        depth += 1
        parent = belief_map.get(current.parent_id)
        if parent is None:
            break
        current = parent

    return depth


class MutationEngineerAgent:
    """
    Proposes mutations for beliefs meeting trigger conditions (spec 3.4.7).
    Uses rule-based templates (spec 3.4.9) unless LLM mode is configured.
    """

    def __init__(
        self,
        tension_threshold: float = settings.tension_threshold_mutation,
        confidence_threshold: float = settings.confidence_threshold_mutation,
        max_depth: int = settings.max_mutation_depth,
        strategy: str = settings.mutation_strategy,
        llm_provider=None,
    ):
        self._tension_threshold = tension_threshold
        self._confidence_threshold = confidence_threshold
        self._max_depth = max_depth
        self._strategy = strategy
        self._llm_provider = llm_provider

    def set_llm_provider(self, provider) -> None:
        """Configure LLM provider for mutation."""
        self._llm_provider = provider

    def _should_mutate(self, belief: Belief) -> bool:
        """Spec 3.4.7: tension >= threshold AND confidence < threshold."""
        return (
            belief.tension >= self._tension_threshold
            and belief.confidence < self._confidence_threshold
            and belief.status == BeliefStatus.Active
        )

    def _apply_hedge(self, content: str) -> str:
        """Wrap content in hedging language."""
        # avoid double-hedging
        if content.lower().startswith(("it may be", "perhaps", "possibly")):
            return content
        return f"It may be that {content[0].lower()}{content[1:]}"

    def _apply_condition(self, content: str, date_hint: str) -> str:
        """Add temporal condition to content. Avoids double-conditioning."""
        # avoid adding "at least as of" if already present
        if "at least as of" in content.lower():
            return content
        # strip trailing punctuation for clean append
        cleaned = content.rstrip(".,;")
        return f"{cleaned}, at least as of {date_hint}"

    def _apply_scope_narrow(self, content: str) -> str:
        """Narrow broad claims."""
        # replace absolutes with softer versions
        result = content
        replacements = [
            (r"\balways\b", "usually"),
            (r"\bnever\b", "rarely"),
            (r"\ball\b", "most"),
            (r"\bevery\b", "most"),
            (r"\bnone\b", "few"),
            (r"\bno one\b", "few people"),
            (r"\beveryone\b", "most people"),
            (r"\beverything\b", "most things"),
        ]
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.I)

        # if no replacement made, append qualifier
        if result == content:
            result = f"{content.rstrip('.,;')} in most cases"

        return result

    def _apply_source_attribute(self, content: str, source: str) -> str:
        """Attribute content to its source."""
        if content.lower().startswith("according to"):
            return content
        return f"According to {source}, {content[0].lower()}{content[1:]}"

    def _select_strategy(
        self, belief: Belief, contradicting: Optional[Belief] = None
    ) -> str:
        """Pick mutation strategy based on belief characteristics."""
        # priority order per spec 3.4.9

        # If belief already has temporal condition, don't apply condition again
        if "at least as of" in belief.content.lower():
            # Already has temporal qualification, use hedge or source
            if belief.confidence < 0.3:
                return "source_attribute"
            return "hedge"

        # 1. if contradicting belief has temporal marker, use condition
        # But not if it's an "as of" from our own mutation
        if contradicting:
            cont_content = contradicting.content.lower()
            has_real_temporal = _has_temporal_marker(contradicting.content)
            # Exclude our own mutation markers
            if "at least as of" in cont_content:
                has_real_temporal = False
            if has_real_temporal:
                return "condition"

        # 2. if belief has broad claim and contradiction exists, narrow scope
        if _has_broad_claim(belief.content) and contradicting:
            return "scope_narrow"

        # 3. if low confidence, attribute to source
        if belief.confidence < 0.3:
            return "source_attribute"

        # 4. default: hedge
        return "hedge"

    def _create_mutated_belief(
        self, original: Belief, new_content: str, strategy: str
    ) -> Belief:
        """Build mutated belief with proper lineage.

        Preserves relative confidence - hedging reduces confidence slightly
        but doesn't reset to 50%. A belief reinforced to 90% becomes 85%,
        while one at 80% becomes 75%.
        """
        now = datetime.now(timezone.utc)

        # Reduce confidence slightly due to uncertainty, but preserve relative strength
        # A 90% belief with contradiction becomes ~80%, an 80% becomes ~70%
        confidence_penalty = 0.1 + (original.tension * 0.1)  # higher tension = more penalty
        new_confidence = max(0.3, original.confidence - confidence_penalty)

        return Belief(
            content=new_content,
            confidence=new_confidence,
            origin=OriginMetadata(
                source=f"mutation:{strategy}",
                timestamp=now,
                last_reinforced=now,
                episode_id=original.origin.episode_id,
            ),
            tags=original.tags.copy(),  # inherit tags
            tension=0.0,  # fresh start
            cluster_id=original.cluster_id,
            status=BeliefStatus.Active,
            parent_id=original.id,  # lineage
        )

    def propose_mutation(
        self,
        belief: Belief,
        contradicting: Optional[Belief] = None,
        all_beliefs: Optional[list[Belief]] = None,
    ) -> Optional[MutationProposal]:
        """
        Propose a mutation for a single belief if it meets trigger conditions.
        Returns None if mutation not warranted or blocked by depth limit.
        """
        if not self._should_mutate(belief):
            return None

        # check mutation depth
        belief_map = {b.id: b for b in (all_beliefs or [])}
        belief_map[belief.id] = belief
        depth = _count_mutation_depth(belief, belief_map)

        if depth >= self._max_depth:
            logger.warning(
                f"belief {belief.id} at max mutation depth ({depth}), skipping"
            )
            return None

        strategy = self._select_strategy(belief, contradicting)
        content = belief.content

        if strategy == "hedge":
            new_content = self._apply_hedge(content)
        elif strategy == "condition":
            date_hint = _extract_date_hint(
                contradicting.content if contradicting else content
            )
            if not date_hint:
                date_hint = "recent information"
            new_content = self._apply_condition(content, date_hint)
        elif strategy == "scope_narrow":
            new_content = self._apply_scope_narrow(content)
        elif strategy == "source_attribute":
            new_content = self._apply_source_attribute(content, belief.origin.source)
        else:
            # fallback
            new_content = self._apply_hedge(content)
            strategy = "hedge"

        mutated = self._create_mutated_belief(belief, new_content, strategy)

        return MutationProposal(
            original_id=belief.id,
            mutated_belief=mutated,
            strategy=strategy,
            timestamp=datetime.now(timezone.utc),
        )

    async def propose_mutation_llm(
        self,
        belief: Belief,
        contradicting: Optional[Belief] = None,
        all_beliefs: Optional[list[Belief]] = None,
        context: Optional[str] = None,
    ) -> Optional[MutationProposal]:
        """
        Propose mutation using LLM. Falls back to rule-based if LLM unavailable.
        """
        if not self._should_mutate(belief):
            return None

        # check depth
        belief_map = {b.id: b for b in (all_beliefs or [])}
        belief_map[belief.id] = belief
        depth = _count_mutation_depth(belief, belief_map)

        if depth >= self._max_depth:
            logger.warning(f"belief {belief.id} at max mutation depth, skipping")
            return None

        # try LLM if available
        if self._llm_provider:
            result = await self._llm_provider.mutate(belief, contradicting, context)
            if result and result.mutated_content != belief.content:
                mutated = self._create_mutated_belief(
                    belief, result.mutated_content, result.strategy
                )
                return MutationProposal(
                    original_id=belief.id,
                    mutated_belief=mutated,
                    strategy=result.strategy,
                    timestamp=datetime.now(timezone.utc),
                )

        # fallback to rule-based
        return self.propose_mutation(belief, contradicting, all_beliefs)

    async def process_beliefs(
        self,
        beliefs: list[Belief],
        tension_map: Optional[dict[UUID, float]] = None,
        contradiction_pairs: Optional[list[tuple[UUID, UUID, float]]] = None,
    ) -> list[MutationProposal]:
        """
        Scan beliefs and propose mutations where warranted.

        Args:
            beliefs: Active beliefs to evaluate
            tension_map: Pre-computed tension scores per belief (optional, uses belief.tension if missing)
            contradiction_pairs: (belief_a, belief_b, score) tuples for finding contradicting beliefs

        Returns:
            List of mutation proposals
        """
        if not beliefs:
            return []

        # build lookup
        belief_map = {b.id: b for b in beliefs}

        # apply tension scores if provided
        if tension_map:
            for b in beliefs:
                if b.id in tension_map:
                    b.tension = tension_map[b.id]

        # build contradiction lookup for finding the contradicting belief
        contra_lookup: dict[UUID, UUID] = {}
        if contradiction_pairs:
            for a_id, b_id, score in contradiction_pairs:
                if score >= self._tension_threshold:
                    # store the "other" belief for each side
                    if a_id not in contra_lookup:
                        contra_lookup[a_id] = b_id
                    if b_id not in contra_lookup:
                        contra_lookup[b_id] = a_id

        proposals = []
        for belief in beliefs:
            contradicting = None
            if belief.id in contra_lookup:
                contradicting = belief_map.get(contra_lookup[belief.id])

            proposal = self.propose_mutation(belief, contradicting, beliefs)
            if proposal:
                proposals.append(proposal)

        if proposals:
            logger.info(f"proposed {len(proposals)} mutations")

        return proposals


__all__ = ["MutationEngineerAgent", "MutationProposal"]
