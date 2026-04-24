# Author: Bradley R. Kinnard
"""
ResolutionStrategistAgent - handles high-tension, high-confidence belief conflicts.
Applies reconciliation via integrate, split conditions, or deprecate loser strategies.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus, OriginMetadata

logger = logging.getLogger(__name__)


# negation words to exclude when computing token overlap
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "can't", "couldn't", "shouldn't", "wouldn't",
})

# temporal/contextual markers for split strategy
_CONTEXT_PATTERNS = [
    re.compile(r"\b(before|after|since|until|as of|in \d{4})\b", re.I),
    re.compile(r"\b(when|if|unless|during|while)\b", re.I),
    re.compile(r"\b(sometimes|occasionally|usually|often|rarely)\b", re.I),
]


class ResolutionStrategy(str, Enum):
    Integrate = "integrate"
    Split = "split"
    DeprecateLoser = "deprecate_loser"
    NoAction = "no_action"


@dataclass
class ResolutionResult:
    """Outcome of a resolution attempt."""

    strategy: ResolutionStrategy
    belief_a_id: UUID
    belief_b_id: UUID
    # what changed
    deprecated_id: Optional[UUID] = None
    merged_belief: Optional[Belief] = None
    tags_added: Optional[dict[UUID, list[str]]] = None
    confidence_boost_id: Optional[UUID] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


def _tokenize(text: str) -> set[str]:
    """Simple word tokenization, lowercased, excluding negations."""
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return {w for w in words if w not in _NEGATION_WORDS and len(w) > 2}


def _token_overlap_ratio(text1: str, text2: str) -> float:
    """Jaccard-like overlap of non-negation tokens."""
    t1 = _tokenize(text1)
    t2 = _tokenize(text2)
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union)


def _has_context_marker(text: str) -> bool:
    return any(pat.search(text) for pat in _CONTEXT_PATTERNS)


def _extract_context_hint(text: str) -> Optional[str]:
    """Pull a context marker for tag generation."""
    for pat in _CONTEXT_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(0).lower()
    return None


class ResolutionStrategistAgent:
    """
    Resolves high-tension, high-confidence belief pairs (spec 3.4.8, 3.4.10).
    Strategies: integrate, split conditions, deprecate loser.
    """

    def __init__(
        self,
        tension_threshold: float = settings.tension_threshold_resolution,
        confidence_threshold: float = settings.confidence_threshold_resolution,
        confidence_gap_for_deprecate: float = 0.3,
        overlap_threshold_for_integrate: float = 0.7,
    ):
        self._tension_threshold = tension_threshold
        self._confidence_threshold = confidence_threshold
        self._confidence_gap = confidence_gap_for_deprecate
        self._overlap_threshold = overlap_threshold_for_integrate

    def _should_resolve(
        self, belief_a: Belief, belief_b: Belief, contradiction_score: float
    ) -> bool:
        """
        Spec 3.4.8: contradiction_score >= threshold AND
        min(confidence_a, confidence_b) >= confidence_threshold.
        """
        if contradiction_score < self._tension_threshold:
            return False
        min_conf = min(belief_a.confidence, belief_b.confidence)
        return min_conf >= self._confidence_threshold

    def _select_strategy(
        self, belief_a: Belief, belief_b: Belief
    ) -> ResolutionStrategy:
        """
        Pick strategy per spec 3.4.10:
        1. Integrate if >70% token overlap
        2. Split if temporal/contextual markers present
        3. Deprecate loser if confidence gap > 0.3
        4. No action otherwise
        """
        overlap = _token_overlap_ratio(belief_a.content, belief_b.content)

        # 1. integrate
        if overlap >= self._overlap_threshold:
            return ResolutionStrategy.Integrate

        # 2. split
        if _has_context_marker(belief_a.content) or _has_context_marker(belief_b.content):
            return ResolutionStrategy.Split

        # 3. deprecate loser
        gap = abs(belief_a.confidence - belief_b.confidence)
        if gap > self._confidence_gap:
            return ResolutionStrategy.DeprecateLoser

        # 4. no clear strategy
        return ResolutionStrategy.NoAction

    def _integrate(self, belief_a: Belief, belief_b: Belief) -> Belief:
        """
        Merge into conditional belief. Simple heuristic: combine with "except" clause.
        """
        # order by confidence - higher confidence is the base
        if belief_a.confidence >= belief_b.confidence:
            base, modifier = belief_a, belief_b
        else:
            base, modifier = belief_b, belief_a

        # extract the distinguishing part from modifier
        base_tokens = _tokenize(base.content)
        mod_tokens = _tokenize(modifier.content)
        unique_mod = mod_tokens - base_tokens

        if unique_mod:
            # build conditional: "base except when [unique modifier context]"
            exception_hint = " ".join(sorted(unique_mod)[:3])
            new_content = f"{base.content.rstrip('.,;')}, except {exception_hint}"
        else:
            # fallback: just note it's conditional
            new_content = f"{base.content.rstrip('.,;')}, with exceptions"

        now = datetime.now(timezone.utc)
        merged = Belief(
            content=new_content,
            confidence=max(belief_a.confidence, belief_b.confidence),
            origin=OriginMetadata(
                source="resolution:integrate",
                timestamp=now,
                last_reinforced=now,
                episode_id=base.origin.episode_id,
            ),
            tags=list(set(belief_a.tags + belief_b.tags)),
            tension=0.0,
            cluster_id=base.cluster_id,
            status=BeliefStatus.Active,
            parent_id=base.id,
        )
        return merged

    def _split_conditions(
        self, belief_a: Belief, belief_b: Belief
    ) -> dict[UUID, list[str]]:
        """
        Add scope tags to both beliefs based on detected context markers.
        Returns mapping of belief_id -> tags to add.
        """
        tags_to_add: dict[UUID, list[str]] = {}

        hint_a = _extract_context_hint(belief_a.content)
        hint_b = _extract_context_hint(belief_b.content)

        # generate complementary tags
        if hint_a:
            tags_to_add[belief_a.id] = [f"context:{hint_a}"]
        else:
            tags_to_add[belief_a.id] = ["context:default"]

        if hint_b:
            tags_to_add[belief_b.id] = [f"context:{hint_b}"]
        else:
            tags_to_add[belief_b.id] = ["context:alternate"]

        return tags_to_add

    def _deprecate_loser(
        self, belief_a: Belief, belief_b: Belief
    ) -> tuple[UUID, UUID]:
        """
        Identify winner (higher confidence) and loser.
        Returns (loser_id, winner_id).
        """
        if belief_a.confidence >= belief_b.confidence:
            return belief_b.id, belief_a.id
        return belief_a.id, belief_b.id

    def resolve_pair(
        self, belief_a: Belief, belief_b: Belief, contradiction_score: float
    ) -> Optional[ResolutionResult]:
        """
        Attempt to resolve a contradicting pair.
        Returns None if resolution not warranted.
        """
        if not self._should_resolve(belief_a, belief_b, contradiction_score):
            return None

        strategy = self._select_strategy(belief_a, belief_b)

        if strategy == ResolutionStrategy.NoAction:
            return ResolutionResult(
                strategy=strategy,
                belief_a_id=belief_a.id,
                belief_b_id=belief_b.id,
            )

        if strategy == ResolutionStrategy.Integrate:
            merged = self._integrate(belief_a, belief_b)
            return ResolutionResult(
                strategy=strategy,
                belief_a_id=belief_a.id,
                belief_b_id=belief_b.id,
                merged_belief=merged,
                # both originals should be deprecated after merge
                deprecated_id=None,  # caller decides
            )

        if strategy == ResolutionStrategy.Split:
            tags = self._split_conditions(belief_a, belief_b)
            return ResolutionResult(
                strategy=strategy,
                belief_a_id=belief_a.id,
                belief_b_id=belief_b.id,
                tags_added=tags,
            )

        if strategy == ResolutionStrategy.DeprecateLoser:
            loser_id, winner_id = self._deprecate_loser(belief_a, belief_b)
            return ResolutionResult(
                strategy=strategy,
                belief_a_id=belief_a.id,
                belief_b_id=belief_b.id,
                deprecated_id=loser_id,
                confidence_boost_id=winner_id,
            )

        return None

    async def process_pairs(
        self,
        beliefs: list[Belief],
        contradiction_pairs: list[tuple[UUID, UUID, float]],
    ) -> list[ResolutionResult]:
        """
        Process all high-tension pairs and return resolution results.

        Args:
            beliefs: All beliefs (for lookup)
            contradiction_pairs: (belief_a_id, belief_b_id, score) tuples

        Returns:
            List of resolution results
        """
        if not beliefs or not contradiction_pairs:
            return []

        belief_map = {b.id: b for b in beliefs}
        results = []

        for a_id, b_id, score in contradiction_pairs:
            belief_a = belief_map.get(a_id)
            belief_b = belief_map.get(b_id)

            if belief_a is None or belief_b is None:
                continue

            result = self.resolve_pair(belief_a, belief_b, score)
            if result and result.strategy != ResolutionStrategy.NoAction:
                results.append(result)

        if results:
            strategies = [r.strategy.value for r in results]
            logger.info(f"resolved {len(results)} pairs: {strategies}")

        return results


__all__ = ["ResolutionStrategistAgent", "ResolutionResult", "ResolutionStrategy"]
