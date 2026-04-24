# Author: Bradley R. Kinnard
"""
NarrativeExplainerAgent - generates human-readable explanations of belief ecology state.
Explains belief selection, conflict resolution, and ecology changes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus

logger = logging.getLogger(__name__)


@dataclass
class ExplanationContext:
    """Context for generating explanations."""

    query: Optional[str] = None
    selected_beliefs: list[Belief] = field(default_factory=list)
    all_beliefs: list[Belief] = field(default_factory=list)
    recent_mutations: list[tuple[UUID, UUID]] = field(default_factory=list)  # (old, new)
    recent_deprecations: list[UUID] = field(default_factory=list)
    resolutions: list[dict] = field(default_factory=list)
    tension_scores: dict[UUID, float] = field(default_factory=dict)


@dataclass
class Explanation:
    """A generated explanation."""

    summary: str
    details: list[str]
    belief_references: list[UUID]
    explanation_type: str  # "selection", "mutation", "resolution", "overview"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _confidence_descriptor(conf: float) -> str:
    """Human-readable confidence level."""
    if conf >= 0.8:
        return "high confidence"
    if conf >= 0.5:
        return "moderate confidence"
    if conf >= 0.3:
        return "low confidence"
    return "very low confidence"


def _tension_descriptor(tension: float) -> str:
    """Human-readable tension level."""
    if tension >= 0.8:
        return "high tension"
    if tension >= 0.5:
        return "moderate tension"
    if tension >= 0.3:
        return "some tension"
    return "minimal tension"


def _status_descriptor(status: BeliefStatus) -> str:
    """Human-readable status."""
    return {
        BeliefStatus.Active: "active",
        BeliefStatus.Decaying: "fading",
        BeliefStatus.Deprecated: "deprecated",
        BeliefStatus.Mutated: "evolved",
    }.get(status, str(status.value))


class NarrativeExplainerAgent:
    """
    Generates human-readable explanations (spec 4.1 agent #14).
    Provides text summaries for UI and logging.
    """

    def __init__(self, max_details: int = 10):
        self._max_details = max_details
        self._explanations: list[Explanation] = []

    def explain_selection(
        self, context: ExplanationContext
    ) -> Explanation:
        """
        Explain why certain beliefs were selected for a query.
        """
        if not context.selected_beliefs:
            exp = Explanation(
                summary="No beliefs were selected for this query.",
                details=["The belief ecology contains no relevant beliefs for the current context."],
                belief_references=[],
                explanation_type="selection",
            )
            self._explanations.append(exp)
            return exp

        details = []
        refs = []

        # summarize selection
        top = context.selected_beliefs[0]
        details.append(
            f"Selected {len(context.selected_beliefs)} belief(s) from "
            f"{len(context.all_beliefs)} total."
        )

        # explain top belief
        top_tension = context.tension_scores.get(top.id, top.tension)
        details.append(
            f"Top belief ({_confidence_descriptor(top.confidence)}): "
            f'"{_truncate(top.content)}"'
        )

        if top_tension > 0.3:
            details.append(
                f"Note: This belief has {_tension_descriptor(top_tension)} with other beliefs."
            )

        # mention others
        for belief in context.selected_beliefs[1 : self._max_details]:
            refs.append(belief.id)
            details.append(
                f"Also selected ({_confidence_descriptor(belief.confidence)}): "
                f'"{_truncate(belief.content)}"'
            )

        refs.insert(0, top.id)

        summary = (
            f"Selected {len(context.selected_beliefs)} relevant belief(s). "
            f"Top belief: \"{_truncate(top.content, 40)}\" "
            f"({_confidence_descriptor(top.confidence)})."
        )

        exp = Explanation(
            summary=summary,
            details=details[: self._max_details],
            belief_references=refs,
            explanation_type="selection",
        )
        self._explanations.append(exp)
        return exp

    def explain_mutation(
        self, original: Belief, mutated: Belief, reason: str = ""
    ) -> Explanation:
        """
        Explain why a belief was mutated.
        """
        details = [
            f'Original belief: "{_truncate(original.content)}"',
            f"Original confidence: {original.confidence:.2f}, tension: {original.tension:.2f}",
            f'Mutated to: "{_truncate(mutated.content)}"',
            f"New confidence: {mutated.confidence:.2f}",
        ]

        if reason:
            details.append(f"Reason: {reason}")
        else:
            if original.tension >= 0.6:
                details.append(
                    "Reason: High tension with other beliefs triggered a more nuanced formulation."
                )
            elif original.confidence < 0.5:
                details.append(
                    "Reason: Low confidence suggested hedging or qualification."
                )

        summary = (
            f"Belief evolved from \"{_truncate(original.content, 30)}\" to "
            f"\"{_truncate(mutated.content, 30)}\" due to conflict or uncertainty."
        )

        exp = Explanation(
            summary=summary,
            details=details,
            belief_references=[original.id, mutated.id],
            explanation_type="mutation",
        )
        self._explanations.append(exp)
        return exp

    def explain_resolution(
        self,
        belief_a: Belief,
        belief_b: Belief,
        strategy: str,
        outcome: Optional[Belief] = None,
    ) -> Explanation:
        """
        Explain how a conflict between beliefs was resolved.
        """
        details = [
            f'Belief A: "{_truncate(belief_a.content)}" (confidence: {belief_a.confidence:.2f})',
            f'Belief B: "{_truncate(belief_b.content)}" (confidence: {belief_b.confidence:.2f})',
            f"Resolution strategy: {strategy}",
        ]

        refs = [belief_a.id, belief_b.id]

        if strategy == "integrate":
            details.append("These beliefs were merged into a more nuanced statement.")
            if outcome:
                details.append(f'Result: "{_truncate(outcome.content)}"')
                refs.append(outcome.id)
        elif strategy == "split":
            details.append("Both beliefs were kept with explicit scope tags.")
        elif strategy == "deprecate_loser":
            loser = belief_a if belief_a.confidence < belief_b.confidence else belief_b
            details.append(f'The lower-confidence belief was deprecated: "{_truncate(loser.content)}"')
        else:
            details.append("The conflict was flagged for review.")

        summary = (
            f"Resolved conflict between two beliefs using {strategy} strategy."
        )

        exp = Explanation(
            summary=summary,
            details=details,
            belief_references=refs,
            explanation_type="resolution",
        )
        self._explanations.append(exp)
        return exp

    def explain_deprecation(
        self, belief: Belief, reason: str = ""
    ) -> Explanation:
        """
        Explain why a belief was deprecated.
        """
        details = [
            f'Deprecated belief: "{_truncate(belief.content)}"',
            f"Final confidence: {belief.confidence:.2f}",
            f"Status: {_status_descriptor(belief.status)}",
        ]

        if reason:
            details.append(f"Reason: {reason}")
        elif belief.confidence < 0.1:
            details.append("Reason: Confidence decayed below threshold.")
        elif belief.use_count == 0:
            details.append("Reason: Belief was never used and became stale.")
        else:
            details.append("Reason: Superseded by a more confident belief.")

        summary = f'Belief "{_truncate(belief.content, 40)}" was deprecated.'

        exp = Explanation(
            summary=summary,
            details=details,
            belief_references=[belief.id],
            explanation_type="deprecation",
        )
        self._explanations.append(exp)
        return exp

    def explain_ecology_overview(
        self, beliefs: list[Belief]
    ) -> Explanation:
        """
        Generate a high-level overview of the belief ecology.
        """
        total = len(beliefs)
        active = sum(1 for b in beliefs if b.status == BeliefStatus.Active)
        decaying = sum(1 for b in beliefs if b.status == BeliefStatus.Decaying)
        deprecated = sum(1 for b in beliefs if b.status == BeliefStatus.Deprecated)
        mutated = sum(1 for b in beliefs if b.status == BeliefStatus.Mutated)

        avg_conf = sum(b.confidence for b in beliefs) / total if total else 0.0
        avg_tension = sum(b.tension for b in beliefs) / total if total else 0.0

        high_tension = [b for b in beliefs if b.tension >= 0.6]

        details = [
            f"Total beliefs: {total}",
            f"Active: {active}, Decaying: {decaying}, Deprecated: {deprecated}, Mutated: {mutated}",
            f"Average confidence: {avg_conf:.2f}",
            f"Average tension: {avg_tension:.2f}",
        ]

        if high_tension:
            details.append(f"Beliefs with high tension: {len(high_tension)}")
            for b in high_tension[:3]:
                details.append(f'  - "{_truncate(b.content, 50)}" (tension: {b.tension:.2f})')

        summary = (
            f"Ecology contains {total} beliefs: {active} active, {decaying} decaying. "
            f"Avg confidence: {avg_conf:.2f}, avg tension: {avg_tension:.2f}."
        )

        exp = Explanation(
            summary=summary,
            details=details,
            belief_references=[b.id for b in high_tension[:5]],
            explanation_type="overview",
        )
        self._explanations.append(exp)
        return exp

    def get_recent_explanations(self, limit: int = 20) -> list[Explanation]:
        """Get most recent explanations."""
        return self._explanations[-limit:]

    def get_explanations_by_type(self, exp_type: str) -> list[Explanation]:
        """Filter explanations by type."""
        return [e for e in self._explanations if e.explanation_type == exp_type]

    def clear_history(self) -> None:
        """Clear explanation history."""
        self._explanations.clear()


__all__ = [
    "NarrativeExplainerAgent",
    "Explanation",
    "ExplanationContext",
]
