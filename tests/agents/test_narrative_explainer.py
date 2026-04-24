# Author: Bradley R. Kinnard
"""Tests for NarrativeExplainerAgent."""

import pytest
from uuid import uuid4

from backend.agents.narrative_explainer import (
    NarrativeExplainerAgent,
    Explanation,
    ExplanationContext,
    _truncate,
    _confidence_descriptor,
    _tension_descriptor,
    _status_descriptor,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "test belief",
    confidence: float = 0.8,
    tension: float = 0.0,
    status: BeliefStatus = BeliefStatus.Active,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        status=status,
        origin=OriginMetadata(source="test"),
    )


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_exact_length(self):
        result = _truncate("hello", 5)
        assert result == "hello"


class TestConfidenceDescriptor:
    def test_high(self):
        assert _confidence_descriptor(0.9) == "high confidence"

    def test_moderate(self):
        assert _confidence_descriptor(0.6) == "moderate confidence"

    def test_low(self):
        assert _confidence_descriptor(0.35) == "low confidence"

    def test_very_low(self):
        assert _confidence_descriptor(0.1) == "very low confidence"


class TestTensionDescriptor:
    def test_high(self):
        assert _tension_descriptor(0.9) == "high tension"

    def test_moderate(self):
        assert _tension_descriptor(0.6) == "moderate tension"

    def test_some(self):
        assert _tension_descriptor(0.4) == "some tension"

    def test_minimal(self):
        assert _tension_descriptor(0.1) == "minimal tension"


class TestStatusDescriptor:
    def test_active(self):
        assert _status_descriptor(BeliefStatus.Active) == "active"

    def test_decaying(self):
        assert _status_descriptor(BeliefStatus.Decaying) == "fading"

    def test_deprecated(self):
        assert _status_descriptor(BeliefStatus.Deprecated) == "deprecated"

    def test_mutated(self):
        assert _status_descriptor(BeliefStatus.Mutated) == "evolved"


class TestExplainSelection:
    def test_empty_selection(self):
        agent = NarrativeExplainerAgent()
        ctx = ExplanationContext()

        exp = agent.explain_selection(ctx)

        assert exp.explanation_type == "selection"
        assert "No beliefs" in exp.summary
        assert len(exp.belief_references) == 0

    def test_with_selected_beliefs(self):
        agent = NarrativeExplainerAgent()
        beliefs = [
            _make_belief(content="Coffee is good", confidence=0.9),
            _make_belief(content="Tea is nice", confidence=0.7),
        ]
        ctx = ExplanationContext(
            selected_beliefs=beliefs,
            all_beliefs=beliefs,
        )

        exp = agent.explain_selection(ctx)

        assert "2" in exp.summary  # mentions count
        assert len(exp.belief_references) == 2
        assert beliefs[0].id in exp.belief_references

    def test_mentions_tension(self):
        agent = NarrativeExplainerAgent()
        b = _make_belief(content="Contested belief", tension=0.7)
        ctx = ExplanationContext(
            selected_beliefs=[b],
            all_beliefs=[b],
            tension_scores={b.id: 0.7},
        )

        exp = agent.explain_selection(ctx)

        assert any("tension" in d.lower() for d in exp.details)


class TestExplainMutation:
    def test_basic_mutation(self):
        agent = NarrativeExplainerAgent()
        original = _make_belief(content="Coffee is always good", confidence=0.4, tension=0.7)
        mutated = _make_belief(content="Coffee is usually good", confidence=0.5)

        exp = agent.explain_mutation(original, mutated)

        assert exp.explanation_type == "mutation"
        assert original.id in exp.belief_references
        assert mutated.id in exp.belief_references
        assert "evolved" in exp.summary.lower() or "mutated" in exp.summary.lower()

    def test_with_explicit_reason(self):
        agent = NarrativeExplainerAgent()
        original = _make_belief()
        mutated = _make_belief()

        exp = agent.explain_mutation(original, mutated, reason="Custom reason")

        assert any("Custom reason" in d for d in exp.details)


class TestExplainResolution:
    def test_integrate_strategy(self):
        agent = NarrativeExplainerAgent()
        a = _make_belief(content="X is true", confidence=0.8)
        b = _make_belief(content="X is false", confidence=0.7)
        outcome = _make_belief(content="X is sometimes true")

        exp = agent.explain_resolution(a, b, "integrate", outcome)

        assert exp.explanation_type == "resolution"
        assert "integrate" in exp.summary
        assert len(exp.belief_references) == 3

    def test_split_strategy(self):
        agent = NarrativeExplainerAgent()
        a = _make_belief(content="Before 2024")
        b = _make_belief(content="After 2024")

        exp = agent.explain_resolution(a, b, "split")

        assert "split" in exp.summary
        assert any("scope" in d.lower() for d in exp.details)

    def test_deprecate_loser_strategy(self):
        agent = NarrativeExplainerAgent()
        winner = _make_belief(content="Winner", confidence=0.9)
        loser = _make_belief(content="Loser", confidence=0.3)

        exp = agent.explain_resolution(winner, loser, "deprecate_loser")

        assert any("deprecated" in d.lower() for d in exp.details)


class TestExplainDeprecation:
    def test_low_confidence_reason(self):
        agent = NarrativeExplainerAgent()
        b = _make_belief(confidence=0.05)

        exp = agent.explain_deprecation(b)

        assert exp.explanation_type == "deprecation"
        assert any("confidence" in d.lower() or "threshold" in d.lower() for d in exp.details)

    def test_unused_reason(self):
        agent = NarrativeExplainerAgent()
        b = _make_belief(confidence=0.5)
        b.use_count = 0

        exp = agent.explain_deprecation(b)

        assert any("never used" in d.lower() or "stale" in d.lower() for d in exp.details)

    def test_custom_reason(self):
        agent = NarrativeExplainerAgent()
        b = _make_belief()

        exp = agent.explain_deprecation(b, reason="Superseded by newer belief")

        assert any("Superseded" in d for d in exp.details)


class TestExplainEcologyOverview:
    def test_empty_ecology(self):
        agent = NarrativeExplainerAgent()

        exp = agent.explain_ecology_overview([])

        assert exp.explanation_type == "overview"
        assert "0" in exp.summary

    def test_with_beliefs(self):
        agent = NarrativeExplainerAgent()
        beliefs = [
            _make_belief(status=BeliefStatus.Active),
            _make_belief(status=BeliefStatus.Active),
            _make_belief(status=BeliefStatus.Decaying),
        ]

        exp = agent.explain_ecology_overview(beliefs)

        assert "3" in exp.summary  # total
        assert "2 active" in exp.summary.lower()

    def test_highlights_high_tension(self):
        agent = NarrativeExplainerAgent()
        beliefs = [
            _make_belief(content="Normal", tension=0.1),
            _make_belief(content="Contested", tension=0.8),
        ]

        exp = agent.explain_ecology_overview(beliefs)

        assert any("high tension" in d.lower() or "tension:" in d.lower() for d in exp.details)


class TestExplanationHistory:
    def test_get_recent(self):
        agent = NarrativeExplainerAgent()
        ctx = ExplanationContext(
            selected_beliefs=[_make_belief()],
            all_beliefs=[_make_belief()],
        )

        for _ in range(5):
            agent.explain_selection(ctx)

        recent = agent.get_recent_explanations(limit=3)
        assert len(recent) == 3

    def test_get_by_type(self):
        agent = NarrativeExplainerAgent()

        # create mixed explanations
        agent.explain_ecology_overview([_make_belief()])
        agent.explain_deprecation(_make_belief())
        agent.explain_ecology_overview([_make_belief()])

        by_type = agent.get_explanations_by_type("overview")
        assert len(by_type) == 2

    def test_clear_history(self):
        agent = NarrativeExplainerAgent()
        agent.explain_ecology_overview([_make_belief()])

        agent.clear_history()

        assert len(agent.get_recent_explanations()) == 0


class TestExplanationDataclass:
    def test_has_timestamp(self):
        agent = NarrativeExplainerAgent()
        exp = agent.explain_ecology_overview([])

        assert exp.timestamp is not None
