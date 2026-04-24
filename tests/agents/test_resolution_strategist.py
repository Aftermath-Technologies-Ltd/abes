# Author: Bradley R. Kinnard
"""Tests for ResolutionStrategistAgent."""

import pytest
from uuid import uuid4

from backend.agents.resolution_strategist import (
    ResolutionStrategistAgent,
    ResolutionResult,
    ResolutionStrategy,
    _tokenize,
    _token_overlap_ratio,
    _has_context_marker,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.8,
    tension: float = 0.7,
    status: BeliefStatus = BeliefStatus.Active,
    source: str = "test",
    tags: list[str] = None,
) -> Belief:
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        status=status,
        origin=OriginMetadata(source=source),
        tags=tags or [],
    )


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("The quick brown fox")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_excludes_negations(self):
        tokens = _tokenize("I don't like coffee")
        assert "don't" not in tokens
        assert "not" not in tokens
        assert "like" in tokens
        assert "coffee" in tokens

    def test_excludes_short_words(self):
        tokens = _tokenize("I am a cat")
        assert "am" not in tokens
        assert "cat" in tokens

    def test_lowercases(self):
        tokens = _tokenize("COFFEE Tea")
        assert "coffee" in tokens
        assert "tea" in tokens


class TestTokenOverlap:
    def test_identical_texts(self):
        ratio = _token_overlap_ratio("coffee is good", "coffee is good")
        assert ratio == 1.0

    def test_no_overlap(self):
        ratio = _token_overlap_ratio("coffee is great", "python rocks hard")
        assert ratio == 0.0

    def test_partial_overlap(self):
        ratio = _token_overlap_ratio("coffee is good", "coffee is bad")
        # "coffee" overlaps, "good" and "bad" don't
        assert 0.3 < ratio < 0.7

    def test_empty_text(self):
        ratio = _token_overlap_ratio("", "coffee")
        assert ratio == 0.0


class TestContextMarker:
    def test_detects_before(self):
        assert _has_context_marker("before 2024")

    def test_detects_since(self):
        assert _has_context_marker("since last year")

    def test_detects_sometimes(self):
        assert _has_context_marker("sometimes I prefer tea")

    def test_detects_when(self):
        assert _has_context_marker("when it's cold")

    def test_no_marker(self):
        assert not _has_context_marker("coffee is great")


class TestShouldResolve:
    def test_triggers_high_score_high_confidence(self):
        agent = ResolutionStrategistAgent(tension_threshold=0.7, confidence_threshold=0.6)
        a = _make_belief(confidence=0.8)
        b = _make_belief(confidence=0.7)
        assert agent._should_resolve(a, b, 0.8)

    def test_no_trigger_low_score(self):
        agent = ResolutionStrategistAgent(tension_threshold=0.7, confidence_threshold=0.6)
        a = _make_belief(confidence=0.8)
        b = _make_belief(confidence=0.7)
        assert not agent._should_resolve(a, b, 0.5)

    def test_no_trigger_low_confidence(self):
        agent = ResolutionStrategistAgent(tension_threshold=0.7, confidence_threshold=0.6)
        a = _make_belief(confidence=0.8)
        b = _make_belief(confidence=0.4)  # below threshold
        assert not agent._should_resolve(a, b, 0.8)


class TestStrategySelection:
    def test_integrate_high_overlap(self):
        agent = ResolutionStrategistAgent(overlap_threshold_for_integrate=0.4)
        # same core words, different qualifiers
        a = _make_belief(content="Coffee is good for productivity and focus")
        b = _make_belief(content="Coffee is good for energy and focus")
        strategy = agent._select_strategy(a, b)
        assert strategy == ResolutionStrategy.Integrate

    def test_split_with_context_marker(self):
        agent = ResolutionStrategistAgent(overlap_threshold_for_integrate=0.9)
        a = _make_belief(content="Coffee is good")
        b = _make_belief(content="Sometimes tea is better")
        strategy = agent._select_strategy(a, b)
        assert strategy == ResolutionStrategy.Split

    def test_deprecate_loser_large_gap(self):
        agent = ResolutionStrategistAgent(
            overlap_threshold_for_integrate=0.9,
            confidence_gap_for_deprecate=0.3,
        )
        a = _make_belief(content="Coffee helps focus", confidence=0.9)
        b = _make_belief(content="Tea helps focus more", confidence=0.5)
        strategy = agent._select_strategy(a, b)
        assert strategy == ResolutionStrategy.DeprecateLoser

    def test_no_action_fallback(self):
        agent = ResolutionStrategistAgent(
            overlap_threshold_for_integrate=0.9,
            confidence_gap_for_deprecate=0.5,
        )
        a = _make_belief(content="Coffee is good", confidence=0.7)
        b = _make_belief(content="Tea is better", confidence=0.6)
        strategy = agent._select_strategy(a, b)
        assert strategy == ResolutionStrategy.NoAction


class TestIntegrate:
    def test_creates_merged_belief(self):
        agent = ResolutionStrategistAgent()
        a = _make_belief(content="Coffee is good for focus", confidence=0.8)
        b = _make_belief(content="Coffee is bad for sleep", confidence=0.7)
        merged = agent._integrate(a, b)

        assert merged.parent_id == a.id  # higher confidence is base
        assert "except" in merged.content.lower()
        assert merged.confidence == 0.8

    def test_combines_tags(self):
        agent = ResolutionStrategistAgent()
        a = _make_belief(content="Coffee is good", tags=["preference"])
        b = _make_belief(content="Coffee is bad sometimes", tags=["health"])
        merged = agent._integrate(a, b)

        assert "preference" in merged.tags
        assert "health" in merged.tags


class TestSplitConditions:
    def test_adds_context_tags(self):
        agent = ResolutionStrategistAgent()
        a = _make_belief(content="Coffee is good before noon")
        b = _make_belief(content="Tea is better after lunch")
        tags = agent._split_conditions(a, b)

        assert a.id in tags
        assert b.id in tags
        assert any("context:" in t for t in tags[a.id])
        assert any("context:" in t for t in tags[b.id])

    def test_default_tag_when_no_marker(self):
        agent = ResolutionStrategistAgent()
        a = _make_belief(content="Coffee is good")
        b = _make_belief(content="Sometimes tea is better")
        tags = agent._split_conditions(a, b)

        assert "context:default" in tags[a.id]
        assert any("context:sometimes" in t for t in tags[b.id])


class TestDeprecateLoser:
    def test_lower_confidence_loses(self):
        agent = ResolutionStrategistAgent()
        a = _make_belief(confidence=0.9)
        b = _make_belief(confidence=0.5)
        loser_id, winner_id = agent._deprecate_loser(a, b)

        assert loser_id == b.id
        assert winner_id == a.id

    def test_symmetric(self):
        agent = ResolutionStrategistAgent()
        a = _make_belief(confidence=0.5)
        b = _make_belief(confidence=0.9)
        loser_id, winner_id = agent._deprecate_loser(a, b)

        assert loser_id == a.id
        assert winner_id == b.id


class TestResolvePair:
    def test_returns_none_if_not_warranted(self):
        agent = ResolutionStrategistAgent(tension_threshold=0.7, confidence_threshold=0.6)
        a = _make_belief(confidence=0.3)
        b = _make_belief(confidence=0.3)
        result = agent.resolve_pair(a, b, 0.8)
        assert result is None

    def test_integrate_result(self):
        agent = ResolutionStrategistAgent(
            tension_threshold=0.7,
            confidence_threshold=0.6,
            overlap_threshold_for_integrate=0.5,
        )
        a = _make_belief(content="Coffee is good for you", confidence=0.8)
        b = _make_belief(content="Coffee is not good for sleep", confidence=0.7)
        result = agent.resolve_pair(a, b, 0.8)

        assert result.strategy == ResolutionStrategy.Integrate
        assert result.merged_belief is not None
        assert result.merged_belief.parent_id == a.id

    def test_split_result(self):
        agent = ResolutionStrategistAgent(
            tension_threshold=0.7,
            confidence_threshold=0.6,
            overlap_threshold_for_integrate=0.95,
        )
        a = _make_belief(content="Coffee is good", confidence=0.8)
        b = _make_belief(content="Sometimes tea is better", confidence=0.7)
        result = agent.resolve_pair(a, b, 0.8)

        assert result.strategy == ResolutionStrategy.Split
        assert result.tags_added is not None

    def test_deprecate_result(self):
        agent = ResolutionStrategistAgent(
            tension_threshold=0.7,
            confidence_threshold=0.5,
            overlap_threshold_for_integrate=0.95,
            confidence_gap_for_deprecate=0.3,
        )
        # very different content so no overlap, no context markers
        a = _make_belief(content="Coffee helps productivity", confidence=0.9)
        b = _make_belief(content="Tea improves relaxation", confidence=0.55)  # gap > 0.3
        result = agent.resolve_pair(a, b, 0.8)

        assert result is not None
        assert result.strategy == ResolutionStrategy.DeprecateLoser
        assert result.deprecated_id == b.id
        assert result.confidence_boost_id == a.id


class TestProcessPairs:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        agent = ResolutionStrategistAgent()
        result = await agent.process_pairs([], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_missing_beliefs(self):
        agent = ResolutionStrategistAgent()
        fake_id = uuid4()
        b = _make_belief(confidence=0.8)
        result = await agent.process_pairs([b], [(fake_id, b.id, 0.9)])
        assert result == []

    @pytest.mark.asyncio
    async def test_processes_multiple_pairs(self):
        agent = ResolutionStrategistAgent(
            tension_threshold=0.7,
            confidence_threshold=0.5,  # lowered so min confidence passes
            confidence_gap_for_deprecate=0.3,
        )
        a = _make_belief(content="X", confidence=0.9)
        b = _make_belief(content="Y", confidence=0.6)  # gap = 0.3
        c = _make_belief(content="Z", confidence=0.55)  # gap = 0.35

        pairs = [
            (a.id, b.id, 0.8),
            (a.id, c.id, 0.8),
        ]
        results = await agent.process_pairs([a, b, c], pairs)

        # both pairs should result in deprecate_loser (big confidence gap)
        assert len(results) == 2
        assert all(r.strategy == ResolutionStrategy.DeprecateLoser for r in results)

    @pytest.mark.asyncio
    async def test_filters_no_action(self):
        agent = ResolutionStrategistAgent(
            tension_threshold=0.7,
            confidence_threshold=0.6,
            overlap_threshold_for_integrate=0.99,
            confidence_gap_for_deprecate=0.9,
        )
        a = _make_belief(content="Coffee", confidence=0.7)
        b = _make_belief(content="Tea", confidence=0.7)

        # no overlap, no context marker, no gap -> NoAction
        pairs = [(a.id, b.id, 0.8)]
        results = await agent.process_pairs([a, b], pairs)

        # NoAction results are filtered out
        assert results == []
