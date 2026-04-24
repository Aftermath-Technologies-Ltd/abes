# Author: Bradley R. Kinnard
"""Tests for belief stack selection and competition."""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.core.models.belief import Belief, BeliefLink, BeliefStatus, OriginMetadata
from backend.core.bel.stack import (
    select_belief_stack,
    compete_for_attention,
    _recency_score,
    _graph_spread_score,
)


def _make_belief(
    content: str = "Test belief",
    confidence: float = 0.7,
    salience: float = 0.5,
    status: BeliefStatus = BeliefStatus.Active,
    updated_at: datetime | None = None,
    links: list | None = None,
    **kwargs,
) -> Belief:
    now = datetime.now(timezone.utc)
    b = Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        status=status,
        origin=OriginMetadata(source="test"),
        updated_at=updated_at or now,
        **kwargs,
    )
    if links:
        b.links = links
    return b


class TestRecencyScore:
    def test_fresh_belief_scores_high(self):
        b = _make_belief(updated_at=datetime.now(timezone.utc))
        score = _recency_score(b)
        assert score > 0.9

    def test_old_belief_scores_low(self):
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        b = _make_belief(updated_at=old)
        score = _recency_score(b, max_age_hours=168)
        assert score == 0.0

    def test_week_old_belief_near_zero(self):
        week_ago = datetime.now(timezone.utc) - timedelta(hours=167)
        b = _make_belief(updated_at=week_ago)
        score = _recency_score(b)
        assert 0.0 < score < 0.05


class TestGraphSpreadScore:
    def test_no_links_zero_spread(self):
        b = _make_belief()
        assert _graph_spread_score(b, set()) == 0.0

    def test_links_to_active_beliefs(self):
        id1, id2 = uuid4(), uuid4()
        links = [
            BeliefLink(target_id=id1, relation="reinforces", weight=1.0),
            BeliefLink(target_id=id2, relation="contradicts", weight=0.5),
        ]
        b = _make_belief(links=links)
        score = _graph_spread_score(b, {id1, id2})
        assert abs(score - 0.2) < 0.01  # 2/10

    def test_links_to_missing_beliefs_ignored(self):
        missing_id = uuid4()
        links = [BeliefLink(target_id=missing_id, relation="reinforces", weight=1.0)]
        b = _make_belief(links=links)
        assert _graph_spread_score(b, set()) == 0.0

    def test_spread_capped_at_one(self):
        ids = [uuid4() for _ in range(15)]
        links = [BeliefLink(target_id=i, relation="reinforces", weight=1.0) for i in ids]
        b = _make_belief(links=links)
        assert _graph_spread_score(b, set(ids)) == 1.0


class TestSelectBeliefStack:
    def test_empty_beliefs(self):
        result = select_belief_stack([])
        assert result == []

    def test_filters_non_active_beliefs(self):
        active = _make_belief(content="active", status=BeliefStatus.Active, salience=0.9)
        deprecated = _make_belief(content="dead", status=BeliefStatus.Deprecated, salience=0.9)
        dormant = _make_belief(content="sleeping", status=BeliefStatus.Dormant, salience=0.0)
        result = select_belief_stack([active, deprecated, dormant])
        assert len(result) == 1
        assert result[0].content == "active"

    def test_includes_decaying_beliefs(self):
        decaying = _make_belief(content="fading", status=BeliefStatus.Decaying, salience=0.4)
        result = select_belief_stack([decaying])
        assert len(result) == 1

    def test_respects_stack_size(self):
        beliefs = [_make_belief(content=f"b{i}", salience=0.5 + i * 0.01) for i in range(20)]
        result = select_belief_stack(beliefs, stack_size=5)
        assert len(result) == 5

    def test_higher_salience_preferred(self):
        low = _make_belief(content="low sal", salience=0.1)
        high = _make_belief(content="high sal", salience=0.9)
        result = select_belief_stack([low, high], stack_size=1)
        assert result[0].content == "high sal"

    def test_relevance_scores_affect_ranking(self):
        b1 = _make_belief(content="irrelevant", salience=0.5)
        b2 = _make_belief(content="relevant", salience=0.5)
        relevance = {b1.id: 0.0, b2.id: 1.0}
        result = select_belief_stack([b1, b2], context_relevance=relevance, stack_size=1)
        assert result[0].content == "relevant"

    def test_required_ids_force_included(self):
        low = _make_belief(content="forced in", salience=0.01)
        high = _make_belief(content="high sal", salience=0.99)
        result = select_belief_stack(
            [low, high],
            stack_size=1,
            required_ids={low.id},
        )
        ids = {b.id for b in result}
        assert low.id in ids

    def test_custom_weights(self):
        b = _make_belief(salience=0.1)
        # only weight salience
        result = select_belief_stack(
            [b],
            weights={"salience": 1.0, "relevance": 0.0, "recency": 0.0, "graph_spread": 0.0},
        )
        assert len(result) == 1


class TestCompeteForAttention:
    def test_no_competition_needed(self):
        beliefs = [_make_belief(content=f"b{i}") for i in range(3)]
        winners, losers = compete_for_attention(beliefs, stack_size=10)
        assert len(winners) == 3
        assert losers == []

    def test_competition_selects_top(self):
        beliefs = []
        for i in range(10):
            beliefs.append(_make_belief(
                content=f"b{i}",
                confidence=0.1 * (i + 1),
                salience=0.1 * (i + 1),
            ))
        winners, losers = compete_for_attention(beliefs, stack_size=3)
        assert len(winners) == 3
        assert len(losers) == 7
        # winners should have highest salience*confidence
        min_winner_score = min(w.salience * w.confidence for w in winners)
        max_loser_score = max(l.salience * l.confidence for l in losers)
        assert min_winner_score >= max_loser_score

    def test_only_active_beliefs_compete(self):
        active = _make_belief(content="active", salience=0.5, status=BeliefStatus.Active)
        dormant = _make_belief(content="dormant", salience=0.5, status=BeliefStatus.Dormant)
        deprecated = _make_belief(content="deprecated", salience=0.5, status=BeliefStatus.Deprecated)
        winners, losers = compete_for_attention([active, dormant, deprecated], stack_size=10)
        assert len(winners) == 1
        assert winners[0].content == "active"

    def test_competition_deterministic(self):
        """Same input should produce same output."""
        beliefs = [_make_belief(content=f"b{i}", salience=0.1 * i, confidence=0.5) for i in range(8)]
        w1, l1 = compete_for_attention(beliefs, stack_size=3)
        w2, l2 = compete_for_attention(beliefs, stack_size=3)
        assert [b.id for b in w1] == [b.id for b in w2]
