# Author: Bradley R. Kinnard
"""Tests for tension-based clustering and fitness-based population selection."""

from uuid import uuid4

import pytest

from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata
from backend.agents.mutation_engineer import cluster_high_tension_beliefs
from backend.agents.consolidation import compute_belief_fitness, population_selection


def _make_belief(
    content: str = "test belief",
    confidence: float = 0.8,
    salience: float = 0.5,
    use_count: int = 1,
    is_axiom: bool = False,
) -> Belief:
    """Build a minimal Belief for testing."""
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        use_count=use_count,
        is_axiom=is_axiom,
        origin=OriginMetadata(source="test"),
    )


class TestClusterHighTensionBeliefs:
    """cluster_high_tension_beliefs groups overlapping high-tension beliefs."""

    def test_empty_beliefs_returns_empty(self) -> None:
        """No beliefs in, no clusters out."""
        result = cluster_high_tension_beliefs([], {})
        assert result == []

    def test_low_tension_beliefs_excluded(self) -> None:
        """Beliefs below threshold produce no clusters."""
        b = _make_belief(content="cats are mammals")
        scores = {b.id: 0.1}
        result = cluster_high_tension_beliefs([b], scores, tension_threshold=0.3)
        assert result == []

    def test_high_tension_singletons_form_separate_clusters(self) -> None:
        """Beliefs with no token overlap stay in separate clusters."""
        b1 = _make_belief(content="cats are mammals")
        b2 = _make_belief(content="lightning strikes frequently")
        scores = {b1.id: 0.8, b2.id: 0.9}
        clusters = cluster_high_tension_beliefs([b1, b2], scores, tension_threshold=0.3)
        assert len(clusters) == 2
        # each cluster has exactly one belief
        assert all(len(c) == 1 for c in clusters)

    def test_overlapping_beliefs_merged_into_one_cluster(self) -> None:
        """Beliefs sharing significant token overlap land in the same cluster."""
        b1 = _make_belief(content="the sky is blue during the day")
        b2 = _make_belief(content="the sky is not blue it is grey during the day")
        b3 = _make_belief(content="the sky is red at sunset during the day")
        scores = {b1.id: 0.7, b2.id: 0.8, b3.id: 0.6}
        clusters = cluster_high_tension_beliefs(
            [b1, b2, b3], scores, tension_threshold=0.3, overlap_threshold=0.2
        )
        # all three share "the sky is ... during the day" tokens so should cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_every_high_tension_belief_appears_exactly_once(self) -> None:
        """No belief is duplicated or omitted across clusters."""
        beliefs = [_make_belief(content=f"topic alpha beta {i}") for i in range(6)]
        scores = {b.id: 0.5 for b in beliefs}
        clusters = cluster_high_tension_beliefs(beliefs, scores, tension_threshold=0.3)
        all_ids = [b.id for cluster in clusters for b in cluster]
        assert len(all_ids) == len(set(all_ids)) == 6


class TestComputeBeliefFitness:
    """compute_belief_fitness applies the expected formula."""

    def test_base_formula(self) -> None:
        b = _make_belief(confidence=0.6, salience=0.5, use_count=3)
        expected = 0.6 * 0.5 * 3
        assert compute_belief_fitness(b) == pytest.approx(expected)

    def test_use_count_zero_treated_as_one(self) -> None:
        b = _make_belief(confidence=0.8, salience=0.5, use_count=0)
        expected = 0.8 * 0.5 * 1
        assert compute_belief_fitness(b) == pytest.approx(expected)

    def test_axiom_bonus_dominates_non_axiom(self) -> None:
        axiom = _make_belief(confidence=0.01, salience=0.01, use_count=1, is_axiom=True)
        strong = _make_belief(confidence=1.0, salience=1.0, use_count=1000)
        assert compute_belief_fitness(axiom) > compute_belief_fitness(strong)


class TestPopulationSelection:
    """population_selection returns the top-N by fitness."""

    def test_returns_target_count(self) -> None:
        beliefs = [_make_belief(confidence=round(i * 0.1, 1)) for i in range(1, 8)]
        selected = population_selection(beliefs, 3)
        assert len(selected) == 3

    def test_highest_fitness_selected(self) -> None:
        low = _make_belief(content="low", confidence=0.1, salience=0.1, use_count=1)
        high = _make_belief(content="high", confidence=0.9, salience=0.9, use_count=5)
        mid = _make_belief(content="mid", confidence=0.5, salience=0.5, use_count=2)
        selected = population_selection([low, mid, high], 2)
        ids = {b.id for b in selected}
        assert high.id in ids
        assert mid.id in ids
        assert low.id not in ids

    def test_target_exceeds_pool_returns_all_sorted(self) -> None:
        beliefs = [_make_belief(confidence=0.3), _make_belief(confidence=0.7)]
        selected = population_selection(beliefs, 100)
        assert len(selected) == 2
        assert selected[0].confidence == pytest.approx(0.7)

    def test_zero_target_returns_empty(self) -> None:
        beliefs = [_make_belief()]
        assert population_selection(beliefs, 0) == []

    def test_axioms_rank_first(self) -> None:
        regular = _make_belief(confidence=1.0, salience=1.0, use_count=100)
        axiom = _make_belief(confidence=0.1, salience=0.1, use_count=1, is_axiom=True)
        selected = population_selection([regular, axiom], 1)
        assert selected[0].id == axiom.id
