# Author: Bradley R. Kinnard
"""Tests for ConsistencyCheckerAgent."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from backend.agents.consistency_checker import (
    ConsistencyCheckerAgent,
    ConsistencyProbe,
    ConsistencyResult,
    ConsistencyMetrics,
    _hash_query,
    _text_similarity,
    _belief_overlap,
)
from backend.core.models.belief import Belief, OriginMetadata


def _make_belief(content: str = "test") -> Belief:
    return Belief(
        content=content,
        confidence=0.8,
        origin=OriginMetadata(source="test"),
    )


class TestHashQuery:
    def test_stable_hash(self):
        h1 = _hash_query("What is the weather?")
        h2 = _hash_query("What is the weather?")
        assert h1 == h2

    def test_normalizes_case(self):
        h1 = _hash_query("HELLO")
        h2 = _hash_query("hello")
        assert h1 == h2

    def test_strips_whitespace(self):
        h1 = _hash_query("  hello  ")
        h2 = _hash_query("hello")
        assert h1 == h2

    def test_different_queries_different_hash(self):
        h1 = _hash_query("hello")
        h2 = _hash_query("goodbye")
        assert h1 != h2


class TestTextSimilarity:
    def test_identical(self):
        sim = _text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_completely_different(self):
        sim = _text_similarity("hello", "goodbye")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = _text_similarity("hello world", "hello there")
        # overlap: {hello}, union: {hello, world, there}
        assert abs(sim - 1 / 3) < 0.01

    def test_empty_string(self):
        sim = _text_similarity("", "hello")
        assert sim == 0.0


class TestBeliefOverlap:
    def test_identical(self):
        ids = [uuid4(), uuid4()]
        overlap = _belief_overlap(ids, ids)
        assert overlap == 1.0

    def test_no_overlap(self):
        ids_a = [uuid4()]
        ids_b = [uuid4()]
        overlap = _belief_overlap(ids_a, ids_b)
        assert overlap == 0.0

    def test_partial_overlap(self):
        shared = uuid4()
        ids_a = [shared, uuid4()]
        ids_b = [shared, uuid4()]
        # intersection: 1, union: 3
        overlap = _belief_overlap(ids_a, ids_b)
        assert abs(overlap - 1 / 3) < 0.01

    def test_empty_list(self):
        overlap = _belief_overlap([], [uuid4()])
        assert overlap == 0.0


class TestRecordProbe:
    def test_records_new_probe(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.record_probe(
            query="What is the weather?",
            response="It's sunny",
            belief_ids=[uuid4()],
        )

        assert probe.query == "What is the weather?"
        assert probe.original_response == "It's sunny"
        assert len(probe.original_beliefs) == 1

    def test_dedup_same_query(self):
        agent = ConsistencyCheckerAgent()
        probe1 = agent.record_probe("hello", "response1", [])
        probe2 = agent.record_probe("hello", "response2", [])

        # should return the original probe
        assert probe1 is probe2
        assert len(agent._probes) == 1

    def test_max_probes_enforced(self):
        agent = ConsistencyCheckerAgent(max_probes=3)

        for i in range(5):
            agent.record_probe(f"query {i}", f"response {i}", [])

        assert len(agent._probes) == 3

    def test_tags_recorded(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.record_probe("q", "r", [], tags=["important", "test"])
        assert probe.tags == ["important", "test"]


class TestGetProbe:
    def test_retrieves_existing(self):
        agent = ConsistencyCheckerAgent()
        agent.record_probe("hello", "world", [])

        probe = agent.get_probe("hello")
        assert probe is not None
        assert probe.original_response == "world"

    def test_returns_none_for_missing(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.get_probe("nonexistent")
        assert probe is None


class TestListProbes:
    def test_list_all(self):
        agent = ConsistencyCheckerAgent()
        agent.record_probe("q1", "r1", [])
        agent.record_probe("q2", "r2", [])

        probes = agent.list_probes()
        assert len(probes) == 2

    def test_filter_by_tag(self):
        agent = ConsistencyCheckerAgent()
        agent.record_probe("q1", "r1", [], tags=["important"])
        agent.record_probe("q2", "r2", [], tags=["other"])

        probes = agent.list_probes(tag="important")
        assert len(probes) == 1
        assert probes[0].query == "q1"


class TestCheckConsistency:
    def test_consistent_response(self):
        agent = ConsistencyCheckerAgent(
            similarity_threshold=0.5,
            belief_overlap_threshold=0.5,
        )
        ids = [uuid4()]
        probe = agent.record_probe("query", "hello world", ids)

        result = agent.check_consistency(probe, "hello world", ids)

        assert result.is_consistent
        assert result.similarity_score == 1.0
        assert result.belief_overlap == 1.0

    def test_inconsistent_response(self):
        agent = ConsistencyCheckerAgent(
            similarity_threshold=0.8,
            belief_overlap_threshold=0.8,
        )
        ids = [uuid4()]
        probe = agent.record_probe("query", "hello world", ids)

        result = agent.check_consistency(probe, "goodbye universe", [uuid4()])

        assert not result.is_consistent
        assert result.similarity_score < 0.8
        assert result.belief_overlap < 0.8

    def test_result_logged(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.record_probe("q", "r", [])
        agent.check_consistency(probe, "r", [])

        assert len(agent._results) == 1


class TestRunChecks:
    @pytest.mark.asyncio
    async def test_runs_on_probes(self):
        agent = ConsistencyCheckerAgent(
            similarity_threshold=0.5,
            belief_overlap_threshold=0.0,
        )
        ids = [uuid4()]
        agent.record_probe("q1", "response one", ids)
        agent.record_probe("q2", "response two", ids)

        def gen(query, beliefs):
            return "response one"

        beliefs = [_make_belief()]
        results = await agent.run_checks(beliefs, gen)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_sample_size_limits(self):
        agent = ConsistencyCheckerAgent()
        for i in range(10):
            agent.record_probe(f"q{i}", f"r{i}", [])

        def gen(query, beliefs):
            return "response"

        results = await agent.run_checks([], gen, sample_size=3)
        assert len(results) == 3


class TestMetrics:
    def test_empty_results(self):
        metrics = ConsistencyMetrics.compute([])
        assert metrics.total_probes == 0
        assert metrics.consistency_rate == 0.0

    def test_computes_correctly(self):
        probe = ConsistencyProbe(
            query="q",
            query_hash="h",
            original_response="r",
            original_beliefs=[],
            timestamp=datetime.now(timezone.utc),
        )
        results = [
            ConsistencyResult(
                probe=probe,
                current_response="r",
                current_beliefs=[],
                similarity_score=1.0,
                belief_overlap=1.0,
                is_consistent=True,
            ),
            ConsistencyResult(
                probe=probe,
                current_response="x",
                current_beliefs=[],
                similarity_score=0.0,
                belief_overlap=0.0,
                is_consistent=False,
            ),
        ]

        metrics = ConsistencyMetrics.compute(results)
        assert metrics.total_probes == 2
        assert metrics.consistent_count == 1
        assert metrics.inconsistent_count == 1
        assert metrics.consistency_rate == 0.5
        assert metrics.avg_similarity == 0.5

    def test_agent_get_metrics(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.record_probe("q", "r", [])
        agent.check_consistency(probe, "r", [])

        metrics = agent.get_metrics()
        assert metrics.checks_performed == 1


class TestResultRetrieval:
    def test_get_recent_results(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.record_probe("q", "r", [])

        for i in range(10):
            agent.check_consistency(probe, f"r{i}", [])

        recent = agent.get_recent_results(limit=5)
        assert len(recent) == 5

    def test_get_inconsistencies(self):
        agent = ConsistencyCheckerAgent(similarity_threshold=0.99)
        ids = [uuid4()]
        probe = agent.record_probe("q", "exact response", ids)

        # one consistent
        agent.check_consistency(probe, "exact response", ids)
        # one inconsistent
        agent.check_consistency(probe, "different", [])

        inconsistent = agent.get_inconsistencies()
        assert len(inconsistent) == 1

    def test_clear_results(self):
        agent = ConsistencyCheckerAgent()
        probe = agent.record_probe("q", "r", [])
        agent.check_consistency(probe, "r", [])

        agent.clear_results()
        assert len(agent._results) == 0
