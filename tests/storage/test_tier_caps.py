# Author: Bradley R. Kinnard
"""Tests for memory tier cap enforcement and fitness-based rebalancing."""

import asyncio
from uuid import uuid4

import pytest

from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata
from backend.storage.in_memory import InMemoryBeliefStore, _belief_fitness


def _make_belief(
    content: str = "test belief",
    confidence: float = 0.8,
    salience: float = 0.8,
    use_count: int = 1,
    tier: str = "L1",
    is_axiom: bool = False,
) -> Belief:
    """Build a minimal Belief for testing."""
    return Belief(
        content=content,
        confidence=confidence,
        salience=salience,
        use_count=use_count,
        memory_tier=tier,
        is_axiom=is_axiom,
        origin=OriginMetadata(source="test"),
    )


class TestTierCapEnforcement:
    """Verify that adding more beliefs than L1 cap triggers automatic demotion."""

    def test_l1_overflow_demotes_lowest_fitness(self) -> None:
        """Adding 60 L1 beliefs with cap=50 should demote the 10 lowest-fitness."""
        store = InMemoryBeliefStore(tier_caps={"L1": 50, "L2": 200, "L3": 1000})

        async def run():
            # Add 60 beliefs directly; all start as L1
            for i in range(60):
                # First 50 have higher fitness (confidence=0.9), last 10 are lower (0.3)
                conf = 0.9 if i < 50 else 0.3
                b = _make_belief(
                    content=f"belief {i}",
                    confidence=conf,
                    salience=0.8,
                    use_count=1,
                    tier="L1",
                )
                store._beliefs[b.id] = b

            await store.rebalance_tiers()

            l1 = [b for b in store._beliefs.values() if b.memory_tier == "L1"]
            assert len(l1) == 50, f"Expected 50 in L1, got {len(l1)}"

            # The 10 low-confidence beliefs should have been demoted
            demoted = [
                b for b in store._beliefs.values()
                if b.memory_tier != "L1" and b.confidence == 0.3
            ]
            assert len(demoted) == 10, f"Expected 10 demoted, got {len(demoted)}"

        asyncio.get_event_loop().run_until_complete(run())


class TestFitnessBasedDemotion:
    """Verify that demotion is ordered by fitness (lowest demoted first)."""

    def test_lowest_fitness_demoted_first(self) -> None:
        """When L1 overflows, beliefs with lowest fitness are demoted."""
        store = InMemoryBeliefStore(tier_caps={"L1": 3, "L2": 200, "L3": 1000})

        async def run():
            # 5 beliefs; 3 high fitness and 2 low fitness
            high_ids = set()
            low_ids = set()
            for i in range(3):
                b = _make_belief(content=f"high {i}", confidence=0.9, salience=0.9, use_count=10, tier="L1")
                store._beliefs[b.id] = b
                high_ids.add(b.id)

            for i in range(2):
                b = _make_belief(content=f"low {i}", confidence=0.2, salience=0.2, use_count=1, tier="L1")
                store._beliefs[b.id] = b
                low_ids.add(b.id)

            await store.rebalance_tiers()

            # High-fitness beliefs remain in L1
            for bid in high_ids:
                assert store._beliefs[bid].memory_tier == "L1", \
                    f"High-fitness belief {bid} should stay in L1"

            # Low-fitness beliefs are demoted
            for bid in low_ids:
                assert store._beliefs[bid].memory_tier != "L1", \
                    f"Low-fitness belief {bid} should be demoted from L1"

        asyncio.get_event_loop().run_until_complete(run())


class TestL3OverflowDeprecation:
    """Verify that L3 overflow results in belief deprecation."""

    def test_l3_overflow_deprecates_lowest_fitness(self) -> None:
        """When L3 is at cap and receives more beliefs, lowest-fitness get deprecated."""
        # Caps: L1=1, L2=1, L3=3 → 8 beliefs means top 5 fill tiers, bottom 3 overflow and deprecate
        store = InMemoryBeliefStore(tier_caps={"L1": 1, "L2": 1, "L3": 3})

        async def run():
            # Add 8 active beliefs all assigned to L3 (overflow scenario)
            belief_ids = []
            for i in range(8):
                conf = 0.9 if i < 5 else 0.1  # first 5 high fitness, last 3 low
                b = _make_belief(
                    content=f"belief {i}",
                    confidence=conf,
                    salience=0.5,
                    use_count=1,
                    tier="L3",
                )
                store._beliefs[b.id] = b
                belief_ids.append((b.id, conf))

            await store.rebalance_tiers()

            deprecated = [
                b for b in store._beliefs.values()
                if b.status == BeliefStatus.Deprecated
            ]
            # 3 beliefs overflowed L3
            assert len(deprecated) == 3, f"Expected 3 deprecated, got {len(deprecated)}"

            # Deprecated beliefs should be the low-fitness ones (confidence=0.1)
            for b in deprecated:
                assert b.confidence == pytest.approx(0.1, abs=0.01), \
                    f"Expected low-fitness belief deprecated, got confidence={b.confidence}"

        asyncio.get_event_loop().run_until_complete(run())


class TestGetTierStats:
    """Verify get_tier_stats returns accurate counts and utilization percentages."""

    def test_tier_stats_accuracy(self) -> None:
        """get_tier_stats should return correct count, cap, and utilization_pct per tier."""
        store = InMemoryBeliefStore(tier_caps={"L1": 50, "L2": 200, "L3": 1000})

        async def run():
            # Add 10 L1, 20 L2, 5 L3 beliefs
            for i in range(10):
                b = _make_belief(content=f"l1 {i}", tier="L1")
                store._beliefs[b.id] = b
            for i in range(20):
                b = _make_belief(content=f"l2 {i}", tier="L2")
                store._beliefs[b.id] = b
            for i in range(5):
                b = _make_belief(content=f"l3 {i}", tier="L3")
                store._beliefs[b.id] = b

            stats = await store.get_tier_stats()

            assert stats["total"] == 35
            assert stats["tiers"]["L1"]["count"] == 10
            assert stats["tiers"]["L1"]["cap"] == 50
            assert stats["tiers"]["L1"]["utilization_pct"] == pytest.approx(20.0, abs=0.1)

            assert stats["tiers"]["L2"]["count"] == 20
            assert stats["tiers"]["L2"]["cap"] == 200
            assert stats["tiers"]["L2"]["utilization_pct"] == pytest.approx(10.0, abs=0.1)

            assert stats["tiers"]["L3"]["count"] == 5
            assert stats["tiers"]["L3"]["cap"] == 1000
            assert stats["tiers"]["L3"]["utilization_pct"] == pytest.approx(0.5, abs=0.1)

        asyncio.get_event_loop().run_until_complete(run())


class TestBeliefFitnessFormula:
    """Verify the fitness formula is correct."""

    def test_fitness_formula(self) -> None:
        """fitness = confidence * salience * max(use_count, 1)"""
        b = _make_belief(confidence=0.8, salience=0.5, use_count=4)
        assert _belief_fitness(b) == pytest.approx(0.8 * 0.5 * 4, abs=1e-6)

    def test_axiom_fitness_bonus(self) -> None:
        """Axioms get a large fitness bonus so they never lose L1 to non-axioms."""
        axiom = _make_belief(confidence=0.1, salience=0.1, use_count=1, is_axiom=True)
        regular = _make_belief(confidence=1.0, salience=1.0, use_count=100)
        assert _belief_fitness(axiom) > _belief_fitness(regular)
