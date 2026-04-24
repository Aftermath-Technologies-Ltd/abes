# Author: Bradley R. Kinnard
"""
Tests for RLBELIntegration (backend/core/bel/rl_integration.py).

Covers:
- Reset and step lifecycle
- Action application to agents
- Step result metrics
- Task success computation
- State tracking
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.agents.rl_policy import PolicyAction
from backend.core.bel.rl_integration import (
    RLBELIntegration,
    StepContext,
    StepResult,
)
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata
from backend.storage.in_memory import InMemoryBeliefStore


def utcnow():
    return datetime.now(timezone.utc)


def make_belief(
    content: str,
    confidence: float = 0.8,
    tension: float = 0.0,
    status: BeliefStatus = BeliefStatus.Active,
) -> Belief:
    """Helper to create test beliefs."""
    return Belief(
        content=content,
        confidence=confidence,
        tension=tension,
        status=status,
        origin=OriginMetadata(source="test"),
    )


@pytest.fixture
def belief_store():
    return InMemoryBeliefStore()


@pytest.fixture
def mock_agents():
    """Create mocked agents for testing."""
    decay = MagicMock()
    decay.process_beliefs = AsyncMock(return_value=([], []))
    decay.set_decay_rate = MagicMock()

    auditor = MagicMock()
    auditor.audit = AsyncMock(return_value=[])
    auditor._compute_tensions_from_cache = MagicMock(return_value={})

    mutation = MagicMock()
    mutation.process_beliefs = AsyncMock(return_value=[])

    resolution = MagicMock()
    resolution.process_pairs = AsyncMock(return_value=[])
    resolution._tension_threshold = 0.7

    consistency = MagicMock()
    consistency.get_probe = MagicMock(return_value=None)
    consistency.check_consistency = MagicMock()

    safety = MagicMock()
    safety.run_all_checks = AsyncMock(return_value=[])
    safety.is_mutation_vetoed = MagicMock(return_value=False)
    safety.is_deprecation_vetoed = MagicMock(return_value=False)

    return {
        "decay": decay,
        "auditor": auditor,
        "mutation": mutation,
        "resolution": resolution,
        "consistency": consistency,
        "safety": safety,
    }


@pytest.fixture
def integration(belief_store, mock_agents):
    return RLBELIntegration(
        belief_store=belief_store,
        decay_controller=mock_agents["decay"],
        contradiction_auditor=mock_agents["auditor"],
        mutation_engineer=mock_agents["mutation"],
        resolution_strategist=mock_agents["resolution"],
        consistency_checker=mock_agents["consistency"],
        safety_agent=mock_agents["safety"],
    )


class TestRLBELIntegrationInit:
    """Test initialization."""

    def test_init_with_store(self, belief_store):
        integration = RLBELIntegration(belief_store=belief_store)
        assert integration.store is belief_store
        assert integration._step_count == 0
        assert integration._beliefs == []

    def test_init_creates_default_agents(self, belief_store):
        integration = RLBELIntegration(belief_store=belief_store)
        assert integration._decay is not None
        assert integration._auditor is not None
        assert integration._mutation is not None


class TestReset:
    """Test reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_loads_active_beliefs(self, integration, belief_store):
        # add beliefs
        b1 = make_belief("Active 1")
        b2 = make_belief("Active 2")
        b3 = make_belief("Decaying", status=BeliefStatus.Decaying)

        await belief_store.create(b1)
        await belief_store.create(b2)
        await belief_store.create(b3)

        beliefs = await integration.reset()

        assert len(beliefs) == 2
        assert all(b.status == BeliefStatus.Active for b in beliefs)

    @pytest.mark.asyncio
    async def test_reset_clears_counters(self, integration, belief_store):
        integration._step_count = 10
        integration._mutations_total = 5
        integration._deprecations_total = 3

        await integration.reset()

        assert integration._step_count == 0
        assert integration._mutations_total == 0
        assert integration._deprecations_total == 0

    @pytest.mark.asyncio
    async def test_reset_empty_store(self, integration):
        beliefs = await integration.reset()
        assert beliefs == []


class TestStep:
    """Test step execution."""

    @pytest.mark.asyncio
    async def test_step_increments_counter(self, integration, belief_store):
        await belief_store.create(make_belief("Test"))
        await integration.reset()

        action = PolicyAction()
        await integration.step(action)

        assert integration._step_count == 1

    @pytest.mark.asyncio
    async def test_step_returns_result(self, integration, belief_store):
        await belief_store.create(make_belief("Test"))
        await integration.reset()

        action = PolicyAction()
        result = await integration.step(action)

        assert isinstance(result, StepResult)
        assert hasattr(result, "beliefs")
        assert hasattr(result, "task_success")
        assert hasattr(result, "rl_state_action")

    @pytest.mark.asyncio
    async def test_step_with_context(self, integration, belief_store):
        await belief_store.create(make_belief("Test"))
        await integration.reset()

        context = StepContext(
            context_text="What is the weather?",
            task_type="qa",
            episode_step=5,
        )
        action = PolicyAction()
        result = await integration.step(action, context)

        assert result.rl_state_action["step"] == 1

    @pytest.mark.asyncio
    async def test_step_calls_decay(self, integration, belief_store, mock_agents):
        await belief_store.create(make_belief("Test"))
        await integration.reset()

        action = PolicyAction()
        await integration.step(action)

        mock_agents["decay"].process_beliefs.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_calls_auditor(self, integration, belief_store, mock_agents):
        await belief_store.create(make_belief("Test"))
        await integration.reset()

        action = PolicyAction()
        await integration.step(action)

        mock_agents["auditor"].audit.assert_called_once()


class TestApplyAction:
    """Test action application to agents."""

    def test_applies_decay_rate(self, integration, mock_agents):
        action = PolicyAction(global_decay_rate=0.98)
        integration._apply_action(action)

        mock_agents["decay"].set_decay_rate.assert_called_with(0.98)

    def test_applies_mutation_threshold(self, integration, mock_agents):
        action = PolicyAction(mutation_threshold=0.55)
        integration._apply_action(action)

        assert integration._mutation._tension_threshold == 0.55

    def test_applies_resolution_threshold(self, integration, mock_agents):
        action = PolicyAction(resolution_threshold=0.65)
        integration._apply_action(action)

        assert integration._resolution._tension_threshold == 0.65


class TestTaskSuccessComputation:
    """Test task success heuristic."""

    def test_empty_beliefs_zero_success(self, integration):
        integration._beliefs = []
        success = integration._compute_task_success()
        assert success == 0.0

    def test_no_active_beliefs_zero_success(self, integration):
        integration._beliefs = [
            make_belief("Deprecated", status=BeliefStatus.Deprecated)
        ]
        success = integration._compute_task_success()
        assert success == 0.0

    def test_high_confidence_increases_success(self, integration):
        integration._beliefs = [
            make_belief("High conf 1", confidence=0.95, tension=0.1),
            make_belief("High conf 2", confidence=0.90, tension=0.1),
        ]
        success = integration._compute_task_success()
        # high confidence should contribute positively
        assert success > 0.3

    def test_high_tension_decreases_success(self, integration):
        integration._beliefs = [
            make_belief("High tension", confidence=0.8, tension=0.9),
        ]
        high_tension_success = integration._compute_task_success()

        integration._beliefs = [
            make_belief("Low tension", confidence=0.8, tension=0.1),
        ]
        low_tension_success = integration._compute_task_success()

        assert low_tension_success > high_tension_success


class TestStepResult:
    """Test StepResult structure."""

    def test_step_result_fields(self):
        result = StepResult(
            beliefs=[],
            task_success=0.75,
            consistency_score=0.9,
            contradiction_errors=2,
            core_beliefs_lost=0,
            mutations_applied=1,
            resolutions_applied=0,
            rl_state_action={"state": {}, "action": {}},
        )

        assert result.task_success == 0.75
        assert result.consistency_score == 0.9
        assert result.contradiction_errors == 2
        assert result.mutations_applied == 1


class TestStepContext:
    """Test StepContext structure."""

    def test_default_values(self):
        context = StepContext()
        assert context.context_text == ""
        assert context.task_type == ""
        assert context.episode_step == 0

    def test_custom_values(self):
        context = StepContext(
            context_text="Hello",
            task_type="chat",
            episode_step=10,
        )
        assert context.context_text == "Hello"
        assert context.task_type == "chat"
        assert context.episode_step == 10


class TestContractionPairExtraction:
    """Test contradiction pair extraction helper."""

    def test_extracts_high_tension_pairs(self, integration):
        b1 = make_belief("Belief 1")
        b2 = make_belief("Belief 2")
        integration._beliefs = [b1, b2]

        tension_map = {b1.id: 0.8, b2.id: 0.75}
        integration._resolution._tension_threshold = 0.7

        pairs = integration._extract_contradiction_pairs(tension_map)

        assert len(pairs) >= 1

    def test_no_pairs_below_threshold(self, integration):
        b1 = make_belief("Belief 1")
        b2 = make_belief("Belief 2")
        integration._beliefs = [b1, b2]

        tension_map = {b1.id: 0.3, b2.id: 0.4}
        integration._resolution._tension_threshold = 0.7

        pairs = integration._extract_contradiction_pairs(tension_map)

        assert pairs == []


class TestStateTracking:
    """Test state tracking methods."""

    @pytest.mark.asyncio
    async def test_get_current_beliefs(self, integration, belief_store):
        b1 = make_belief("Test 1")
        b2 = make_belief("Test 2")
        await belief_store.create(b1)
        await belief_store.create(b2)

        await integration.reset()
        beliefs = integration.get_current_beliefs()

        assert len(beliefs) == 2

    @pytest.mark.asyncio
    async def test_get_step_count(self, integration, belief_store):
        await belief_store.create(make_belief("Test"))
        await integration.reset()

        assert integration.get_step_count() == 0

        await integration.step(PolicyAction())
        assert integration.get_step_count() == 1

        await integration.step(PolicyAction())
        assert integration.get_step_count() == 2
