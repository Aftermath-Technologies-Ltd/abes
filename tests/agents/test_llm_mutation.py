# Author: Bradley R. Kinnard
"""Tests for LLM mutation provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.llm_mutation import (
    LLMMutationResult,
    MockLLMProvider,
    OpenAILLMProvider,
    get_llm_provider,
    set_llm_provider,
    configure_openai_provider,
)
from backend.agents.mutation_engineer import MutationEngineerAgent
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


@pytest.fixture
def sample_belief():
    return Belief(
        content="All cats are black",
        confidence=0.4,
        origin=OriginMetadata(source="test"),
        tension=0.7,
        status=BeliefStatus.Active,
    )


@pytest.fixture
def contradicting_belief():
    return Belief(
        content="Some cats are orange",
        confidence=0.6,
        origin=OriginMetadata(source="test"),
        tension=0.5,
        status=BeliefStatus.Active,
    )


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_mutate(self, sample_belief):
        provider = MockLLMProvider()
        result = await provider.mutate(sample_belief)

        assert result is not None
        assert result.mutated_content.startswith("It appears that")
        assert result.strategy == "llm_hedge"
        assert result.model == "mock"

    @pytest.mark.asyncio
    async def test_mutate_with_contradiction(self, sample_belief, contradicting_belief):
        provider = MockLLMProvider()
        result = await provider.mutate(sample_belief, contradicting_belief)

        assert result is not None
        assert result.mutated_content != sample_belief.content


class TestOpenAIProvider:
    def test_init_no_key(self):
        # should not raise — clear env to avoid picking up real key
        import os
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            provider = OpenAILLMProvider(api_key=None)
            assert provider._api_key is None
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

    def test_init_with_key(self):
        provider = OpenAILLMProvider(api_key="test-key")
        assert provider._api_key == "test-key"

    def test_build_prompt(self, sample_belief):
        provider = OpenAILLMProvider(api_key="test")
        prompt = provider._build_user_prompt(sample_belief)

        assert "All cats are black" in prompt
        assert "confidence:" in prompt.lower()

    def test_build_prompt_with_contradiction(self, sample_belief, contradicting_belief):
        provider = OpenAILLMProvider(api_key="test")
        prompt = provider._build_user_prompt(sample_belief, contradicting_belief)

        assert "Some cats are orange" in prompt
        assert "Contradicting belief" in prompt

    def test_parse_response(self):
        provider = OpenAILLMProvider(api_key="test")
        response = """STRATEGY: hedge
MUTATED: "It may be that most cats are black"
REASONING: Added epistemic hedge to reduce absoluteness"""

        result = provider._parse_response(response, "All cats are black")

        assert result.strategy == "llm_hedge"
        assert "may be" in result.mutated_content
        assert result.reasoning is not None

    def test_parse_malformed_response(self):
        provider = OpenAILLMProvider(api_key="test")
        response = "Just some random text without format"

        result = provider._parse_response(response, "Original content")

        assert result.mutated_content == "Original content"
        assert result.strategy == "llm_unknown"

    @pytest.mark.asyncio
    async def test_mutate_no_key(self, sample_belief):
        provider = OpenAILLMProvider(api_key=None)
        result = await provider.mutate(sample_belief)

        assert result is None


class TestMutationEngineerWithLLM:
    @pytest.mark.asyncio
    async def test_with_mock_provider(self, sample_belief):
        provider = MockLLMProvider()
        agent = MutationEngineerAgent(llm_provider=provider)

        proposal = await agent.propose_mutation_llm(sample_belief)

        assert proposal is not None
        assert "llm" in proposal.strategy

    @pytest.mark.asyncio
    async def test_set_provider(self, sample_belief):
        provider = MockLLMProvider()
        agent = MutationEngineerAgent()

        # initially no LLM
        assert agent._llm_provider is None

        agent.set_llm_provider(provider)

        proposal = await agent.propose_mutation_llm(sample_belief)
        assert proposal is not None

    @pytest.mark.asyncio
    async def test_fallback_to_rule(self, sample_belief):
        # no LLM provider - should fallback to rule-based
        agent = MutationEngineerAgent()
        proposal = await agent.propose_mutation_llm(sample_belief)

        assert proposal is not None
        # rule-based strategies don't have "llm" prefix
        assert "llm" not in proposal.strategy

    @pytest.mark.asyncio
    async def test_respects_depth_limit(self):
        # create deep chain
        beliefs = []
        parent_id = None
        for i in range(6):
            b = Belief(
                content=f"Belief {i}",
                confidence=0.4,
                origin=OriginMetadata(source="test"),
                tension=0.7,
                status=BeliefStatus.Active,
                parent_id=parent_id,
            )
            beliefs.append(b)
            parent_id = b.id

        provider = MockLLMProvider()
        agent = MutationEngineerAgent(llm_provider=provider, max_depth=5)

        # last belief should be blocked
        proposal = await agent.propose_mutation_llm(beliefs[-1], all_beliefs=beliefs)
        assert proposal is None


class TestProviderSingleton:
    def test_get_set(self):
        set_llm_provider(None)
        assert get_llm_provider() is None

        provider = MockLLMProvider()
        set_llm_provider(provider)
        assert get_llm_provider() is provider

        # cleanup
        set_llm_provider(None)

    def test_configure_openai(self):
        provider = configure_openai_provider(api_key="test-key", model="gpt-4")

        assert isinstance(provider, OpenAILLMProvider)
        assert provider._model == "gpt-4"
        assert get_llm_provider() is provider

        # cleanup
        set_llm_provider(None)
