# Author: Bradley R. Kinnard
"""
LLM-based mutation provider for opt-in belief mutation.
Uses OpenAI-compatible API for generating nuanced belief variants.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


@dataclass
class LLMMutationResult:
    """Result from LLM mutation."""
    original_content: str
    mutated_content: str
    strategy: str
    reasoning: Optional[str] = None
    model: str = "unknown"
    tokens_used: int = 0


class LLMMutationProvider(ABC):
    """Abstract interface for LLM-based mutation."""

    @abstractmethod
    async def mutate(
        self,
        belief: Belief,
        contradicting: Optional[Belief] = None,
        context: Optional[str] = None,
    ) -> Optional[LLMMutationResult]:
        """Generate mutated belief content using LLM."""
        ...


class MockLLMProvider(LLMMutationProvider):
    """Mock provider for testing without real API calls."""

    async def mutate(
        self,
        belief: Belief,
        contradicting: Optional[Belief] = None,
        context: Optional[str] = None,
    ) -> Optional[LLMMutationResult]:
        # simple mock: add hedge prefix
        mutated = f"It appears that {belief.content[0].lower()}{belief.content[1:]}"
        return LLMMutationResult(
            original_content=belief.content,
            mutated_content=mutated,
            strategy="llm_hedge",
            reasoning="Added epistemic hedge",
            model="mock",
            tokens_used=0,
        )


class OpenAILLMProvider(LLMMutationProvider):
    """
    OpenAI-compatible LLM mutation provider.
    Works with OpenAI, Azure OpenAI, or compatible APIs.
    """

    SYSTEM_PROMPT = """You are a belief mutation specialist. Your task is to take a belief
that has high tension (contradiction pressure) and low confidence, and propose a more
nuanced version that might better coexist with contradicting information.

Strategies you can use:
1. HEDGE: Add epistemic uncertainty ("It may be that...", "Evidence suggests...")
2. CONDITION: Add temporal/contextual conditions ("As of 2024...", "In certain contexts...")
3. SCOPE_NARROW: Reduce absolute claims ("always" → "usually", "all" → "most")
4. SOURCE_ATTRIBUTE: Attribute to source ("According to X...")
5. SYNTHESIS: Try to integrate both perspectives if a contradiction is provided

Output format:
STRATEGY: <strategy_name>
MUTATED: <the new belief text>
REASONING: <brief explanation>"""

    USER_PROMPT_TEMPLATE = """Original belief (confidence: {confidence:.0%}, tension: {tension:.2f}):
"{content}"

{contradiction_section}

{context_section}

Please propose a mutated version of this belief that might better handle the tension."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required for LLM mutation. Install with: pip install openai")

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**kwargs)

        return self._client

    def _build_user_prompt(
        self,
        belief: Belief,
        contradicting: Optional[Belief] = None,
        context: Optional[str] = None,
    ) -> str:
        contradiction_section = ""
        if contradicting:
            contradiction_section = f'Contradicting belief:\n"{contradicting.content}"'

        context_section = ""
        if context:
            context_section = f"Context: {context}"

        return self.USER_PROMPT_TEMPLATE.format(
            content=belief.content,
            confidence=belief.confidence,
            tension=belief.tension,
            contradiction_section=contradiction_section,
            context_section=context_section,
        )

    def _parse_response(self, text: str, original: str) -> LLMMutationResult:
        """Parse LLM response into structured result."""
        lines = text.strip().split("\n")
        strategy = "llm_unknown"
        mutated = original
        reasoning = None

        for line in lines:
            if line.upper().startswith("STRATEGY:"):
                strategy = f"llm_{line.split(':', 1)[1].strip().lower()}"
            elif line.upper().startswith("MUTATED:"):
                mutated = line.split(":", 1)[1].strip().strip('"')
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return LLMMutationResult(
            original_content=original,
            mutated_content=mutated,
            strategy=strategy,
            reasoning=reasoning,
            model=self._model,
        )

    async def mutate(
        self,
        belief: Belief,
        contradicting: Optional[Belief] = None,
        context: Optional[str] = None,
    ) -> Optional[LLMMutationResult]:
        if not self._api_key:
            logger.warning("No API key configured for LLM mutation")
            return None

        try:
            client = self._get_client()
            user_prompt = self._build_user_prompt(belief, contradicting, context)

            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            text = response.choices[0].message.content or ""
            result = self._parse_response(text, belief.content)
            result.tokens_used = response.usage.total_tokens if response.usage else 0

            logger.info(f"LLM mutation: {result.strategy} ({result.tokens_used} tokens)")
            return result

        except Exception as e:
            logger.error(f"LLM mutation failed: {e}")
            return None


# singleton provider
_llm_provider: Optional[LLMMutationProvider] = None


def get_llm_provider() -> Optional[LLMMutationProvider]:
    """Get the configured LLM mutation provider."""
    return _llm_provider


def set_llm_provider(provider: LLMMutationProvider) -> None:
    """Configure the LLM mutation provider."""
    global _llm_provider
    _llm_provider = provider


def configure_openai_provider(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> OpenAILLMProvider:
    """Configure and set the OpenAI LLM provider."""
    provider = OpenAILLMProvider(api_key=api_key, model=model, **kwargs)
    set_llm_provider(provider)
    return provider


__all__ = [
    "LLMMutationResult",
    "LLMMutationProvider",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "get_llm_provider",
    "set_llm_provider",
    "configure_openai_provider",
]
