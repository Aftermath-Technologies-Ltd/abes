# Author: Bradley R. Kinnard
"""
Hybrid LLM Provider - Routes queries between local (Ollama) and online (OpenAI).

Uses local LLM for belief-grounded responses and online LLM only when:
- Query requires real-time/live information
- Query needs current events, weather, traffic, news
- Query explicitly asks to "look up" or "search for" something

Routing is done via zero-shot classification with regex fallback.
"""

import logging
from typing import AsyncIterator, List, Optional

from ..core.config import settings
from ..core.models.belief import Belief
from .openai_provider import OpenAIProvider
from .provider import ChatMessage, ChatResponse, OllamaProvider, StreamChunk
from .query_classifier import needs_real_time_info, classify_query

logger = logging.getLogger(__name__)


class HybridProvider:
    """
    Routes between Ollama (local) and OpenAI (online) based on query type.

    - Local Ollama: For belief-grounded responses, general conversation
    - Online OpenAI: For real-time info (weather, traffic, news, etc.)
    """

    # Enhanced system prompt for live queries
    LIVE_QUERY_SYSTEM_PROMPT = """You are ABES with access to current information.

The user is asking about real-time or current information. Answer based on your knowledge,
but be clear about what you know vs what might need verification.

What you know about the user:
{belief_context}

IMPORTANT:
- For time-sensitive queries, give your best answer but note the date/time limitations
- Be helpful but honest about information freshness
- If you don't have current data, say so clearly"""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b-instruct-q4_0",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
        timeout: float = 120.0,
    ):
        self._ollama = OllamaProvider(
            base_url=ollama_base_url,
            model=ollama_model,
            timeout=timeout,
        )
        self._openai = OpenAIProvider(
            api_key=openai_api_key or settings.openai_api_key,
            model=openai_model or settings.openai_model,
            timeout=timeout,
        )
        self._openai_available = bool(openai_api_key or settings.openai_api_key)

    def _get_last_user_message(self, messages: List[ChatMessage]) -> str:
        """Extract the last user message for routing."""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    async def chat(
        self,
        messages: List[ChatMessage],
        beliefs: Optional[List[Belief]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Route chat to appropriate provider based on query type."""
        last_message = self._get_last_user_message(messages)

        # Classify query using zero-shot classification
        route_type, confidence = classify_query(last_message)

        if self._openai_available and route_type == "real-time" and confidence >= 0.6:
            logger.info(f"Routing to OpenAI (conf={confidence:.2f}): {last_message[:50]}...")
            return await self._openai.chat(messages, beliefs, temperature, max_tokens)

        # Default to local Ollama for belief-grounded and general queries
        logger.debug(f"Routing to Ollama ({route_type}, conf={confidence:.2f})")
        return await self._ollama.chat(messages, beliefs, temperature, max_tokens)

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        beliefs: Optional[List[Belief]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """Route streaming chat to appropriate provider."""
        last_message = self._get_last_user_message(messages)

        route_type, confidence = classify_query(last_message)

        if self._openai_available and route_type == "real-time" and confidence >= 0.6:
            logger.info(f"Routing stream to OpenAI (conf={confidence:.2f})")
            async for chunk in self._openai.chat_stream(messages, beliefs, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in self._ollama.chat_stream(messages, beliefs, temperature, max_tokens):
                yield chunk

    async def health_check(self) -> bool:
        """Check if primary (Ollama) provider is healthy."""
        return await self._ollama.health_check()

    async def close(self) -> None:
        """Close both providers."""
        await self._ollama.close()
        await self._openai.close()


__all__ = ["HybridProvider", "classify_query", "needs_real_time_info"]
