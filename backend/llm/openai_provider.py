# Author: Bradley R. Kinnard
"""
OpenAI LLM Provider for ABES.
Supports GPT-4, GPT-4o, GPT-4o-mini and other OpenAI models.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional
from uuid import UUID

import httpx

from ..core.config import settings
from ..core.models.belief import Belief
from .provider import ChatMessage, ChatResponse, StreamChunk

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI LLM provider for ABES.
    Uses the same belief context injection as OllamaProvider.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are ABES, a conversational AI assistant with persistent memory.

You remember facts about the user across conversations. The facts below are things the USER told you about THEMSELVES.

IMPORTANT DISAMBIGUATION:
- "What is MY name?" = the USER is asking about THEMSELVES. Answer using the facts below.
- "What is YOUR name?" = the user is asking about YOU. Your name is ABES.
- NEVER confuse these. When the user says "my", they mean their own information.

CRITICAL RULES:
1. These facts are ABOUT THE USER, not about you
2. When asked "what do you know about me?" list ALL facts below
3. Never claim the conversation "just started" if there are facts stored
4. Never deny having information that is listed below
5. Keep responses natural — don't over-explain your own workings

What you know about the user:
{belief_context}

Guidelines:
1. Use these facts to give personalized responses
2. Refer to user's info correctly
3. When new information arrives, acknowledge it naturally"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _format_belief_context(self, beliefs: List[Belief], max_beliefs: int = 15) -> str:
        if not beliefs:
            return "No information learned about the user yet."

        sorted_beliefs = sorted(
            beliefs,
            key=lambda b: (getattr(b, "score", 0.0), b.confidence),
            reverse=True,
        )[:max_beliefs]

        lines = []
        for b in sorted_beliefs:
            conf_pct = int(b.confidence * 100)
            tension_indicator = " ⚠️" if b.tension > 0.3 else ""
            lines.append(f"- {b.content} ({conf_pct}%{tension_indicator})")

        return "\n".join(lines)

    def _build_system_prompt(self, beliefs: List[Belief]) -> str:
        context = self._format_belief_context(beliefs)
        return self.SYSTEM_PROMPT_TEMPLATE.format(belief_context=context)

    async def chat(
        self,
        messages: List[ChatMessage],
        beliefs: List[Belief] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Generate a chat response using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY env var.")

        beliefs = beliefs or []
        client = await self._get_client()

        openai_messages = [
            {"role": "system", "content": self._build_system_prompt(beliefs)}
        ]

        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        start = datetime.now(timezone.utc)

        response = await client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()

        data = response.json()
        duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})

        return ChatResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_completion=usage.get("completion_tokens", 0),
            duration_ms=duration,
            beliefs_used=[b.id for b in beliefs],
        )

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        beliefs: List[Belief] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat response from OpenAI."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        beliefs = beliefs or []
        client = await self._get_client()

        openai_messages = [
            {"role": "system", "content": self._build_system_prompt(beliefs)}
        ]

        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    yield StreamChunk(content="", done=True, model=self.model)
                    break

                import json
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    yield StreamChunk(
                        content=content,
                        done=False,
                        model=data.get("model", self.model),
                    )

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            return False
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


__all__ = ["OpenAIProvider"]
