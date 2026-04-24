# Author: Bradley R. Kinnard
"""
Anthropic LLM Provider for ABES.
Supports Claude 3 family of models.
"""

import json
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


class AnthropicProvider:
    """
    Anthropic Claude provider for ABES.
    Uses the Messages API with belief context injection.
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
        model: str = "claude-3-haiku-20240307",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
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
        """Generate a chat response using Anthropic API."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY env var.")

        beliefs = beliefs or []
        client = await self._get_client()

        # Anthropic uses system as a separate parameter
        anthropic_messages = []
        for msg in messages:
            if msg.role != "system":
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        start = datetime.now(timezone.utc)

        response = await client.post(
            f"{self.base_url}/messages",
            json={
                "model": self.model,
                "system": self._build_system_prompt(beliefs),
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()

        data = response.json()
        duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        # Extract content from Anthropic response format
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})

        return ChatResponse(
            content=content,
            model=data.get("model", self.model),
            tokens_prompt=usage.get("input_tokens", 0),
            tokens_completion=usage.get("output_tokens", 0),
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
        """Stream a chat response from Anthropic."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        beliefs = beliefs or []
        client = await self._get_client()

        anthropic_messages = []
        for msg in messages:
            if msg.role != "system":
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        async with client.stream(
            "POST",
            f"{self.base_url}/messages",
            json={
                "model": self.model,
                "system": self._build_system_prompt(beliefs),
                "messages": anthropic_messages,
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
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")

                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(
                            content=delta.get("text", ""),
                            done=False,
                            model=self.model,
                        )

                elif event_type == "message_stop":
                    yield StreamChunk(content="", done=True, model=self.model)
                    break

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self.api_key:
            return False
        try:
            # Anthropic doesn't have a simple health endpoint, so we just check if we can connect
            client = await self._get_client()
            # Try a minimal request - this will fail with 400 but proves connectivity
            response = await client.post(
                f"{self.base_url}/messages",
                json={"model": self.model, "messages": [], "max_tokens": 1},
            )
            # Even a 400 error means the API is reachable
            return response.status_code in (200, 400)
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


__all__ = ["AnthropicProvider"]
