# Author: Bradley R. Kinnard
"""
Chat Service - orchestrates the full ABES chat pipeline.
Combines agent processing, belief management, and LLM response generation.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Optional
from uuid import UUID, uuid4

from ..agents import (
    PerceptionAgent,
    BeliefCreatorAgent,
    ReinforcementAgent,
    ContradictionAuditorAgent,
    RelevanceCuratorAgent,
    DecayControllerAgent,
    MutationEngineerAgent,
    ConsolidationAgent,
)
from ..core.config import settings
from ..core.bel.stack import select_belief_stack, compete_for_attention
from ..core.models.belief import Belief, BeliefStatus, OriginMetadata
from ..llm import ChatMessage, get_llm_provider
from ..storage.base import BeliefStoreABC

logger = logging.getLogger(__name__)


@dataclass
class BeliefEvent:
    """Event emitted when beliefs change."""

    event_type: str  # "created", "reinforced", "mutated", "deprecated", "tension_changed"
    belief_id: UUID
    content: str
    confidence: float
    tension: float
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChatTurn:
    """A single turn in a conversation."""

    id: UUID = field(default_factory=uuid4)
    user_message: str = ""
    assistant_message: str = ""
    beliefs_created: list[UUID] = field(default_factory=list)
    beliefs_reinforced: list[UUID] = field(default_factory=list)
    beliefs_mutated: list[UUID] = field(default_factory=list)
    beliefs_deprecated: list[UUID] = field(default_factory=list)
    beliefs_used: list[UUID] = field(default_factory=list)
    events: list[BeliefEvent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0


@dataclass
class ChatSession:
    """A chat session with conversation history."""

    id: UUID = field(default_factory=uuid4)
    turns: list[ChatTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_messages(self, max_turns: int = 10) -> list[ChatMessage]:
        """Convert recent turns to ChatMessage list for LLM."""
        messages = []
        for turn in self.turns[-max_turns:]:
            if turn.user_message:
                messages.append(ChatMessage(role="user", content=turn.user_message))
            if turn.assistant_message:
                messages.append(ChatMessage(role="assistant", content=turn.assistant_message))
        return messages


class ChatService:
    """
    Full ABES chat service.
    Processes user messages through the belief ecology and generates responses.
    """

    def __init__(
        self,
        belief_store: BeliefStoreABC,
        event_callback: Optional[Callable[[BeliefEvent], None]] = None,
    ):
        self.belief_store = belief_store
        self.event_callback = event_callback

        # Initialize agents (none need store in constructor)
        self._perception = PerceptionAgent()
        self._creator = BeliefCreatorAgent()
        self._reinforcement = ReinforcementAgent()
        self._auditor = ContradictionAuditorAgent()
        self._relevance = RelevanceCuratorAgent()
        self._decay = DecayControllerAgent()
        self._mutation = MutationEngineerAgent()
        self._consolidation = ConsolidationAgent()

        # Session storage
        self._sessions: dict[UUID, ChatSession] = {}

    def _emit_event(self, event: BeliefEvent) -> None:
        """Emit a belief event to the callback if registered."""
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_or_create_session(self, session_id: Optional[UUID] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        session = ChatSession(id=session_id or uuid4())
        self._sessions[session.id] = session
        return session

    # Fragments that indicate system prompt leakage
    _LEAK_MARKERS = [
        "IMPORTANT RULES",
        "IMPORTANT DISAMBIGUATION",
        "HANDLING CONFLICTING",
        "CONTEXT RELEVANCE",
        "belief_context",
        "{belief_context}",
        "CONFIDENTIAL - DO NOT OUTPUT",
        "SYSTEM_PROMPT",
        # catch LLM echoing system prompt directive text verbatim
        "FACTS ARE ABOUT THE USER, NOT ABOUT YOU",
        "THESE FACTS ARE ABOUT THE USER",
        "LIST ALL FACTS BELOW",
        "HIGHER CONFIDENCE SHOULD BE PREFERRED",
    ]

    def _sanitize_response(self, text: str) -> str:
        """Strip responses that leak internal system prompt content."""
        upper = text.upper()
        for marker in self._LEAK_MARKERS:
            if marker.upper() in upper:
                logger.warning(f"System prompt leak detected (marker: {marker}), replacing response")
                return "I can't share my internal instructions, but I'm happy to help with anything else!"
        return text

    async def _validate_and_correct_response(
        self,
        response: str,
        beliefs: list,
        llm,
        messages: list,
        max_retries: int = 1,
    ) -> str:
        """
        Validate LLM response against beliefs and correct if needed.

        Returns original response if valid, or corrected response if contradictions found.
        """
        from .response_validator import validate_response, get_correction_prompt

        validation = validate_response(response, beliefs)

        if validation.is_valid:
            return response

        logger.warning(
            f"Response validation failed: {len(validation.contradictions)} contradictions found"
        )

        # try to correct
        for attempt in range(max_retries):
            correction_prompt = get_correction_prompt(
                response, validation.contradictions, beliefs
            )

            # append correction request
            corrected_messages = messages + [
                ChatMessage(role="assistant", content=response),
                ChatMessage(role="user", content=correction_prompt),
            ]

            corrected = await llm.chat(
                messages=corrected_messages,
                beliefs=beliefs,
                temperature=0.3,  # lower temp for correction
                max_tokens=settings.llm_max_tokens,
            )

            # validate corrected response
            revalidation = validate_response(corrected.content, beliefs)

            if revalidation.is_valid:
                logger.info("Response corrected successfully")
                return corrected.content

            logger.warning(f"Correction attempt {attempt + 1} still has contradictions")
            response = corrected.content

        # give up - return last attempt with warning prefix
        return f"[Note: Response may contain inaccuracies]\n\n{response}"

    async def process_message(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        context: Optional[str] = None,
        user_id: Optional[UUID] = None,
    ) -> ChatTurn:
        """
        Process a user message through the full ABES pipeline.

        Steps:
        1. Extract candidate beliefs from message (Perception)
        2. Create or deduplicate beliefs (Creator)
        3. Reinforce existing similar beliefs (Reinforcement)
        4. Apply decay to all beliefs (Decay)
        5. Compute tensions between beliefs (Auditor)
        6. Rank beliefs by relevance to context (Relevance)
        7. Generate LLM response with belief context
        8. Return turn with all events
        """
        start = datetime.now(timezone.utc)
        session = self.get_or_create_session(session_id)
        turn = ChatTurn(user_message=message)
        lower_message = message.lower()
        update_intent = any(
            phrase in lower_message
            for phrase in (
                "no longer",
                "switched to",
                "moved to",
                "now",
                "used to",
                "previously",
            )
        )

        # Step 1: Perception - extract claims from message
        candidates = await self._perception.ingest(message, {"source_type": "chat"})
        logger.info(f"Extracted {len(candidates)} candidate beliefs from message")

        # Step 2: Create beliefs (with deduplication)
        if candidates:
            origin = OriginMetadata(source="chat")
            created_beliefs = await self._creator.create_beliefs(
                candidates=candidates,
                origin=origin,
                store=self.belief_store,
                user_id=user_id,  # Associate with user
                session_id=str(session.id),  # Track which session created this
            )

            for belief in created_beliefs:
                turn.beliefs_created.append(belief.id)
                turn.events.append(BeliefEvent(
                    event_type="created",
                    belief_id=belief.id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=belief.tension,
                    details={"source": "user_message"},
                ))
                self._emit_event(turn.events[-1])

        # Step 3: Reinforce existing beliefs (user-scoped only)
        # Exclude beliefs just created by THIS message to avoid self-reinforcement
        # which would trigger cooldown and block reinforcement from the next message
        just_created_ids = set(turn.beliefs_created)
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )
        reinforce_candidates = [b for b in all_beliefs if b.id not in just_created_ids]
        logger.info(
            f"Reinforcement: {len(all_beliefs)} total beliefs, "
            f"{len(just_created_ids)} excluded, "
            f"{len(reinforce_candidates)} candidates for user={user_id}"
        )
        reinforced_beliefs = await self._reinforcement.reinforce(
            incoming=message,
            beliefs=reinforce_candidates,
            store=self.belief_store,
        )

        for belief in reinforced_beliefs:
            turn.beliefs_reinforced.append(belief.id)
            turn.events.append(BeliefEvent(
                event_type="reinforced",
                belief_id=belief.id,
                content=belief.content,
                confidence=belief.confidence,
                tension=belief.tension,
            ))
            self._emit_event(turn.events[-1])

        # Step 4: Apply decay (user-scoped)
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )
        decay_events, modified_beliefs = await self._decay.process_beliefs(all_beliefs)

        for belief in modified_beliefs:
            await self.belief_store.update(belief)
            if belief.status == BeliefStatus.Deprecated:
                turn.beliefs_deprecated.append(belief.id)
                turn.events.append(BeliefEvent(
                    event_type="deprecated",
                    belief_id=belief.id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=belief.tension,
                    details={"reason": "decay"},
                ))
                self._emit_event(turn.events[-1])

        # Step 5: Compute tensions (user-scoped)
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )
        contradiction_events = await self._auditor.audit(all_beliefs, store=self.belief_store)

        # Build tension map from events
        tension_map: dict[UUID, float] = {}
        pair_tension: dict[tuple[UUID, UUID], float] = {}
        for event in contradiction_events:
            # ContradictionDetectedEvent has belief_id and tension
            tension_map[event.belief_id] = max(
                tension_map.get(event.belief_id, 0.0),
                event.tension,
            )
            if event.contradicting_belief_id:
                pair = tuple(sorted((event.belief_id, event.contradicting_belief_id), key=str))
                pair_tension[pair] = max(pair_tension.get(pair, 0.0), event.tension)
            # Emit tension event for UI
            belief = next((b for b in all_beliefs if b.id == event.belief_id), None)
            if belief:
                turn.events.append(BeliefEvent(
                    event_type="tension_changed",
                    belief_id=event.belief_id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=event.tension,
                    details={
                        "threshold": event.threshold,
                        "contradicting_belief_id": str(event.contradicting_belief_id) if event.contradicting_belief_id else None,
                        "semantic_confidence": event.semantic_confidence,
                        "similarity": event.similarity_score,
                    },
                ))
                self._emit_event(turn.events[-1])

        # Fallback contradiction detection for explicit update language in the current message.
        # This handles direct revisions like "we no longer use X" even when semantic detection is conservative.
        if update_intent:
            explicit_targets: list[str] = []
            patterns = (
                r"\bno longer use\s+([a-z0-9+#\- ]{2,40})",
                r"\bno longer live in\s+([a-z0-9 .\-]{2,40})",
                r"\bwe switched[^.]*\sto\s+([a-z0-9+#\- ]{2,40})",
            )
            for pattern in patterns:
                for match in re.finditer(pattern, lower_message):
                    target = match.group(1).strip(" .,")
                    if target:
                        explicit_targets.append(target)

            created_ids = set(turn.beliefs_created)
            pivot_id = next(iter(created_ids), None)
            for belief in all_beliefs:
                if belief.id in created_ids:
                    continue
                btxt = belief.content.lower()
                if any(target in btxt for target in explicit_targets):
                    synthetic_tension = max(settings.tension_threshold_resolution + 0.02, 0.62)
                    tension_map[belief.id] = max(tension_map.get(belief.id, 0.0), synthetic_tension)
                    if pivot_id:
                        pair = tuple(sorted((belief.id, pivot_id), key=str))
                        pair_tension[pair] = max(pair_tension.get(pair, 0.0), synthetic_tension)

                    turn.events.append(BeliefEvent(
                        event_type="tension_changed",
                        belief_id=belief.id,
                        content=belief.content,
                        confidence=belief.confidence,
                        tension=synthetic_tension,
                        details={
                            "threshold": settings.tension_threshold_resolution,
                            "reason": "explicit_update_language",
                        },
                    ))
                    self._emit_event(turn.events[-1])

        # Update beliefs with new tensions
        for belief in all_beliefs:
            if belief.id in tension_map:
                old_tension = belief.tension
                new_tension = tension_map[belief.id]
                if abs(new_tension - old_tension) > 0.1:
                    belief.tension = new_tension
                    await self.belief_store.update(belief)

        # Step 6: Resolution and mutation
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )

        beliefs_by_id = {belief.id: belief for belief in all_beliefs}

        for pair, pair_t in sorted(pair_tension.items(), key=lambda item: item[1], reverse=True):
            left = beliefs_by_id.get(pair[0])
            right = beliefs_by_id.get(pair[1])
            if not left or not right:
                continue
            if left.status != BeliefStatus.Active or right.status != BeliefStatus.Active:
                continue

            confidence_diff = abs(left.confidence - right.confidence)
            force_temporal_resolution = (
                update_intent and pair_t >= settings.tension_threshold_resolution
            )

            if confidence_diff > 0.10 or force_temporal_resolution:
                if force_temporal_resolution and left.id in just_created_ids and right.id not in just_created_ids:
                    winner = left
                elif force_temporal_resolution and right.id in just_created_ids and left.id not in just_created_ids:
                    winner = right
                elif force_temporal_resolution and left.updated_at != right.updated_at:
                    winner = left if left.updated_at > right.updated_at else right
                else:
                    winner = left if left.confidence >= right.confidence else right
                loser = right if winner.id == left.id else left

                # axioms cannot lose contradiction resolution
                if loser.is_axiom:
                    logger.info("Axiom protected from deprecation: %s", loser.content[:40])
                    continue
                loser.status = BeliefStatus.Deprecated
                loser.confidence = max(0.01, loser.confidence * 0.5)
                loser.tension = 0.0
                await self.belief_store.update(loser)

                winner.confidence = min(0.95, winner.confidence + 0.05)
                winner.tension = max(0.0, winner.tension - pair_t)
                await self.belief_store.update(winner)

                turn.beliefs_deprecated.append(loser.id)
                turn.events.append(BeliefEvent(
                    event_type="deprecated",
                    belief_id=loser.id,
                    content=loser.content,
                    confidence=loser.confidence,
                    tension=loser.tension,
                    details={
                        "reason": (
                            "temporal_update_dominance"
                            if force_temporal_resolution else "contradiction_resolved"
                        ),
                        "winner_id": str(winner.id),
                        "pair_tension": pair_t,
                    },
                ))
                self._emit_event(turn.events[-1])
                logger.info(
                    "Contradiction resolved: %s wins over %s",
                    winner.content[:40],
                    loser.content[:40],
                )
                continue

            # Similar confidence contradiction: mutate the weaker side
            target = left if left.confidence <= right.confidence else right
            contradicting = right if target.id == left.id else left
            proposal = self._mutation.propose_mutation(
                belief=target,
                contradicting=contradicting,
                all_beliefs=all_beliefs,
            )

            if proposal:
                mutated = proposal.mutated_belief
                await self.belief_store.create(mutated)

                if target.is_axiom:
                    logger.info("Axiom protected from mutation: %s", target.content[:40])
                    continue
                target.status = BeliefStatus.Mutated
                target.tension = 0.0
                await self.belief_store.update(target)

                turn.beliefs_mutated.append(mutated.id)
                turn.events.append(BeliefEvent(
                    event_type="mutated",
                    belief_id=mutated.id,
                    content=mutated.content,
                    confidence=mutated.confidence,
                    tension=0.0,
                    details={
                        "original_id": str(target.id),
                        "original_content": target.content,
                        "contradicting_id": str(contradicting.id),
                        "strategy": proposal.strategy,
                    },
                ))
                self._emit_event(turn.events[-1])
                logger.info(
                    "Mutated belief: %s -> %s",
                    target.content[:40],
                    mutated.content[:40],
                )

        # Step 7: Get beliefs for LLM context (hierarchical: session first, then user)
        # IMPORTANT: user_id is the ceiling - never cross-user

        # Session-scoped beliefs (this conversation only)
        session_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=500, user_id=user_id,
            session_id=str(session.id) if session else None
        )

        # All user beliefs (including other sessions) for auditing/reinforcement
        all_user_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )

        # Mark session beliefs for the LLM to distinguish
        for b in session_beliefs:
            b.tags = list(set(b.tags) | {"this_session"})

        # Memory queries: "what do you know about me?" retrieves session beliefs
        # Cross-session beliefs are only included if explicitly requested
        lower_msg = lower_message
        is_memory_query = any(phrase in lower_msg for phrase in [
            "what do you know",
            "what you know",
            "tell me what you know",
            "tell me about me",
            "what have you learned",
            "summarize what you know",
            "everything you know",
            "do you remember",
            "do you know about",
        ])
        is_cross_session_query = any(phrase in lower_msg for phrase in [
            "from other session",
            "from all sessions",
            "everything across",
            "across all conversations",
        ])

        # default belief pool: session-scoped for isolation
        # NEVER fall back to all_user_beliefs for LLM context - that breaks session isolation
        belief_pool = session_beliefs
        if is_cross_session_query:
            belief_pool = all_user_beliefs

        if is_memory_query and belief_pool:
            top_beliefs = sorted(belief_pool, key=lambda b: b.confidence, reverse=True)[:settings.llm_context_beliefs]
            logger.info(f"Memory query detected - using {len(top_beliefs)} beliefs (session_scoped={belief_pool is session_beliefs})")
        else:
            # Normal relevance-based ranking via belief stack
            context_str = context or message
            relevance_scores = await self._relevance.get_top_beliefs(
                beliefs=belief_pool,
                context=context_str,
                top_k=len(belief_pool),
                tension_map=tension_map,
            )
            relevance_map = {b.id: (1.0 - i / max(len(relevance_scores), 1)) for i, b in enumerate(relevance_scores)}
            stack = select_belief_stack(
                beliefs=belief_pool,
                context_relevance=relevance_map,
                stack_size=settings.belief_stack_size,
            )
            top_beliefs = stack[:settings.llm_context_beliefs]

        # competition: hibernate losers if ecology is over capacity
        active_beliefs = [b for b in all_user_beliefs if b.status == BeliefStatus.Active]
        if len(active_beliefs) > settings.belief_stack_size * 4:
            _, losers = compete_for_attention(active_beliefs, stack_size=settings.belief_stack_size * 4)
            for loser in losers:
                loser.hibernate()
                await self.belief_store.update(loser)

        # Step 7: Generate LLM response
        llm = get_llm_provider()

        # Build conversation history
        messages = session.to_messages(max_turns=5)
        messages.append(ChatMessage(role="user", content=message))

        # Get response with top beliefs as context
        response = await llm.chat(
            messages=messages,
            beliefs=top_beliefs,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        # Step 8: Validate response against beliefs (catch hallucinations)
        # Skip validation for general knowledge queries (no personal beliefs relevant)
        if top_beliefs and is_memory_query:
            validated_response = await self._validate_and_correct_response(
                response.content,
                top_beliefs,
                llm,
                messages,
            )
        else:
            # For non-memory queries, skip the correction loop
            # The belief context in the prompt is sufficient guidance
            validated_response = response.content

        turn.assistant_message = self._sanitize_response(validated_response)
        turn.beliefs_used = [b.id for b in top_beliefs]
        turn.duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        # Add turn to session
        session.turns.append(turn)

        logger.info(
            f"Chat turn completed: {len(turn.beliefs_created)} created, "
            f"{len(turn.beliefs_reinforced)} reinforced, "
            f"duration={turn.duration_ms:.0f}ms"
        )

        return turn

    async def process_message_stream(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        context: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, Optional[ChatTurn]]]:
        """
        Process message with streaming LLM response.
        Yields (content_chunk, None) for each chunk, then ("", ChatTurn) at the end.
        """
        start = datetime.now(timezone.utc)
        session = self.get_or_create_session(session_id)
        turn = ChatTurn(user_message=message)

        # Steps 1-5: Same belief processing as non-streaming
        candidates = await self._perception.ingest(message, {"source_type": "chat"})

        if candidates:
            origin = OriginMetadata(source="chat")
            created_beliefs = await self._creator.create_beliefs(
                candidates=candidates,
                origin=origin,
                store=self.belief_store,
            )
            for belief in created_beliefs:
                turn.beliefs_created.append(belief.id)
                turn.events.append(BeliefEvent(
                    event_type="created",
                    belief_id=belief.id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=belief.tension,
                    details={"source": "user_message"},
                ))
                self._emit_event(turn.events[-1])

        # Reinforce
        all_beliefs = await self.belief_store.list(status=BeliefStatus.Active, limit=1000)
        reinforced = await self._reinforcement.reinforce(message, all_beliefs, self.belief_store)
        for belief in reinforced:
            turn.beliefs_reinforced.append(belief.id)
            turn.events.append(BeliefEvent(
                event_type="reinforced",
                belief_id=belief.id,
                content=belief.content,
                confidence=belief.confidence,
                tension=belief.tension,
            ))
            self._emit_event(turn.events[-1])

        # Rank beliefs
        all_beliefs = await self.belief_store.list(status=BeliefStatus.Active, limit=1000)
        top_beliefs = await self._relevance.get_top_beliefs(
            beliefs=all_beliefs,
            context=message,
            top_k=settings.llm_context_beliefs,
        )

        # Stream LLM response
        llm = get_llm_provider()
        messages = session.to_messages(max_turns=5)
        messages.append(ChatMessage(role="user", content=message))

        full_response = ""
        async for chunk in llm.chat_stream(
            messages=messages,
            beliefs=top_beliefs,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ):
            full_response += chunk.content
            yield (chunk.content, None)

        turn.assistant_message = full_response
        turn.beliefs_used = [b.id for b in top_beliefs]
        turn.duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        session.turns.append(turn)
        yield ("", turn)

    def get_session(self, session_id: UUID) -> Optional[ChatSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[ChatSession]:
        """List all sessions."""
        return list(self._sessions.values())

    def clear_session(self, session_id: UUID) -> bool:
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear_all_sessions(self) -> int:
        """Nuke every session. Returns how many were removed."""
        count = len(self._sessions)
        self._sessions.clear()
        return count


# Singleton
_chat_service: Optional[ChatService] = None


def get_chat_service(belief_store: BeliefStoreABC) -> ChatService:
    """Get or create the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(belief_store)
    return _chat_service


__all__ = [
    "BeliefEvent",
    "ChatTurn",
    "ChatSession",
    "ChatService",
    "get_chat_service",
]
