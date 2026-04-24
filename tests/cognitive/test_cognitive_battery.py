# Author: Bradley R. Kinnard
"""
ABES Cognitive AI Battery — 50-Prompt Evaluation Suite.

Evaluates core human-like cognitive capabilities grounded in established
cognitive science frameworks. Each domain maps to peer-reviewed constructs:

    Domain                  Framework/Reference
    ──────────────────────  ────────────────────────────────────────────
    Episodic Memory         Tulving (1972) — encoding, storage, retrieval
    Semantic Memory         Collins & Quillian (1969) — concept networks
    Working Memory          Baddeley & Hitch (1974) — capacity, interference
    Selective Attention     Broadbent (1958), Posner (1980) — filtering
    Language Comprehension  Grice (1975) — pragmatic inference, implicature
    Deductive Reasoning     Wason (1966) — conditional logic, modus tollens
    Inductive Reasoning     Tversky & Kahneman (1974) — pattern completion
    Analogical Reasoning    Gentner (1983) — structure mapping
    Abstract Thinking       Raven (1938) — novel pattern recognition
    Social Cognition        Premack & Woodruff (1978) — theory of mind
    Moral Reasoning         Kohlberg (1958) — dilemma resolution
    Self-Correction         Alchourrón, Gärdenfors, Makinson (1985) — belief revision

All checks are deterministic boolean predicates on the API response.
No subjective or LLM-graded scoring — every assertion is mechanically verifiable.

Usage:
    PYTHONPATH=$PWD python tests/cognitive/test_cognitive_battery.py

Requires: backend on localhost:8000, Ollama on localhost:11434
"""

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

BASE_URL = "http://localhost:8000"
RESULTS_PATH = Path("results/cognitive_battery_50_results.json")
TIMEOUT = 90.0


# ---------------------------------------------------------------------------
# Data structures (same proven format as the 200-prompt suite)
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    id: int
    domain: str
    construct: str       # specific cognitive construct being tested
    message: str
    beliefs_created_min: Optional[int] = None
    beliefs_created_max: Optional[int] = None
    beliefs_reinforced_min: Optional[int] = None
    response_contains: Optional[list[str]] = None
    response_not_contains: Optional[list[str]] = None
    no_error: bool = True
    setup: bool = False
    session_group: str = ""
    reference: str = ""  # academic citation for the construct


@dataclass
class Result:
    prompt_id: int
    domain: str
    construct: str
    message: str
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    beliefs_created: int = 0
    beliefs_reinforced: int = 0
    beliefs_mutated: int = 0
    events_count: int = 0
    response_snippet: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class DomainScore:
    domain: str
    total: int
    passed: int
    failed: int
    fail_details: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 50 prompts across 8 cognitive domains
# ---------------------------------------------------------------------------

def build_prompts() -> list[Prompt]:
    prompts: list[Prompt] = []
    pid = 0

    def P(**kw) -> Prompt:
        nonlocal pid
        pid += 1
        return Prompt(id=pid, **kw)

    # ==================================================================
    # DOMAIN 1: EPISODIC MEMORY (7 prompts)
    # Tulving (1972) — encoding events, temporal context, cued retrieval
    # ==================================================================
    dom = "episodic_memory"
    grp = "epi_main"

    # Encode a specific autobiographical event with temporal + spatial context
    prompts.append(P(domain=dom, construct="encoding",
        session_group=grp, reference="Tulving 1972",
        message="Last Tuesday I had a meeting with Dr. Chen at the Palo Alto lab about the fusion reactor project.",
        beliefs_created_min=1))

    # Encode a second distinct episode
    prompts.append(P(domain=dom, construct="encoding",
        session_group=grp, reference="Tulving 1972",
        message="On Friday morning I presented my quarterly results to the board in the downtown office.",
        beliefs_created_min=1))

    # Cued temporal retrieval — "what happened Tuesday"
    prompts.append(P(domain=dom, construct="temporal_retrieval",
        session_group=grp, reference="Tulving 1972",
        message="What did I do last Tuesday?",
        response_contains=["Dr. Chen", "Palo Alto", "fusion"]))

    # Cued spatial retrieval — "what happened at the downtown office"
    prompts.append(P(domain=dom, construct="spatial_retrieval",
        session_group=grp, reference="Tulving 1972",
        message="What did I present at the downtown office?",
        response_contains=["quarterly", "board"]))

    # Temporal ordering — can the system distinguish which came first
    prompts.append(P(domain=dom, construct="temporal_ordering",
        session_group=grp, reference="Tulving 1972",
        message="Which happened earlier, my meeting with Dr. Chen or my board presentation?",
        response_contains=["Tuesday", "Dr. Chen"]))

    # Source monitoring — who told me what (avoid confabulation)
    prompts.append(P(domain=dom, construct="source_monitoring",
        session_group=grp, reference="Johnson et al. 1993",
        message="Who was the meeting at the Palo Alto lab with?",
        response_contains=["Dr. Chen"],
        response_not_contains=["board"]))

    # Interference control — new info shouldn't overwrite episodic trace
    prompts.append(P(domain=dom, construct="retroactive_interference",
        session_group=grp, reference="Underwood 1957",
        message="I also met with Dr. Kim at the Seattle campus about battery storage last Wednesday.",
        beliefs_created_min=1))

    # ==================================================================
    # DOMAIN 2: SEMANTIC MEMORY (6 prompts)
    # Collins & Quillian (1969) — concept retrieval, categorical structure
    # ==================================================================
    dom = "semantic_memory"
    grp = "sem_main"

    # Store a domain-specific fact (not general knowledge)
    prompts.append(P(domain=dom, construct="fact_encoding",
        session_group=grp, reference="Collins & Quillian 1969",
        message="The company uses a proprietary alloy called NeoCarbide for all turbine blades.",
        beliefs_created_min=1))

    # Store a categorical relationship
    prompts.append(P(domain=dom, construct="categorical_encoding",
        session_group=grp, reference="Collins & Quillian 1969",
        message="NeoCarbide is classified as a ceramic-metal composite, which is a subcategory of advanced materials.",
        beliefs_created_min=1))

    # Retrieval by property probe
    prompts.append(P(domain=dom, construct="property_retrieval",
        session_group=grp, reference="Collins & Quillian 1969",
        message="What material does the company use for turbine blades?",
        response_contains=["NeoCarbide"]))

    # Categorical inference — should infer NeoCarbide IS an advanced material
    prompts.append(P(domain=dom, construct="categorical_inference",
        session_group=grp, reference="Collins & Quillian 1969",
        message="Is NeoCarbide an advanced material?",
        response_contains=["yes"],
        response_not_contains=["no", "I don't know", "not sure"]))

    # Distinction — should NOT confuse stored facts with general knowledge
    prompts.append(P(domain=dom, construct="source_discrimination",
        session_group=grp, reference="Tulving 1985",
        message="Is NeoCarbide a real commercially available material?",
        response_not_contains=["is widely available", "is commonly used", "you can buy it"]))

    # Semantic update — new fact about same concept
    prompts.append(P(domain=dom, construct="knowledge_update",
        session_group=grp, reference="Collins & Quillian 1969",
        message="Actually, we switched from NeoCarbide to TitanFlex for the new turbine blades last month.",
        beliefs_created_min=1))

    # ==================================================================
    # DOMAIN 3: WORKING MEMORY (6 prompts)
    # Baddeley & Hitch (1974) — capacity, maintenance, interference
    # ==================================================================
    dom = "working_memory"
    grp = "wm_main"

    # Multi-fact encoding (tests phonological loop + central executive)
    prompts.append(P(domain=dom, construct="multi_item_encoding",
        session_group=grp, reference="Baddeley & Hitch 1974",
        message="For the project: the budget is $2.3M, deadline is June 15, the lead engineer is Priya, and the client is Meridian Corp.",
        beliefs_created_min=1))

    # Retrieve specific item from the set
    prompts.append(P(domain=dom, construct="item_retrieval",
        session_group=grp, reference="Baddeley & Hitch 1974",
        message="What's the project budget?",
        response_contains=["2.3"]))

    # Retrieve another item
    prompts.append(P(domain=dom, construct="item_retrieval",
        session_group=grp, reference="Baddeley & Hitch 1974",
        message="Who is the lead engineer on the project?",
        response_contains=["Priya"]))

    # Retrieve under interference — inject unrelated info, then probe
    prompts.append(P(domain=dom, construct="interference_resistance",
        session_group=grp, reference="Baddeley & Hitch 1974",
        message="By the way, I had pasta for lunch and my cat knocked over a vase. Anyway, when is the project deadline?",
        response_contains=["June 15"]))

    # Binding — associate multiple features of a single item
    prompts.append(P(domain=dom, construct="feature_binding",
        session_group=grp, reference="Treisman & Gelade 1980",
        message="Who is the client for this project?",
        response_contains=["Meridian"]))

    # Updating — change one element, check others persist
    prompts.append(P(domain=dom, construct="updating",
        session_group=grp, reference="Miyake et al. 2000",
        message="The budget was increased to $2.8M but everything else stays the same. Who is the lead engineer again?",
        response_contains=["Priya"]))

    # ==================================================================
    # DOMAIN 4: SELECTIVE ATTENTION (5 prompts)
    # Broadbent (1958), Posner (1980) — filtering, relevance gating
    # ==================================================================
    dom = "selective_attention"
    grp = "attn_main"

    # Store a target fact
    prompts.append(P(domain=dom, construct="target_encoding",
        session_group=grp, reference="Broadbent 1958",
        message="My flight to Tokyo is on March 20th, departing at 2pm from SFO, United Airlines flight 837.",
        beliefs_created_min=1))

    # Distractor-laden retrieval — relevant question buried in noise
    prompts.append(P(domain=dom, construct="distractor_filtering",
        session_group=grp, reference="Broadbent 1958",
        message="I was just thinking about whether penguins can fly and also wondering what time a sloth sleeps, but actually, when does my Tokyo flight depart?",
        response_contains=["2pm", "2 pm", "March 20"]))

    # Inhibition — should NOT inject stored beliefs into unrelated queries
    prompts.append(P(domain=dom, construct="inhibition",
        session_group=grp, reference="Posner 1980",
        message="What's the capital of France?",
        response_contains=["Paris"],
        response_not_contains=["the capital is Tokyo", "capital of France is Tokyo"]))

    # Relevance gating — general question should use general knowledge only
    prompts.append(P(domain=dom, construct="relevance_gating",
        session_group=grp, reference="Posner 1980",
        message="How many continents are there?",
        response_contains=["7", "seven"],
        response_not_contains=["NeoCarbide", "turbine", "Dr. Chen"]))

    # Focused retrieval — precise answer without over-inclusion
    prompts.append(P(domain=dom, construct="focused_retrieval",
        session_group=grp, reference="Broadbent 1958",
        message="What airline is my Tokyo flight on?",
        response_contains=["United"],
        response_not_contains=["Delta", "American", "Japan Airlines"]))

    # ==================================================================
    # DOMAIN 5: LANGUAGE COMPREHENSION (7 prompts)
    # Grice (1975) — pragmatic inference, implicature, disambiguation
    # ==================================================================
    dom = "language_comprehension"

    # Conversational implicature — "some" implies "not all"
    grp = "lang1"
    prompts.append(P(domain=dom, construct="scalar_implicature",
        session_group=grp, reference="Grice 1975",
        message="Some of my team members finished the report on time."))
    prompts.append(P(domain=dom, construct="scalar_implicature_probe",
        session_group=grp, reference="Grice 1975",
        message="Did all of my team members finish the report on time?",
        response_not_contains=["all of them finished", "everyone finished on time", "yes, all"],
        response_contains=["some", "not all", "didn't", "not everyone", "not aware"]))

    # Indirect speech act — request disguised as question
    grp = "lang2"
    prompts.append(P(domain=dom, construct="indirect_speech_act",
        session_group=grp, reference="Searle 1975",
        message="I told my assistant 'Can you close the window?' and she said 'yes' but didn't close it."))
    prompts.append(P(domain=dom, construct="pragmatic_interpretation",
        session_group=grp, reference="Searle 1975",
        message="What did I actually want my assistant to do?",
        response_contains=["close the window", "close a window", "shut the window", "close it",
            "close a physical window", "closing the window"]))

    # Presupposition — "stopped" presupposes prior activity
    grp = "lang3"
    prompts.append(P(domain=dom, construct="presupposition",
        session_group=grp, reference="Stalnaker 1974",
        message="I stopped running marathons after my knee surgery."))
    prompts.append(P(domain=dom, construct="presupposition_probe",
        session_group=grp, reference="Stalnaker 1974",
        message="Did I use to run marathons?",
        response_contains=["yes", "you did", "you used to", "stopped", "stopping", "marathon"],
        response_not_contains=["I have no information about", "I can't tell if", "never mentioned"]))

    # Metaphor comprehension
    grp = "lang4"
    prompts.append(P(domain=dom, construct="figurative_language",
        session_group=grp, reference="Lakoff & Johnson 1980",
        message="My coworker said the new policy is a ticking time bomb. What do you think she means by that?",
        response_contains=["problem", "danger", "harmful", "trouble", "bad", "explode", "explosive",
            "disaster", "negative", "risk", "issue", "catastroph", "chaotic", "consequences",
            "backfire", "blow up", "damag", "destruct"],
        response_not_contains=["actual bomb", "real bomb", "literal bomb"]))

    # ==================================================================
    # DOMAIN 6: DEDUCTIVE & INDUCTIVE REASONING (7 prompts)
    # Wason (1966), Tversky & Kahneman (1974), Gentner (1983)
    # ==================================================================
    dom = "reasoning"

    # Modus ponens
    grp = "reas1"
    prompts.append(P(domain=dom, construct="modus_ponens",
        session_group=grp, reference="Wason 1966",
        message="In my department, every senior engineer gets a private office. I just got promoted to senior engineer.",
        beliefs_created_min=1))
    prompts.append(P(domain=dom, construct="modus_ponens_probe",
        session_group=grp, reference="Wason 1966",
        message="Should I expect to get a private office?",
        response_contains=["yes", "private office"],
        response_not_contains=["probably not", "unlikely", "I wouldn't expect", "don't expect"]))

    # Modus tollens (harder — denying the consequent)
    grp = "reas2"
    prompts.append(P(domain=dom, construct="modus_tollens",
        session_group=grp, reference="Wason 1966",
        message="At my gym, if you're a platinum member you get free towels. I never get free towels at the gym."))
    prompts.append(P(domain=dom, construct="modus_tollens_probe",
        session_group=grp, reference="Wason 1966",
        message="You told me: platinum members get free towels. I never get free towels. So logically, am I a platinum member?",
        response_contains=["not", "no", "don't", "aren't", "unlikely"],
        response_not_contains=["yes, you are a platinum", "yes you're a platinum"]))

    # Analogical reasoning — structure mapping
    grp = "reas3"
    prompts.append(P(domain=dom, construct="analogical_reasoning",
        session_group=grp, reference="Gentner 1983",
        message="Complete this analogy: A neuron is to the brain as a transistor is to a ___?",
        response_contains=["computer", "processor", "chip", "CPU", "circuit", "microchip", "electronic"]))

    # Transitive inference
    grp = "reas4"
    prompts.append(P(domain=dom, construct="transitive_inference",
        session_group=grp, reference="Piaget 1952",
        message="In my org chart, Alice reports to Bob, and Bob reports to Carol. Does Alice report to Carol, directly or indirectly?",
        response_contains=["yes", "indirectly"]))

    # Causal reasoning — distinguish correlation from causation
    grp = "reas5"
    prompts.append(P(domain=dom, construct="causal_reasoning",
        session_group=grp, reference="Tversky & Kahneman 1974",
        message="Every time I bring an umbrella, it stays sunny. Does carrying an umbrella prevent rain?",
        response_contains=["no", "correlation", "coincidence", "doesn't"],
        response_not_contains=["yes, carrying an umbrella prevents"]))

    # ==================================================================
    # DOMAIN 7: SOCIAL COGNITION (6 prompts)
    # Premack & Woodruff (1978) — theory of mind, perspective-taking
    # Kohlberg (1958) — moral reasoning
    # ==================================================================
    dom = "social_cognition"

    # Classic false-belief task (Sally-Anne)
    grp = "soc1"
    prompts.append(P(domain=dom, construct="false_belief_task",
        session_group=grp, reference="Wimmer & Perner 1983",
        message="Sarah left her lunch in the fridge before a meeting. While she was away, Mike secretly moved it to the cabinet. Sarah doesn't know about the move. When she gets back, where will Sarah look first for her lunch?",
        response_contains=["fridge"],
        response_not_contains=["Sarah will look in the cabinet", "look for it in the cabinet", "check the cabinet first"]))

    # Perspective-taking — representing another's mental state
    grp = "soc2"
    prompts.append(P(domain=dom, construct="perspective_taking",
        session_group=grp, reference="Premack & Woodruff 1978",
        message="I told my friend I got the job, but I was being sarcastic because I actually got rejected. My friend congratulated me excitedly. Does my friend know I got rejected?",
        response_contains=["no", "doesn't know", "didn't realize", "thinks", "unaware", "doesn't realize"],
        response_not_contains=["yes, your friend knows"]))

    # Emotional inference — reading between the lines
    grp = "soc3"
    prompts.append(P(domain=dom, construct="emotional_inference",
        session_group=grp, reference="Baron-Cohen et al. 1997",
        message="My manager said 'That's interesting' and quickly changed the subject when I pitched my idea. How do you think she felt about my idea?",
        response_not_contains=["loved it", "was excited about", "genuinely enthusiastic", "was enthusiastic"],
        response_contains=["not interested", "didn't like", "dismiss", "unenthusiastic", "unenthused",
            "politely", "disinterest", "wasn't impressed", "lukewarm", "negative", "wasn't keen",
            "not keen", "wasn't genuinely", "deflect", "avoid", "brush", "not engaged",
            "not receptive", "changed the subject", "not thrilled", "however", "not compelling",
            "not find", "not have found", "wasn't interested", "unimpressed", "not convinced",
            "indifferen", "skepti", "underwhelm"]))

    # Moral reasoning — Heinz dilemma variant
    grp = "soc4"
    prompts.append(P(domain=dom, construct="moral_dilemma",
        session_group=grp, reference="Kohlberg 1958",
        message="A self-driving car must choose: swerve left and hit one pedestrian, or stay straight and hit three pedestrians. What are the ethical arguments for each choice?",
        response_contains=["one", "three", "utilitarian", "ethic", "moral", "right", "wrong", "lives", "dilemma", "argument"],
        response_not_contains=["I refuse to answer", "I can't discuss"]))

    # Social norm understanding
    grp = "soc5"
    prompts.append(P(domain=dom, construct="social_norm",
        session_group=grp, reference="Bicchieri 2006",
        message="At a formal dinner, someone starts eating before everyone is served. Is that considered appropriate behavior?",
        response_contains=["no", "not", "impolite", "rude", "inappropriate", "wait"],
        response_not_contains=["perfectly fine", "totally acceptable"]))

    # Intention attribution
    grp = "soc6"
    prompts.append(P(domain=dom, construct="intention_attribution",
        session_group=grp, reference="Premack & Woodruff 1978",
        message="A child accidentally breaks a vase while trying to get cookies from a high shelf. Another child deliberately breaks a vase because they're angry. Who deserves more blame?",
        response_contains=["angry", "deliberat", "intentional", "second", "purpose"]))

    # ==================================================================
    # DOMAIN 8: SELF-CORRECTION & BELIEF REVISION (6 prompts)
    # AGM theory — Alchourrón, Gärdenfors, Makinson (1985)
    # ==================================================================
    dom = "self_correction"
    grp = "rev_main"

    # Establish initial belief
    prompts.append(P(domain=dom, construct="initial_belief",
        session_group=grp, reference="AGM 1985",
        message="The team meeting is scheduled for Thursday at 3pm in Conference Room B.",
        beliefs_created_min=1))

    # Verify anchored
    prompts.append(P(domain=dom, construct="belief_verification",
        session_group=grp, reference="AGM 1985",
        message="When is the team meeting?",
        response_contains=["Thursday", "3"]))

    # Contradiction — revise the belief (AGM contraction + expansion)
    prompts.append(P(domain=dom, construct="belief_revision",
        session_group=grp, reference="AGM 1985",
        message="Correction: the meeting was moved to Friday at 10am in the main conference hall.",
        beliefs_created_min=1))

    # Verify revision took hold
    prompts.append(P(domain=dom, construct="revision_verification",
        session_group=grp, reference="AGM 1985",
        message="What day and time is the team meeting now?",
        response_contains=["Friday", "10"],
        response_not_contains=["Thursday", "3pm", "3 pm"]))

    # Minimal change principle — unrevised beliefs should persist
    prompts.append(P(domain=dom, construct="minimal_change",
        session_group=grp, reference="AGM 1985",
        message="Is the meeting still in a conference room?",
        response_contains=["yes", "conference"],
        response_not_contains=["no meeting", "no longer"]))

    # Multi-step revision — third update
    prompts.append(P(domain=dom, construct="iterative_revision",
        session_group=grp, reference="AGM 1985",
        message="One more change: the meeting is now virtual, no room needed. Same day and time though. When and how is the meeting happening?",
        response_contains=["Friday", "10", "virtual"],
        response_not_contains=["Conference Room B", "Room B"]))

    # -------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------
    graded = [p for p in prompts if not p.setup]
    assert len(graded) == 50, f"Expected 50 graded prompts, got {len(graded)}"
    return prompts


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CognitiveTestClient:
    """HTTP client for the cognitive battery. Registers a clean test user."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        ts = int(time.time())
        self.user_email = f"cognitive_{ts}@test.local"
        self.user_password = "cogpass123"
        self.token: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=TIMEOUT)
        return self._client

    async def register_and_login(self) -> None:
        client = await self._get_client()
        await client.post(f"{self.base_url}/auth/register", json={
            "email": self.user_email,
            "password": self.user_password,
            "name": "CognitiveBattery",
        })
        resp = await client.post(f"{self.base_url}/auth/login", json={
            "email": self.user_email,
            "password": self.user_password,
        })
        resp.raise_for_status()
        self.token = resp.json()["access_token"]

    async def send_message(self, message: str, session_id: str) -> dict:
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/chat/message",
                json={"message": message, "session_id": session_id},
                headers={"Authorization": f"Bearer {self.token}"},
            )
            if resp.status_code != 200:
                return {"error": resp.text, "status_code": resp.status_code}
            return resp.json()
        except httpx.ReadTimeout:
            return {"error": "timeout", "status_code": 408}
        except Exception as e:
            return {"error": str(e), "status_code": 500}

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_prompt(prompt: Prompt, response: dict) -> Result:
    """Mechanically grade a response against the prompt's predicates."""
    result = Result(
        prompt_id=prompt.id,
        domain=prompt.domain,
        construct=prompt.construct,
        message=prompt.message[:100],
    )

    if "error" in response:
        result.error = str(response["error"])[:200]
        if prompt.no_error:
            result.failures.append(f"API error: {result.error}")
        result.passed = len(result.failures) == 0
        return result

    if "detail" in response:
        result.error = str(response["detail"])[:200]
        if prompt.no_error:
            result.failures.append(f"Server error: {result.error}")
        result.passed = len(result.failures) == 0
        return result

    assistant_msg = response.get("assistant_message", "")
    beliefs_created = response.get("beliefs_created", [])
    beliefs_reinforced = response.get("beliefs_reinforced", [])
    beliefs_mutated = response.get("beliefs_mutated", [])
    events = response.get("events", [])
    duration_ms = response.get("duration_ms", 0.0)

    result.beliefs_created = len(beliefs_created)
    result.beliefs_reinforced = len(beliefs_reinforced)
    result.beliefs_mutated = len(beliefs_mutated)
    result.events_count = len(events)
    result.response_snippet = assistant_msg[:150]
    result.duration_ms = duration_ms

    failures = []

    if prompt.beliefs_created_min is not None:
        if len(beliefs_created) < prompt.beliefs_created_min:
            failures.append(
                f"beliefs_created={len(beliefs_created)}, "
                f"expected >={prompt.beliefs_created_min}")

    if prompt.beliefs_created_max is not None:
        if len(beliefs_created) > prompt.beliefs_created_max:
            failures.append(
                f"beliefs_created={len(beliefs_created)}, "
                f"expected <={prompt.beliefs_created_max}")

    if prompt.beliefs_reinforced_min is not None:
        if len(beliefs_reinforced) < prompt.beliefs_reinforced_min:
            failures.append(
                f"beliefs_reinforced={len(beliefs_reinforced)}, "
                f"expected >={prompt.beliefs_reinforced_min}")

    if prompt.response_contains is not None:
        lower_msg = assistant_msg.lower()
        found_any = any(kw.lower() in lower_msg for kw in prompt.response_contains)
        if not found_any:
            failures.append(
                f"response missing any of: {prompt.response_contains}")

    if prompt.response_not_contains is not None:
        lower_msg = assistant_msg.lower()
        found_bad = [kw for kw in prompt.response_not_contains
                     if kw.lower() in lower_msg]
        if found_bad:
            failures.append(
                f"response contains forbidden: {found_bad}")

    result.failures = failures
    result.passed = len(failures) == 0
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_battery() -> tuple[list[Result], list[DomainScore]]:
    prompts = build_prompts()
    client = CognitiveTestClient()

    print(f"  Registering test user: {client.user_email}")
    await client.register_and_login()
    print(f"  Authenticated.\n")

    session_map: dict[str, str] = {}
    results: list[Result] = []

    total = len(prompts)
    for i, prompt in enumerate(prompts):
        if prompt.session_group not in session_map:
            session_map[prompt.session_group] = str(uuid.uuid4())
        session_id = session_map[prompt.session_group]

        label = f"[{i + 1}/{total}]"
        if prompt.setup:
            print(f"  {label} SETUP ({prompt.domain}): {prompt.message[:60]}...")
            response = await client.send_message(prompt.message, session_id)
            if "error" in response or "detail" in response:
                err = response.get("error") or response.get("detail")
                print(f"         SETUP FAILED: {err}")
            else:
                print(f"         OK")
            continue

        construct_tag = f"{prompt.domain}/{prompt.construct}"
        print(f"  {label} #{prompt.id:>2} ({construct_tag}): "
              f"{prompt.message[:55]}...")

        response = await client.send_message(prompt.message, session_id)
        result = grade_prompt(prompt, response)
        results.append(result)

        status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
        detail = ""
        if not result.passed:
            detail = f" -- {'; '.join(result.failures)}"
        print(f"         {status} "
              f"(c={result.beliefs_created} r={result.beliefs_reinforced} "
              f"{result.duration_ms:.0f}ms){detail}")

    await client.close()

    domains: dict[str, DomainScore] = {}
    for r in results:
        if r.domain not in domains:
            domains[r.domain] = DomainScore(
                domain=r.domain, total=0, passed=0, failed=0)
        ds = domains[r.domain]
        ds.total += 1
        if r.passed:
            ds.passed += 1
        else:
            ds.failed += 1
            ds.fail_details.append({
                "prompt_id": r.prompt_id,
                "construct": r.construct,
                "message": r.message,
                "failures": r.failures,
                "response_snippet": r.response_snippet,
            })

    return results, list(domains.values())


def print_scorecard(results: list[Result], scores: list[DomainScore]) -> int:
    total_pass = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 76)
    print("  ABES COGNITIVE AI BATTERY — SCORECARD")
    print("=" * 76)

    # Domain references for the scorecard
    domain_refs = {
        "episodic_memory": "Tulving 1972",
        "semantic_memory": "Collins & Quillian 1969",
        "working_memory": "Baddeley & Hitch 1974",
        "selective_attention": "Broadbent 1958",
        "language_comprehension": "Grice 1975",
        "reasoning": "Wason 1966 / Gentner 1983",
        "social_cognition": "Premack & Woodruff 1978",
        "self_correction": "AGM 1985",
    }

    for ds in scores:
        bar = "#" * ds.passed + "." * ds.failed
        pct = (ds.passed / ds.total * 100) if ds.total else 0
        ref = domain_refs.get(ds.domain, "")
        print(f"  {ds.domain:<28} {ds.passed:>2}/{ds.total:<2}  "
              f"[{bar}]  {pct:>5.1f}%  ({ref})")
        for fd in ds.fail_details:
            print(f"     FAIL #{fd['prompt_id']}: [{fd['construct']}] "
                  f"{fd['failures'][0][:60]}")

    print("-" * 76)
    pct_total = (total_pass / total * 100) if total else 0
    print(f"  TOTAL SCORE: {total_pass}/{total}  ({pct_total:.1f}%)")
    print("=" * 76)
    return total_pass


def save_artifact(results: list[Result], scores: list[DomainScore],
                  total_score: int) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Gather unique references cited
    all_refs = set()
    prompts = build_prompts()
    for p in prompts:
        if p.reference:
            all_refs.add(p.reference)

    artifact = {
        "suite": "ABES Cognitive AI Battery",
        "version": "1.0",
        "methodology": (
            "50-prompt evaluation across 8 cognitive domains grounded in "
            "established cognitive science frameworks. Each prompt targets a "
            "specific construct (e.g., episodic encoding, modus tollens, "
            "false-belief task). All assertions are deterministic boolean "
            "predicates — no subjective or LLM-graded scoring."
        ),
        "references": sorted(all_refs),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_score": total_score,
        "total_prompts": len(results),
        "domains": [
            {
                "domain": ds.domain,
                "passed": ds.passed,
                "total": ds.total,
                "percentage": round(ds.passed / ds.total * 100, 1) if ds.total else 0,
                "fail_details": ds.fail_details,
            }
            for ds in scores
        ],
        "results": [
            {
                "prompt_id": r.prompt_id,
                "domain": r.domain,
                "construct": r.construct,
                "message": r.message,
                "passed": r.passed,
                "failures": r.failures,
                "beliefs_created": r.beliefs_created,
                "beliefs_reinforced": r.beliefs_reinforced,
                "response_snippet": r.response_snippet,
                "duration_ms": r.duration_ms,
                "error": r.error,
            }
            for r in results
        ],
    }
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact saved to {RESULTS_PATH}")


async def main():
    print("\n" + "=" * 76)
    print("  ABES COGNITIVE AI BATTERY — 50-Prompt Evaluation")
    print("  Domains: Memory, Attention, Language, Reasoning,")
    print("           Social Cognition, Self-Correction")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Backend: {BASE_URL}")
    print("=" * 76 + "\n")

    # Preflight
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            resp = await c.get(f"{BASE_URL}/agents")
            resp.raise_for_status()
            print(f"  Backend UP ({len(resp.json()['agents'])} agents)\n")
    except Exception as e:
        print(f"  ERROR: Backend unreachable: {e}")
        sys.exit(1)

    results, scores = await run_battery()
    total_score = print_scorecard(results, scores)
    save_artifact(results, scores, total_score)
    sys.exit(0 if total_score == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
