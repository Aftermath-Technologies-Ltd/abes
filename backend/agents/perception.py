# Author: Bradley R. Kinnard
"""
Perception Agent - extracts candidate beliefs from input text.
Filters noise, commands, and filler to surface factual claims.
"""

import re
from collections import OrderedDict
from typing import Callable

_IMPERATIVES = frozenset(
    {
        "check",
        "fix",
        "show",
        "tell",
        "give",
        "make",
        "run",
        "stop",
        "start",
        "restart",
        "verify",
        "confirm",
        "look",
        "see",
        "investigate",
        "update",
        "change",
        "remove",
        "add",
        "delete",
        "create",
        "find",
        "test",
        "get",
        "set",
        "explain",
        "debug",
        "inspect",
    }
)

_MODALS = frozenset({"should", "must", "need", "needs", "ought", "supposed"})
_ADVERBS = frozenset({"really", "probably", "definitely", "maybe", "certainly", "just"})
_FILLERS = frozenset(
    {"hi", "hello", "hey", "thanks", "ok", "okay", "yes", "no", "sure", "yep", "nope"}
)

_STATE_VERBS = frozenset(
    {
        "is",
        "was",
        "are",
        "were",
        "has",
        "have",
        "had",
        "been",
        "be",
    }
)

# Outcome verbs - both base and past forms for negation matching
_OUTCOME_VERBS = frozenset(
    {
        "succeed",
        "succeeded",
        "complete",
        "completed",
        "finish",
        "finished",
        "start",
        "started",
        "stop",
        "stopped",
        "load",
        "loaded",
        "save",
        "saved",
        "update",
        "updated",
        "sync",
        "synced",
        "converge",
        "converged",
        "store",
        "stored",
        "initialize",
        "initialized",
        "terminate",
        "terminated",
        "reset",
        "clear",
        "cleared",
        "flush",
        "flushed",
    }
)

# System state indicators that always qualify
_CRITICAL = frozenset(
    {
        "oom",
        "timeout",
        "retrying",
        "deadlock",
        "stall",
        "freeze",
        "crashed",
        "failed",
        "corrupt",
        "mismatch",
        "diverged",
        "spiking",
        "stalled",
        "leaked",
        "overflow",
        "underflow",
    }
)

# Technical vocabulary - needs supporting context
_DOMAIN = frozenset(
    {
        "belief",
        "beliefs",
        "weights",
        "cache",
        "tensor",
        "model",
        "training",
        "iteration",
        "epoch",
        "loss",
        "gradient",
        "memory",
        "gpu",
        "cpu",
        "reload",
        "update",
        "sync",
        "cluster",
        "agent",
        "snapshot",
        "decay",
        "config",
        "parameter",
        "vector",
        "embedding",
        "index",
        "queue",
        "batch",
    }
)

_ERROR_PATTERN = re.compile(
    r"\b(failed|corrupt|broken|missing|error|spike|stalled|timeout|crashed|leak|mismatch|diverged)\b"
)

_FAILURE_PATTERN = re.compile(
    r"(failed|error|crash|timeout|exceed|spike|corrupt|stall|threshold|limit|reached)",
    re.I,
)

# Filler phrases that disqualify trailing clauses
_FILLER_PHRASES = re.compile(
    r"^(or\s+)?(you know|whatever|anyway|idk|i guess|like|basically|honestly)\b", re.I
)

_LOG_PREFIXES = [
    re.compile(r"^\[\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2}Z?\]\s*"),
    re.compile(r"^\[(INFO|WARN|WARNING|ERROR|DEBUG|TRACE)(:\w+)?\]\s*", re.I),
    re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2}Z?\s*"),
    re.compile(r"^\d{2}:\d{2}:\d{2}\s+"),
    re.compile(r"^(INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\s*[-:|]?\s*", re.I),
    re.compile(r"^\|\s*"),
]

# Split protection into focused patterns - easier to debug than one mega-regex
_PROTECT_URL = re.compile(r"https?://\S+|www\.\S+")
_PROTECT_FILENAME = re.compile(r"\w+\.(?:py|js|json|yaml|log|txt|md|cfg)\b")
_PROTECT_VERSION = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b|\b[a-zA-Z]+\d+\.\d+\b")
# Protect name abbreviations and titles: Mr., Mrs., Dr., Jr., Sr., single initials like "R."
_PROTECT_ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Jr|Sr|Prof|Rev|Gen|Col|Lt|Sgt|Capt|Hon)\.|"  # Titles
    r"\b[A-Z]\."  # Single letter initials (R., J., etc.)
)


class PerceptionAgent:
    """Extracts candidate belief strings from user messages, tool outputs, or events."""

    def __init__(self, cache_size: int = 500):
        self._seen: OrderedDict[str, bool] = OrderedDict()
        self._cache_size = cache_size

        # Claim extraction rules: (pattern, extractor, purpose)
        self._claim_rules: list[tuple[re.Pattern, Callable[[re.Match], str], str]] = [
            (
                re.compile(r"\bthat\s+(.{6,}?)(?:[.!?]|$)", re.I),
                lambda m: m.group(1).strip().rstrip(".!?"),
                "that-clause",
            ),
            (
                re.compile(
                    r"\b(?:explain|check|verify)\s+(?:why|how|what)\s+([^.!?]+)", re.I
                ),
                lambda m: m.group(1).strip(),
                "interrogative",
            ),
            (
                re.compile(r"\b(?:whether|if)\s+([^.!?]{5,})", re.I),
                lambda m: m.group(1).strip(),
                "conditional",
            ),
            (
                re.compile(r"\b(?:because|since)\s+([^.!?]{6,})", re.I),
                lambda m: m.group(1).strip(),
                "causal",
            ),
            (
                re.compile(r"[,;]\s+(.{6,})"),
                lambda m: m.group(1).strip().rstrip(".!?"),
                "trailing",
            ),
        ]

    async def ingest(self, raw_input: str, context: dict) -> list[str]:
        """Route to appropriate extractor based on input source."""
        if not raw_input or not raw_input.strip():
            return []

        source = context.get("source_type", "chat")
        if source in ("tool", "system", "log"):
            return self._from_structured(raw_input)
        return self._from_chat(raw_input)

    def _from_chat(self, text: str) -> list[str]:
        """Extract beliefs from conversational input."""
        out = []
        # Track duplicates within THIS call only (not across users/requests)
        seen_this_call: set[str] = set()
        for sent in self._split_sentences(text):
            sent = sent.strip()
            lower = sent.lower()

            if len(sent) < 4 and lower not in _CRITICAL:
                continue

            if self._is_filler(lower):
                continue

            if self._is_command(lower):
                claim = self._extract_claim(sent)
                if claim:
                    sent = claim
                else:
                    continue

            # For chat, accept any substantive factual statement
            if not self._has_chat_substance(sent):
                continue

            key = sent.lower()
            if key in seen_this_call:
                continue
            seen_this_call.add(key)

            out.append(sent)
        return out

    def _has_chat_substance(self, text: str) -> bool:
        """Check if text has substance for conversational context.

        More permissive than _has_substance - accepts personal facts,
        preferences, opinions, and any declarative statements.
        """
        lower = text.lower()
        words = lower.split()

        # Too short
        if len(words) < 3:
            return False

        # Questions aren't beliefs
        if text.strip().endswith("?"):
            return False

        # Reject repetitive gibberish (same word repeated 3+ times)
        unique_words = set(w for w in words if len(w) > 1)
        if len(unique_words) <= 1 and len(words) > 3:
            return False

        # Reject SQL/code injection attempts
        code_patterns = [
            r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\s+(FROM|INTO|TABLE|INDEX)\b",
            r"<\s*script\b",
            r"--\s*$",
            r"\b(eval|exec|import\s+os|subprocess)\b",
        ]
        for pat in code_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return False

        # Vague filler phrases that aren't beliefs
        filler_phrases = [
            r"^hmm\b",
            r"^(let me|lemme)\s+(think|see|check|consider)",
            r"^(hang on|hold on|wait)\b",
            r"^(sure|fine|great|good|nice|cool|awesome|perfect|right|alright)\s*(,|!|\.)?$",
            r"^(got it|i see|makes sense|fair enough|sounds? good|no worries|never mind)",
            r"^(moving on|anyway|anyhow|so\b)",
        ]
        for pat in filler_phrases:
            if re.search(pat, lower):
                return False

        # Conversational directives — requests to change topic, not facts about the user
        directive_patterns = [
            r"^let['\u2019]?s\s+(talk|discuss|chat|move|switch|go|get|start)",
            r"^(tell|teach|explain|describe|show)\s+me\s+(about|how|what|why)",
            r"^(can|could|would)\s+(you|we)\s+(talk|discuss|explain|tell)",
            r"^(i['\u2019]?d\s+like|i\s+want)\s+to\s+(talk|discuss|hear|learn|know)\b",
            r"^(how\s+does?|how\s+do|what\s+is|what\s+are|why\s+does?)\b",
        ]
        for pat in directive_patterns:
            if re.search(pat, lower):
                return False

        # Pure filler
        filler_only = {"um", "uh", "hmm", "huh", "oh", "ah", "yeah", "yep", "nope", "ok", "okay"}
        if set(words) <= filler_only:
            return False

        # Filter out meta-comments about the conversation itself
        # These are complaints/statements about the AI, not facts about the user
        meta_patterns = [
            r"\byou\s+(forgot|missed|didn't|don't|should|shouldn't|failed)\b",
            r"\bwhy\s+didn't\s+you\b",
            r"\bthis\s+conversation\b",
            r"\bmemory\s+(issue|problem|trouble|lapse)\b",
            r"\byou('re|r| are)\s+(having|experiencing|denying|gaslighting)\b",
            r"\byou\s+know\s+(my|this|that|about)\b",  # "you know my dogs" - directive not fact
            r"\byou\s+already\s+know\b",
            r"\bwhat\s+caused\s+you\b",
            r"\bplease\s+(tell|show|explain|remember)\b",
            r"\bprove\s+(your|it|this|that)\b",  # "prove your response" - directive
            r"\b(show|tell|explain)\s+(me|us)\b",  # "show me", "tell me" - directives
            r"\bwhat\s+do\s+you\s+know\b",  # questions, not facts
            r"\bwho\s+am\s+i\b",  # questions
            r"\bdo\s+you\s+(know|remember|recall)\b",  # questions
        ]
        for pat in meta_patterns:
            if re.search(pat, lower):
                return False

        # Personal fact patterns - "I am", "I have", "My X is", "I like", etc.
        # Allow optional adverbs (recently, just, also, always, never, etc.) between "I" and verb
        _ADV = r"(?:recently|just|also|always|never|still|already|finally|actually|currently|usually|often|sometimes|definitely|rarely)?\s*"
        personal_patterns = [
            r"\bmy\s+\w+\s+(is|are|was|were)\b",  # My name is, My dogs are
            rf"\bi\s+{_ADV}(am|was|have|had|love|like|prefer|enjoy|hate|dislike|want|need|work|play|live|study|teach|drive|read|write|speak|sing|cook|travel|run|manage|own|build|design|switched|graduated|moved|find|found|collect|picked|bought|sold|adopted|started|broke|finished|got|made|met|joined|left|quit|learned|tried)\b",
            r"\bi['\u2019]m\b",  # I'm
            r"\bi['\u2019]ve\b",  # I've
            r"\b(he|she|it|they)\s+(is|are|was|were|has|have)\b",
            r"\b(his|her|their|its)\s+\w+\s+(is|are)\b",
            r"\bits\s+\w+\b",  # "its cold" - informal contraction without apostrophe
            r"\b(it['\u2019]s)\s+\w+\b",  # "it's cold"
        ]
        for pat in personal_patterns:
            if re.search(pat, lower):
                return True

        # Declarative with state verb - "X is Y", "X has Y"
        if bool(_STATE_VERBS & set(words)):
            # Has a subject and predicate (at least 3 words with state verb not at start)
            for i, w in enumerate(words):
                if w in _STATE_VERBS and i > 0:
                    return True

        # Opinion/preference markers
        opinion_markers = {"think", "believe", "feel", "prefer", "love", "hate", "like", "dislike", "enjoy", "favorite", "best", "worst"}
        if opinion_markers & set(words):
            return True

        # Named entities (capitalized words not at sentence start)
        if any(w[0].isupper() for w in text.split()[1:] if w and w[0].isalpha()):
            return True

        # Fall back to technical substance check
        return self._has_substance(text)

    def _from_structured(self, text: str) -> list[str]:
        """Extract from logs/tool output with diversity-aware repeat limits."""
        out = []
        seen: dict[str, int] = {}
        unique_failures: set[str] = set()
        max_repeats_per_failure = 5

        for line in text.strip().split("\n"):
            line = self._strip_log_prefix(line.strip())
            if len(line) < 3:
                continue

            if line in ("{", "}", "[", "]", "null", "---", "..."):
                continue

            if line.startswith("File ") and ", line " in line:
                continue

            if m := re.match(r"^(\w+Error|\w+Exception):\s+(.+)", line):
                line = m.group(2)
            elif m := re.match(
                r"^Caused by:\s*(?:\w+Error|\w+Exception)?:?\s*(.+)", line
            ):
                line = m.group(1)

            if line.startswith("During handling") or line.startswith(
                "The above exception"
            ):
                continue

            # skip progress bars like "epoch 3/10 done"
            if re.match(r"^\w+\s+\d+/\d+\s+\w+", line):
                continue

            lower = line.lower()
            has_failure = bool(_FAILURE_PATTERN.search(lower))

            # skip timing/progress noise unless it contains a failure
            if not has_failure:
                if re.search(r"epoch\s+\d+|batch\s+\d+|\b\d+\s*(ms|s|sec|min)\b|completed in \d+", lower):
                    continue

            is_procedural = bool(
                re.match(
                    r"^(done|finished|retry|attempt|start|starting|stop|stopping)\b",
                    lower,
                )
            )

            if is_procedural and not has_failure:
                continue

            key = lower

            # Diversity-aware limits: unlimited distinct failures, capped repeats
            if has_failure:
                unique_failures.add(key)
                max_repeats = max_repeats_per_failure
            elif is_procedural:
                max_repeats = 1
            else:
                max_repeats = 3

            count = seen.get(key, 0)
            if count >= max_repeats:
                continue
            seen[key] = count + 1
            out.append(line)

        return out

    def _split_sentences(self, text: str) -> list[str]:
        """Split on sentence boundaries, preserving technical tokens.

        For chat, also splits on commas separating independent clauses
        (e.g., "my name is Brad, I like Python" -> 2 beliefs).
        """
        protected: list[tuple[str, str]] = []

        def _protect(m: re.Match[str]) -> str:
            tok = f"__P{len(protected)}__"
            protected.append((tok, m.group(0)))
            return tok

        # protect tokens that contain dots but aren't sentence breaks
        text = _PROTECT_URL.sub(_protect, text)
        text = _PROTECT_FILENAME.sub(_protect, text)
        text = _PROTECT_VERSION.sub(_protect, text)
        text = _PROTECT_ABBREVIATIONS.sub(_protect, text)  # protect Mr., R., etc.
        text = re.sub(r"\.{3}", _protect, text)  # protect ellipses before collapse

        text = re.sub(r"([.!?])\1+", r"\1", text)  # collapse repeated punct

        # Split on punct+space+letter OR punct+letter (handles "completed.Next" and "failed.retrying")
        parts = re.split(r"[.!?](?=\s+[A-Za-z0-9])|[.!?](?=[A-Za-z])|\n+", text)

        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # restore protected tokens
            for tok, orig in protected:
                p = p.replace(tok, orig)

            # Split on comma + pronoun patterns (independent clauses)
            # "my name is Brad, I like Python" -> ["my name is Brad", "I like Python"]
            comma_parts = re.split(r",\s*(?=(?:i|he|she|they|we|it|my|his|her|their|our|you)\s)", p, flags=re.I)
            for cp in comma_parts:
                cp = cp.strip()
                if cp:
                    out.append(cp)
        return out

    def _strip_log_prefix(self, line: str) -> str:
        """Strip timestamp/level prefixes until stable."""
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        while True:
            prev = line
            for pat in _LOG_PREFIXES:
                line = pat.sub("", line).strip()
            if line == prev:
                break
        return line

    def _is_filler(self, lower: str) -> bool:
        """Reject greetings and acknowledgments."""
        if lower in _FILLERS:
            return True
        return bool(
            re.match(
                r"^(ok|okay|yes|no|sure)\s+(that\s+)?(just|maybe|idk|not really|but|anyway|makes sense|correct|right|got it|understood|exactly|absolutely)",
                lower,
            )
        )

    def _is_command(self, lower: str) -> bool:
        """Detect action requests. Factual statements pass through."""
        words = lower.split()
        if not words:
            return False

        for prefix in ("please ", "can you ", "could you ", "would you ", "let's ", "lets "):
            if lower.startswith(prefix):
                return True

        if lower.startswith("we need to ") or lower.startswith("you need to "):
            return True

        w0, w1, w2, w3 = (words + [None] * 4)[:4]

        # Direct imperative
        if w0 in _IMPERATIVES:
            return True

        # Pronoun-led commands
        if w0 in ("you", "we"):
            if w1 in _IMPERATIVES:
                return True
            if w1 in _MODALS and w2 in _IMPERATIVES:
                return True
            if w1 in _ADVERBS and w2 in _IMPERATIVES:
                return True
            if w1 in _ADVERBS and w2 in _MODALS and w3 in _IMPERATIVES:
                return True

        # "should check" is a command, but "should update automatically" is factual
        # Only treat modal+imperative as command when imperative is second word
        if w0 in _MODALS and w1 in _IMPERATIVES:
            return True

        # Hedged commands: "you might/may want to X", "you/we probably/maybe should X"
        if w0 in ("you", "we"):
            # "you might want to check", "you may want to fix"
            if (
                w1 in ("might", "may")
                and w2 == "want"
                and len(words) > 4
                and words[4] in _IMPERATIVES
            ):
                return True
            # "you probably should check", "we maybe should fix"
            if w1 in ("probably", "maybe") and w2 in _MODALS and w3 in _IMPERATIVES:
                return True
            # "you might need to restart", "we may need to update"
            if (
                w1 in ("might", "may")
                and w2 == "need"
                and len(words) > 4
                and words[4] in _IMPERATIVES
            ):
                return True
            # "you might possibly need to restart", "we may actually need to update"
            if (
                w1 in ("might", "may")
                and w2 in _ADVERBS
                and w3 == "need"
                and len(words) > 5
                and words[5] in _IMPERATIVES
            ):
                return True

        return False

    def _extract_claim(self, text: str) -> str | None:
        """Extract embedded factual claim from command."""
        for pattern, extractor, purpose in self._claim_rules:
            if m := pattern.search(text):
                result = extractor(m)
                # Filter garbage from trailing clauses
                if purpose == "trailing":
                    if self._is_command(result.lower()):
                        continue
                    if _FILLER_PHRASES.match(result):
                        continue
                    if len(result.split()) < 3:
                        continue
                return result
        return None

    def _has_substance(self, text: str) -> bool:
        """Verify text carries semantic weight worth tracking."""
        lower = text.lower()
        words = lower.split()

        # Critical signals always qualify
        if lower in _CRITICAL:
            return True
        for sig in _CRITICAL:
            if re.search(rf"\b{re.escape(sig)}\b", lower):
                return True

        if _ERROR_PATTERN.search(lower):
            return True

        # Negation detection: negated domain/outcome words always qualify
        negation_patterns = [
            r"\b(?:did|does|do|is|was|are|were|has|have|had)\s*n[o']?t\s+(\w+)",
            r"\b(?:didn|doesn|don|isn|wasn|aren|weren|hasn|haven|hadn)'t\s+(\w+)",
            r"\bnever\s+(\w+)",
        ]
        for pat in negation_patterns:
            if m := re.search(pat, lower):
                word = m.group(1)
                # Check word directly and common verb stems
                candidates = [word]
                if word.endswith("ing"):
                    base = word[:-3]
                    candidates.extend(
                        [base, base + "e"]
                    )  # loading->load, converging->converge
                if word.endswith("ed"):
                    base = word[:-2]
                    candidates.extend([base, base[:-1] if base.endswith("e") else base])
                for candidate in candidates:
                    if candidate in _DOMAIN or candidate in _OUTCOME_VERBS:
                        return True

        # Technical markers
        has_camel = any(
            text[i].isupper() and text[i - 1].islower() for i in range(1, len(text))
        )
        has_tech = (
            has_camel or "_" in text or any(c.isdigit() for c in text) or "/" in text
        )

        # State indicators
        has_state = bool(_STATE_VERBS & set(words))
        has_outcome = bool(_OUTCOME_VERBS & set(words))
        has_modal = bool(_MODALS & set(words))

        # Domain words qualify with state/outcome/tech context
        if _DOMAIN & set(words):
            if has_state or has_outcome or has_tech:
                return True
            # Modal only qualifies when state, outcome, or tech also present
            if has_modal and (has_state or has_outcome or has_tech):
                return True
            return False

        # Single words need tech markers
        if len(words) == 1:
            return (
                any(c.isupper() for c in text[1:])
                or any(c.isdigit() for c in text)
                or "_" in text
            )

        return has_tech


__all__ = ["PerceptionAgent"]
