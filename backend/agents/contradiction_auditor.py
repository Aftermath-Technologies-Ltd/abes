# Author: Bradley R. Kinnard
"""
ContradictionAuditorAgent - detects high-tension belief pairs.
Uses semantic rule-based detection with fallback to embedding heuristics.
"""

import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple
from uuid import UUID

import numpy as np

from ..core.config import settings
from ..core.models.belief import Belief
from ..core.bel.semantic_contradiction import (
    check_contradiction,
    ContradictionResult,
)

logger = logging.getLogger(__name__)


# tags that lower the threshold - more aggressive detection for high-stakes beliefs
# tradeoff: hardcoded values, but easy to tune if we find better thresholds empirically
SENSITIVE_TAGS: Dict[str, float] = {
    "critical": 0.4,
    "safety": 0.5,
    "core_value": 0.5,
}

# LRU cap - 2000 is plenty for typical workloads
EMBEDDING_CACHE_MAX = 2000

# keeps the O(n^2) pairwise check from killing us
MAX_BELIEFS_PER_AUDIT = 500

# above this count, switch from full O(n^2) to neighborhood-only auditing
NEIGHBORHOOD_AUDIT_THRESHOLD = 100

# how many neighbors to check per belief in neighborhood mode
NEIGHBORHOOD_K = 20

# cap tension so downstream doesn't see crazy values
MAX_TENSION_VALUE = 10.0

# safety limit on pairwise comparisons - derived from MAX_BELIEFS_PER_AUDIT
# at 500 beliefs we get 500*499/2 = 124,750 pairs, so 5M is generous headroom
MAX_PAIRWISE_COMPARISONS = 5_000_000


def _stable_hash(content: str) -> str:
    """SHA256 hash for cache invalidation. Deterministic across runs unlike hash()."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """Extract numbers with their context (unit hints).

    Returns list of (number, context) tuples.
    Example: "40 degrees" -> [(40.0, "degrees")]
    """
    # Match numbers with optional units
    pattern = r'(\d+(?:\.\d+)?)\s*(%|degrees?|°|F|C|mph|dollars?|\$|minutes?|hours?|days?|years?|feet|ft|miles?|lbs?|kg)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for num_str, unit in matches:
        try:
            num = float(num_str)
            results.append((num, unit.lower() if unit else ""))
        except ValueError:
            pass
    return results


def _has_numeric_conflict(text1: str, text2: str, similarity: float) -> bool:
    """Detect numeric contradictions in semantically similar statements.

    If two statements are about the same topic (high embedding similarity)
    but have significantly different numbers, they likely contradict.

    Example: "it was 40 degrees" vs "it was 70 degrees" - same topic, different values.
    """
    if similarity < 0.5:  # not similar enough to be about the same thing
        return False

    nums1 = _extract_numbers(text1)
    nums2 = _extract_numbers(text2)

    if not nums1 or not nums2:
        return False

    # Check if any numbers with same/similar units differ significantly
    for n1, u1 in nums1:
        for n2, u2 in nums2:
            # Units should match (or both be empty)
            if u1 != u2 and u1 and u2:
                continue

            # Check if numbers differ significantly (>20% difference)
            if n1 == 0 and n2 == 0:
                continue
            max_val = max(abs(n1), abs(n2))
            if max_val > 0:
                diff_pct = abs(n1 - n2) / max_val
                if diff_pct > 0.2:  # more than 20% different
                    return True

    return False


def _contains_word(text: str, word: str) -> bool:
    # simple word boundary check - escape in case word has regex chars
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _is_likely_negation(text1: str, text2: str) -> bool:
    """Cheap negation heuristic. Catches "X" vs "not X" style contradictions.

    Tradeoff: simple and fast, but will miss subtle stuff like "I prefer coffee"
    vs "tea is my favorite". Good enough for a first pass - we can layer LLM
    scoring later if needed.

    Caller is responsible for gating on semantic similarity before invoking.
    """
    t1 = text1.lower()
    t2 = text2.lower()

    negations = ["not", "no", "never", "don't", "doesn't", "didn't",
                 "isn't", "aren't", "wasn't", "weren't", "won't"]

    neg1 = sum(1 for neg in negations if f" {neg} " in f" {t1} ")
    neg2 = sum(1 for neg in negations if f" {neg} " in f" {t2} ")

    # one has negation, other doesn't = likely contradiction
    if (neg1 > 0) != (neg2 > 0):
        return True

    # Common antonym pairs for detecting semantic contradictions
    opposites = [
        # Boolean/logical
        ("true", "false"), ("yes", "no"), ("always", "never"),
        # Sentiment
        ("good", "bad"), ("like", "dislike"), ("love", "hate"),
        # Temperature
        ("hot", "cold"), ("warm", "cold"), ("warm", "cool"), ("hot", "cool"),
        # Light
        ("bright", "dark"), ("light", "dark"), ("sunny", "dark"),
        ("shining", "dark"), ("shiny", "dark"), ("bright", "dim"),
        # Weather
        ("sunny", "cloudy"), ("clear", "cloudy"), ("dry", "wet"), ("rainy", "sunny"),
        # Size/quantity
        ("big", "small"), ("large", "small"), ("many", "few"), ("more", "less"),
        # Speed
        ("fast", "slow"), ("quick", "slow"),
        # Position
        ("up", "down"), ("high", "low"), ("left", "right"), ("in", "out"),
        # State
        ("open", "closed"), ("on", "off"), ("alive", "dead"),
        # Quality
        ("new", "old"), ("young", "old"), ("clean", "dirty"),
        # Emotion
        ("happy", "sad"), ("joy", "sorrow"), ("calm", "angry"),
        # Agreement
        ("agree", "disagree"), ("accept", "reject"), ("include", "exclude"),
    ]

    for w1, w2 in opposites:
        if (_contains_word(t1, w1) and _contains_word(t2, w2)) or \
           (_contains_word(t1, w2) and _contains_word(t2, w1)):
            return True

    return False


@dataclass
class ContradictionDetectedEvent:
    """Fired when belief tension crosses threshold."""

    belief_id: UUID
    tension: float
    threshold: float
    timestamp: datetime
    # display context
    contradicting_belief_id: UUID = None
    belief_content: str = ""
    contradicting_content: str = ""
    similarity_score: float = 0.0
    # semantic detection fields
    reason_codes: list = field(default_factory=list)
    semantic_confidence: float = 0.0
    fallback_used: bool = False


class ContradictionAuditorAgent:
    """Scans beliefs for high tension and emits events."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        # debounce: track which beliefs were already flagged
        self._above_threshold: Set[UUID] = set()
        # LRU cache: (content_hash, embedding) - hash is SHA256 for stability
        self._embedding_cache: OrderedDict[UUID, Tuple[str, list]] = OrderedDict()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_threshold_for_belief(self, belief: Belief) -> float:
        """Threshold can be lowered by sensitive tags. Takes min of all matches."""
        base = settings.tension_threshold_high
        thresholds = [base]

        for tag in belief.tags:
            tag_lower = tag.lower()
            for sensitive_key, sensitive_threshold in SENSITIVE_TAGS.items():
                # match "safety" or "safety.high" etc
                if tag_lower == sensitive_key or tag_lower.startswith(sensitive_key + "."):
                    thresholds.append(sensitive_threshold)
                    break

        return min(thresholds)

    def _cache_embeddings(self, beliefs: List[Belief]) -> None:
        """Encode beliefs we haven't seen (or whose content changed)."""
        model = self._get_model()
        to_encode = []
        to_encode_ids = []

        for b in beliefs:
            content_hash = _stable_hash(b.content)

            if b.id in self._embedding_cache:
                cached_hash, _ = self._embedding_cache[b.id]
                if cached_hash == content_hash:
                    self._embedding_cache.move_to_end(b.id)  # bump LRU
                    continue
                # stale - content changed

            to_encode.append(b.content)
            to_encode_ids.append((b.id, content_hash))

        if to_encode:
            embs = model.encode(to_encode, convert_to_numpy=True)
            for i, (bid, chash) in enumerate(to_encode_ids):
                self._embedding_cache[bid] = (chash, embs[i].tolist())

            # evict oldest
            while len(self._embedding_cache) > EMBEDDING_CACHE_MAX:
                self._embedding_cache.popitem(last=False)

    def _compute_tensions_from_cache(
        self, beliefs: List[Belief], similarity_threshold: float = 0.25
    ) -> Tuple[dict[UUID, float], dict[UUID, Tuple[UUID, str, float, ContradictionResult]]]:
        """Pairwise tension via cached embeddings and semantic analysis.

        For small sets (<NEIGHBORHOOD_AUDIT_THRESHOLD), does full O(n^2).
        For larger sets, uses neighborhood-based auditing: each belief is only
        checked against its K nearest semantic neighbors, bringing complexity
        from O(n^2) to O(n*K).

        The similarity threshold is intentionally low (0.25) to catch implicit
        contradictions (e.g. "I'm vegetarian" vs "my favorite food is steak")
        that have low embedding similarity but high NLI contradiction scores.
        The semantic contradiction detector and NLI model filter false positives.

        Returns:
            tension_scores: dict mapping belief_id to total tension
            top_contradictors: dict mapping belief_id to
                (contradicting_id, content, similarity, semantic_result)
        """
        if not beliefs:
            return {}, {}

        # sort by UUID hex for deterministic pairwise ordering
        beliefs = sorted(beliefs, key=lambda b: str(b.id))

        # re-encode any beliefs that got evicted from cache
        missing = [b for b in beliefs if b.id not in self._embedding_cache]
        if missing:
            self._cache_embeddings(missing)

        embeddings = []
        for b in beliefs:
            _, emb = self._embedding_cache[b.id]
            embeddings.append(np.array(emb))

        tension_scores: dict[UUID, float] = {b.id: 0.0 for b in beliefs}
        top_contradictors: dict[UUID, Tuple[UUID, str, float, ContradictionResult]] = {}

        n = len(beliefs)

        # generate candidate pairs based on set size
        if n > NEIGHBORHOOD_AUDIT_THRESHOLD:
            pairs = self._neighborhood_pairs(beliefs, embeddings, similarity_threshold)
            logger.info("neighborhood audit: %d beliefs, %d candidate pairs", n, len(pairs))
        else:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if len(pairs) > MAX_PAIRWISE_COMPARISONS:
            logger.warning("pairs (%d) exceeds limit, truncating", len(pairs))
            pairs = pairs[:MAX_PAIRWISE_COMPARISONS]

        for i, j in pairs:
            b1, b2 = beliefs[i], beliefs[j]

            denom = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            if denom == 0:
                continue
            similarity = float(np.dot(embeddings[i], embeddings[j]) / denom)

            if similarity > similarity_threshold:
                result = check_contradiction(b1.content, b2.content)
                is_contradiction = result.label == "contradiction"

                if is_contradiction:
                    avg_confidence = (b1.confidence + b2.confidence) / 2.0
                    opposition = result.confidence

                    # for NLI-detected contradictions, the model is the authority —
                    # don't let low embedding similarity drag down the tension
                    nli_driven = "NLI_MODEL" in result.reason_codes
                    if nli_driven and opposition >= 0.8:
                        # NLI high-confidence: tension directly from semantic confidence
                        tension_val = avg_confidence * opposition
                    else:
                        # rule-based: similarity-weighted as before
                        tension_val = similarity * avg_confidence * (0.5 + 0.5 * opposition)
                    tension_scores[b1.id] += tension_val
                    tension_scores[b2.id] += tension_val

                    # populate graph edges on the beliefs themselves
                    b1.add_link(b2.id, "contradicts", weight=tension_val)
                    b2.add_link(b1.id, "contradicts", weight=tension_val)

                    # track top contradictor by semantic confidence
                    current_b1 = top_contradictors.get(b1.id)
                    if current_b1 is None or result.confidence > current_b1[3].confidence:
                        top_contradictors[b1.id] = (b2.id, b2.content, similarity, result)

                    current_b2 = top_contradictors.get(b2.id)
                    if current_b2 is None or result.confidence > current_b2[3].confidence:
                        top_contradictors[b2.id] = (b1.id, b1.content, similarity, result)

        # clamp
        for bid in tension_scores:
            if tension_scores[bid] > MAX_TENSION_VALUE:
                tension_scores[bid] = MAX_TENSION_VALUE

        return tension_scores, top_contradictors

    def _neighborhood_pairs(
        self,
        beliefs: List[Belief],
        embeddings: list,
        similarity_threshold: float,
    ) -> list[Tuple[int, int]]:
        """
        Find candidate pairs using semantic neighborhood.
        For each belief, compute cosine similarity against all others via matrix ops,
        then keep top-K neighbors above the similarity threshold.
        Returns deduplicated list of (i, j) index pairs where i < j.
        """
        emb_matrix = np.stack(embeddings)
        # normalize for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = emb_matrix / norms

        # full cosine similarity matrix via single matmul
        sim_matrix = normed @ normed.T

        n = len(beliefs)
        k = min(NEIGHBORHOOD_K, n - 1)
        pairs: set[Tuple[int, int]] = set()

        for i in range(n):
            row = sim_matrix[i].copy()
            row[i] = -1.0  # exclude self
            top_k_idx = np.argpartition(row, -k)[-k:]
            for j_val in top_k_idx:
                j = int(j_val)
                if row[j] >= similarity_threshold:
                    pair = (min(i, j), max(i, j))
                    pairs.add(pair)

        return sorted(pairs)

    async def _load_persisted_state(self, store) -> None:
        """Try loading debounce state from store."""
        if not hasattr(store, "load_contradiction_state"):
            return
        try:
            loaded = await store.load_contradiction_state()
            if loaded:
                self._above_threshold = set(loaded)
        except Exception:
            pass  # memory fallback is fine

    async def _save_persisted_state(self, store) -> None:
        if not hasattr(store, "save_contradiction_state"):
            return
        try:
            await store.save_contradiction_state(list(self._above_threshold))
        except Exception:
            pass  # not the end of the world

    def _prune_stale_debounce_state(self, belief_ids: Set[UUID]) -> None:
        self._above_threshold &= belief_ids

    async def audit(
        self, beliefs: List[Belief], store=None
    ) -> List[ContradictionDetectedEvent]:
        """
        Compute tensions and emit events for beliefs that newly crossed threshold.
        Optionally persists state via store if it has save/load_contradiction_state.
        """
        if not beliefs:
            return []

        if len(beliefs) > MAX_BELIEFS_PER_AUDIT:
            logger.warning(
                f"audit got {len(beliefs)} beliefs, capping at {MAX_BELIEFS_PER_AUDIT}"
            )
            beliefs = beliefs[:MAX_BELIEFS_PER_AUDIT]

        if store is not None:
            await self._load_persisted_state(store)

        # clean up stale debounce entries
        current_ids = {b.id for b in beliefs}
        self._prune_stale_debounce_state(current_ids)

        self._cache_embeddings(beliefs)
        tension_scores, top_contradictors = self._compute_tensions_from_cache(beliefs)

        # Build content lookup for event population
        belief_content_map = {b.id: b.content for b in beliefs}

        for b in beliefs:
            if b.id not in tension_scores:
                logger.warning(f"tension missing for belief {b.id}")

        now = datetime.now(timezone.utc)
        events = []
        currently_above: Set[UUID] = set()

        for belief in beliefs:
            t = tension_scores.get(belief.id, 0.0)
            threshold = self._get_threshold_for_belief(belief)

            if t > threshold:
                currently_above.add(belief.id)
                # debounce: only fire if newly above
                if belief.id not in self._above_threshold:
                    # get contradictor info if available
                    contradictor = top_contradictors.get(belief.id)

                    # extract semantic result if present
                    semantic_result = contradictor[3] if contradictor else None

                    events.append(
                        ContradictionDetectedEvent(
                            belief_id=belief.id,
                            tension=t,
                            threshold=threshold,
                            timestamp=now,
                            contradicting_belief_id=contradictor[0] if contradictor else None,
                            belief_content=belief.content,
                            contradicting_content=contradictor[1] if contradictor else "",
                            similarity_score=contradictor[2] if contradictor else 0.0,
                            reason_codes=semantic_result.reason_codes if semantic_result else [],
                            semantic_confidence=semantic_result.confidence if semantic_result else 0.0,
                            fallback_used=semantic_result.fallback_used if semantic_result else True,
                        )
                    )

        self._above_threshold = currently_above

        if store is not None:
            await self._save_persisted_state(store)

        return events


__all__ = ["ContradictionAuditorAgent", "ContradictionDetectedEvent"]
