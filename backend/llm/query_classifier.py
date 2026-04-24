# Author: Bradley R. Kinnard
"""
Query Classifier - zero-shot classification for routing queries.
Uses a pre-trained model to classify queries into routing categories.
"""

import logging
import re
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

# lazy-loaded pipeline
_classifier = None
_classifier_available: bool | None = None

# query route types
RouteType = Literal["real-time", "belief-grounded", "general"]

# labels for zero-shot classification
ROUTE_LABELS = [
    "This query needs real-time or current information like weather, news, stock prices, traffic, or sports scores",
    "This query is about personal facts, preferences, or information the user previously shared",
    "This is a general question or conversation that doesn't need current data or personal memory",
]

LABEL_TO_ROUTE: dict[str, RouteType] = {
    ROUTE_LABELS[0]: "real-time",
    ROUTE_LABELS[1]: "belief-grounded",
    ROUTE_LABELS[2]: "general",
}


def _load_classifier():
    """Lazy load zero-shot classifier."""
    global _classifier, _classifier_available

    if _classifier_available is False:
        return None
    if _classifier is not None:
        return _classifier

    try:
        from transformers import pipeline

        logger.info("Loading zero-shot classifier for query routing")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU
        )
        _classifier_available = True
        logger.info("Query classifier loaded")
        return _classifier
    except Exception as e:
        logger.warning(f"Zero-shot classifier unavailable: {e}")
        _classifier_available = False
        return None


def is_classifier_available() -> bool:
    """Check if classifier is available."""
    _load_classifier()
    return _classifier_available is True


# regex fallback patterns for when classifier unavailable
_LIVE_INFO_PATTERNS = [
    r"\b(right now|currently|today|tonight|this morning|this evening)\b.*\b(weather|traffic|news|stock|price)\b",
    r"\b(weather|traffic|news|stock|price).*\b(right now|currently|today|tonight)\b",
    r"\bwhat('s| is) the (weather|traffic|news|time)\b",
    r"\bhow('s| is) the (weather|traffic)\b",
    r"\b(current|live|real-time|latest)\s+(weather|traffic|news|stock|price|score)\b",
    r"\b(look up|search for|find out|check|google)\b",
    r"\bcan you (find|search|look|check)\b",
    r"\b(latest|recent|breaking|current)\s+(news|events|updates|headlines)\b",
    r"\bwhat('s| is) happening\b",
    r"\bwhat happened (today|yesterday|this week)\b",
    r"\b(score|result)s?\s+(of|for|in)\b.*\b(game|match)\b",
    r"\bwho (won|is winning|scored)\b",
    r"\b(stock|share|crypto|bitcoin|eth)\s+price\b",
    r"\bhow much is\s+\w+\s+(worth|trading)\b",
    r"\bwhat time is it in\b",
]
_LIVE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in _LIVE_INFO_PATTERNS]


def _regex_fallback(query: str) -> RouteType:
    """Fallback to regex when classifier unavailable."""
    if any(pat.search(query) for pat in _LIVE_PATTERNS_COMPILED):
        return "real-time"
    return "general"


@lru_cache(maxsize=256)
def classify_query(query: str) -> tuple[RouteType, float]:
    """
    Classify a query into routing category.

    Returns:
        Tuple of (route_type, confidence)
    """
    classifier = _load_classifier()

    if classifier is None:
        # fallback to regex
        route = _regex_fallback(query)
        return (route, 0.7 if route == "real-time" else 0.5)

    try:
        result = classifier(
            query,
            ROUTE_LABELS,
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]
        route = LABEL_TO_ROUTE.get(top_label, "general")

        logger.debug(f"Query classified as {route} (conf={top_score:.2f}): {query[:50]}")
        return (route, top_score)

    except Exception as e:
        logger.warning(f"Classification failed, using fallback: {e}")
        route = _regex_fallback(query)
        return (route, 0.5)


def needs_real_time_info(query: str, threshold: float = 0.6) -> bool:
    """
    Check if query needs real-time information.

    Uses zero-shot classifier with regex fallback.
    """
    route, confidence = classify_query(query)
    return route == "real-time" and confidence >= threshold


__all__ = [
    "RouteType",
    "classify_query",
    "needs_real_time_info",
    "is_classifier_available",
]
