# Author: Bradley R. Kinnard
"""
NLI (Natural Language Inference) model-based contradiction detection.
Uses pre-trained transformer models for high-accuracy fallback when
rule-based detection returns unknown or low confidence results.
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# lazy-loaded model
_nli_pipeline = None
_nli_available: bool | None = None
_nli_model_used: str = ""

# model to use - cross-encoder DeBERTa v3 trained on NLI tasks, best accuracy for contradiction detection
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
FALLBACK_MODEL = "facebook/bart-large-mnli"  # backup if cross-encoder unavailable


@dataclass
class NLIResult:
    """Result from NLI model inference."""

    label: Literal["contradiction", "entailment", "neutral"]
    score: float
    raw_scores: dict[str, float]
    model_used: str


def _load_nli_pipeline():
    """Lazy load the NLI pipeline. Returns None if unavailable."""
    global _nli_pipeline, _nli_available, _nli_model_used

    if _nli_available is False:
        return None
    if _nli_pipeline is not None:
        return _nli_pipeline

    try:
        from transformers import pipeline

        logger.info(f"Loading NLI model: {DEFAULT_MODEL}")
        _nli_pipeline = pipeline(
            "text-classification",
            model=DEFAULT_MODEL,
            top_k=None,
        )
        _nli_model_used = DEFAULT_MODEL
        _nli_available = True
        logger.info("NLI model loaded successfully")
        return _nli_pipeline

    except Exception as e:
        logger.warning(f"Failed to load {DEFAULT_MODEL}: {e}")

        try:
            from transformers import pipeline

            logger.info(f"Trying fallback model: {FALLBACK_MODEL}")
            _nli_pipeline = pipeline(
                "text-classification",
                model=FALLBACK_MODEL,
                top_k=None,
            )
            _nli_model_used = FALLBACK_MODEL
            _nli_available = True
            logger.info("Fallback NLI model loaded")
            return _nli_pipeline

        except Exception as e2:
            logger.warning(f"NLI models unavailable: {e2}")
            _nli_available = False
            return None


def is_nli_available() -> bool:
    """Check if NLI model is available (attempts lazy load)."""
    _load_nli_pipeline()
    return _nli_available is True


def classify_nli(premise: str, hypothesis: str) -> NLIResult | None:
    """
    Classify relationship between premise and hypothesis using NLI model.

    Returns NLIResult with label (contradiction/entailment/neutral) and score,
    or None if model unavailable.
    """
    pipe = _load_nli_pipeline()
    if pipe is None:
        return None

    try:
        # cross-encoder/nli-deberta-v3-base accepts "[SEP]" between premise and hypothesis
        input_text = f"{premise} [SEP] {hypothesis}"

        results = pipe(input_text)

        # parse results - cross-encoder returns list of list of dicts with 'label' and 'score'
        # flatten if nested (top_k=None wraps in extra list)
        if isinstance(results, list) and len(results) > 0:
            items = results[0] if isinstance(results[0], list) else results
            if isinstance(items, list) and len(items) > 0 and isinstance(items[0], dict) and "label" in items[0]:
                raw_scores = {r["label"].lower(): r["score"] for r in items}

                # normalize label names (some models use LABEL_0, etc.)
                label_map = {
                    "contradiction": "contradiction",
                    "entailment": "entailment",
                    "neutral": "neutral",
                    "label_0": "entailment",
                    "label_1": "neutral",
                    "label_2": "contradiction",
                }

                normalized_scores = {}
                for label, score in raw_scores.items():
                    norm_label = label_map.get(label, label)
                    normalized_scores[norm_label] = score

                # find top label
                top_label = max(normalized_scores, key=normalized_scores.get)
                top_score = normalized_scores[top_label]

                return NLIResult(
                    label=top_label,
                    score=top_score,
                    raw_scores=normalized_scores,
                    model_used=_nli_model_used,
                )

        logger.warning(f"Unexpected NLI output format: {results}")
        return None

    except Exception as e:
        logger.error(f"NLI inference failed: {e}")
        return None


def check_contradiction_nli(text_a: str, text_b: str) -> tuple[bool, float] | None:
    """
    Check if two texts contradict using NLI model.

    Returns (is_contradiction, confidence) tuple, or None if model unavailable.
    Checks both directions (A→B and B→A) for robustness.
    """
    # check A implies B
    result_ab = classify_nli(text_a, text_b)
    if result_ab is None:
        return None

    # check B implies A (for symmetric contradiction detection)
    result_ba = classify_nli(text_b, text_a)
    if result_ba is None:
        return None

    # contradiction if either direction shows contradiction with high confidence
    ab_contra = result_ab.label == "contradiction"
    ba_contra = result_ba.label == "contradiction"

    if ab_contra or ba_contra:
        # take the higher confidence
        confidence = max(
            result_ab.score if ab_contra else 0.0,
            result_ba.score if ba_contra else 0.0,
        )
        return (True, confidence)

    # not a contradiction - return inverse of contradiction scores
    avg_non_contra = 1.0 - (
        result_ab.raw_scores.get("contradiction", 0.0)
        + result_ba.raw_scores.get("contradiction", 0.0)
    ) / 2
    return (False, avg_non_contra)


__all__ = [
    "NLIResult",
    "is_nli_available",
    "classify_nli",
    "check_contradiction_nli",
]
