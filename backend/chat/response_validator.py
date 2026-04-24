# Author: Bradley R. Kinnard
"""
Response Validator - validates LLM responses against stored beliefs.
Extracts claims from LLM output and checks for contradictions with the belief store.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models.belief import Belief

logger = logging.getLogger(__name__)

# lazy-loaded
_nlp = None


def _get_nlp():
    """Lazy load spacy for claim extraction."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception as e:
        logger.warning(f"spacy unavailable for claim extraction: {e}")
        return None


@dataclass
class ExtractedClaim:
    """A factual claim extracted from LLM response."""
    text: str
    sentence: str
    confidence: float = 1.0  # how certain we are this is a factual claim
    is_hedged: bool = False  # "might be", "could be", etc.


@dataclass
class ValidationResult:
    """Result of validating LLM response against beliefs."""
    is_valid: bool = True
    contradictions: list[dict] = field(default_factory=list)
    claims_checked: int = 0
    flagged_claims: list[ExtractedClaim] = field(default_factory=list)


# hedging phrases that reduce claim confidence
HEDGE_PHRASES = frozenset([
    "might", "may", "could", "possibly", "perhaps", "maybe",
    "i think", "i believe", "it seems", "appears to",
    "not sure", "uncertain", "likely", "probably",
    "in my opinion", "as far as i know", "i'm not certain",
])

# phrases indicating the LLM is citing stored beliefs (should trust these)
CITATION_PHRASES = frozenset([
    "you mentioned", "you said", "you told me",
    "according to what you said", "based on what you told me",
    "from our conversation", "as you noted", "you indicated",
])


def extract_claims(response: str) -> list[ExtractedClaim]:
    """
    Extract factual claims from LLM response text.
    Filters out questions, hedged statements, and non-factual content.
    """
    claims = []
    nlp = _get_nlp()

    if nlp is None:
        # fallback: sentence splitting without NLP
        sentences = re.split(r'[.!?]+', response)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            if sent.endswith('?'):
                continue  # skip questions

            # check for hedging
            lower_sent = sent.lower()
            is_hedged = any(h in lower_sent for h in HEDGE_PHRASES)

            # skip if citing user's beliefs
            if any(c in lower_sent for c in CITATION_PHRASES):
                continue

            # basic factual claim detection: contains "is", "are", "was", "were", numbers
            is_factual = bool(re.search(r'\b(is|are|was|were|has|have|had|will|can|does|did)\b', lower_sent))
            is_factual = is_factual or bool(re.search(r'\d+', sent))

            if is_factual:
                claims.append(ExtractedClaim(
                    text=sent,
                    sentence=sent,
                    confidence=0.5 if is_hedged else 0.8,
                    is_hedged=is_hedged,
                ))
        return claims

    # NLP-based extraction
    doc = nlp(response)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 10:
            continue

        # skip questions
        if sent_text.endswith('?'):
            continue

        lower_sent = sent_text.lower()

        # skip if citing user's beliefs
        if any(c in lower_sent for c in CITATION_PHRASES):
            continue

        # check for hedging
        is_hedged = any(h in lower_sent for h in HEDGE_PHRASES)

        # check if sentence contains factual assertions
        has_verb = any(tok.pos_ == "VERB" for tok in sent)
        has_subj = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
        has_entity = any(ent.label_ in ("PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT") for ent in sent.ents)
        has_number = any(tok.like_num for tok in sent)

        # factual if has subject+verb or contains entities/numbers
        is_factual = (has_verb and has_subj) or has_entity or has_number

        if is_factual:
            confidence = 0.5 if is_hedged else 0.9
            claims.append(ExtractedClaim(
                text=sent_text,
                sentence=sent_text,
                confidence=confidence,
                is_hedged=is_hedged,
            ))

    return claims


def validate_response(
    response: str,
    beliefs: list["Belief"],
    contradiction_threshold: float = 0.6,
) -> ValidationResult:
    """
    Validate LLM response against stored beliefs.

    Extracts claims from response and checks each against beliefs
    for contradictions.

    Args:
        response: LLM response text
        beliefs: List of user beliefs to check against
        contradiction_threshold: Min confidence to flag contradiction

    Returns:
        ValidationResult with any contradictions found
    """
    from backend.core.bel.semantic_contradiction import check_contradiction

    result = ValidationResult()

    if not beliefs:
        return result  # nothing to validate against

    claims = extract_claims(response)
    result.claims_checked = len(claims)

    if not claims:
        return result  # no factual claims to check

    # check each claim against each belief
    for claim in claims:
        if claim.is_hedged:
            continue  # skip hedged claims

        for belief in beliefs:
            contra_result = check_contradiction(claim.text, belief.content)

            if contra_result.label == "contradiction" and contra_result.confidence >= contradiction_threshold:
                result.is_valid = False
                result.contradictions.append({
                    "claim": claim.text,
                    "belief_id": str(belief.id),
                    "belief_content": belief.content,
                    "confidence": contra_result.confidence,
                    "reason_codes": contra_result.reason_codes,
                })
                result.flagged_claims.append(claim)
                logger.warning(
                    f"LLM claim contradicts belief: '{claim.text[:50]}...' vs '{belief.content[:50]}...'"
                )

    return result


def get_correction_prompt(
    original_response: str,
    contradictions: list[dict],
    beliefs: list["Belief"],
) -> str:
    """
    Generate a prompt to correct LLM response that contradicted beliefs.
    """
    belief_context = "\n".join(
        f"- {b.content} (confidence: {b.confidence:.0%})"
        for b in beliefs[:10]
    )

    contradiction_details = "\n".join(
        f"- Your claim '{c['claim'][:60]}...' contradicts: '{c['belief_content'][:60]}...'"
        for c in contradictions[:5]
    )

    return f"""Your previous response contained claims that contradict what the user has told you.

WHAT THE USER HAS TOLD YOU (trust these):
{belief_context}

CONTRADICTIONS FOUND:
{contradiction_details}

Please regenerate your response, ensuring you don't contradict the user's stated facts. If you're uncertain about something, acknowledge that uncertainty rather than stating incorrect facts.

Original response to fix:
{original_response[:500]}..."""


__all__ = [
    "ExtractedClaim",
    "ValidationResult",
    "extract_claims",
    "validate_response",
    "get_correction_prompt",
]
