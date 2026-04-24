# Author: Bradley R. Kinnard
"""
Semantic contradiction detection using NLP parsing and rule-based analysis.
Extracts propositions from text and applies structured rules to detect conflicts.
Falls back to NLI transformer model when rule-based detection is uncertain.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

# spacy loaded lazily to allow graceful fallback
_nlp = None
_nlp_available = None


def _get_nlp():
    """Lazy load spacy model. Returns None if unavailable."""
    global _nlp, _nlp_available
    if _nlp_available is False:
        return None
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        _nlp_available = True
        return _nlp
    except Exception as e:
        logger.warning(f"spacy unavailable, falling back to legacy: {e}")
        _nlp_available = False
        return None


def _try_nli_fallback(text_a: str, text_b: str) -> tuple[bool, float] | None:
    """Try NLI model for contradiction detection. Returns (is_contradiction, confidence)."""
    try:
        from backend.core.bel.nli_detector import check_contradiction_nli
        return check_contradiction_nli(text_a, text_b)
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"NLI fallback failed: {e}")
        return None


# rule weights for scoring
RULE_WEIGHTS = {
    "NEG_DIRECT": 0.8,
    "NEG_PRED_FLIP": 0.7,
    "MOD_FACTUAL_VS_POSSIBLE": 0.4,
    "MOD_NECESSARY_VS_IMPOSSIBLE": 0.6,
    "TEMP_SAME_ANCHOR_CONFLICT": 0.7,
    "NUM_VALUE_CONFLICT": 0.8,
    "NUM_UNIT_CONVERTED": 0.7,
    "NUM_COMPARATOR_CONFLICT": 0.6,
    "QUANT_UNIVERSAL_VS_NONE": 0.9,
    "QUANT_UNIVERSAL_VS_EXISTENTIAL_NEG": 0.6,
    "ENT_SAME_ATTRIBUTE_CONFLICT": 0.8,
    "ENT_EXCLUSIVE_CATEGORY": 0.9,
    "LEGACY_ANTONYM": 0.3,
    "LEGACY_NEGATION_WORD": 0.3,
    "LEGACY_NUMERIC": 0.3,
}

# modality markers
MODALITY_POSSIBLE = {"might", "may", "could", "possibly", "perhaps", "maybe"}
MODALITY_NECESSARY = {"must", "have to", "need to", "has to", "required"}
MODALITY_IMPOSSIBLE = {"cannot", "can't", "couldn't", "impossible"}

# quantifier markers
QUANTIFIER_UNIVERSAL = {"all", "every", "everyone", "everything", "always", "each"}
QUANTIFIER_NONE = {"no", "none", "nobody", "nothing", "never", "no one"}
QUANTIFIER_EXISTENTIAL = {"some", "someone", "something", "sometimes", "a few", "several"}

# exclusive category pairs
EXCLUSIVE_CATEGORIES = [
    ({"bachelor", "single", "unmarried"}, {"married", "wed", "spouse"}),
    ({"alive", "living"}, {"dead", "deceased"}),
    ({"open"}, {"closed", "shut"}),
    ({"empty"}, {"full"}),
    ({"true"}, {"false"}),
    ({"present", "here"}, {"absent", "gone", "away"}),
]

# antonym pairs for legacy fallback and entity attribute conflicts
ANTONYM_PAIRS = [
    ("true", "false"), ("yes", "no"), ("always", "never"),
    ("good", "bad"), ("like", "dislike"), ("love", "hate"),
    ("hot", "cold"), ("warm", "cold"), ("warm", "cool"), ("hot", "cool"),
    ("bright", "dark"), ("light", "dark"), ("sunny", "dark"),
    ("sunny", "cloudy"), ("clear", "cloudy"), ("dry", "wet"), ("rainy", "sunny"),
    ("big", "small"), ("large", "small"), ("many", "few"), ("more", "less"),
    ("fast", "slow"), ("quick", "slow"),
    ("up", "down"), ("high", "low"), ("left", "right"), ("in", "out"),
    ("open", "closed"), ("on", "off"), ("alive", "dead"),
    ("new", "old"), ("young", "old"), ("clean", "dirty"),
    ("happy", "sad"), ("calm", "angry"),
    ("agree", "disagree"), ("accept", "reject"), ("include", "exclude"),
    ("red", "blue"), ("red", "green"), ("blue", "green"),
    ("white", "black"), ("tall", "short"), ("long", "short"),
]

# negation words for legacy fallback
NEGATION_WORDS = frozenset([
    "not", "no", "never", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "won't", "can't", "cannot"
])

# common unit conversions (base unit -> multiplier)
UNIT_CONVERSIONS = {
    # length to meters
    "m": ("length", 1.0),
    "meter": ("length", 1.0),
    "meters": ("length", 1.0),
    "cm": ("length", 0.01),
    "centimeter": ("length", 0.01),
    "centimeters": ("length", 0.01),
    "mm": ("length", 0.001),
    "km": ("length", 1000.0),
    "kilometer": ("length", 1000.0),
    "kilometers": ("length", 1000.0),
    "ft": ("length", 0.3048),
    "feet": ("length", 0.3048),
    "foot": ("length", 0.3048),
    "in": ("length", 0.0254),
    "inch": ("length", 0.0254),
    "inches": ("length", 0.0254),
    "mile": ("length", 1609.34),
    "miles": ("length", 1609.34),
    # weight to kg
    "kg": ("weight", 1.0),
    "kilogram": ("weight", 1.0),
    "kilograms": ("weight", 1.0),
    "g": ("weight", 0.001),
    "gram": ("weight", 0.001),
    "grams": ("weight", 0.001),
    "lb": ("weight", 0.453592),
    "lbs": ("weight", 0.453592),
    "pound": ("weight", 0.453592),
    "pounds": ("weight", 0.453592),
    # time to seconds
    "s": ("time", 1.0),
    "sec": ("time", 1.0),
    "second": ("time", 1.0),
    "seconds": ("time", 1.0),
    "min": ("time", 60.0),
    "minute": ("time", 60.0),
    "minutes": ("time", 60.0),
    "h": ("time", 3600.0),
    "hr": ("time", 3600.0),
    "hour": ("time", 3600.0),
    "hours": ("time", 3600.0),
    "day": ("time", 86400.0),
    "days": ("time", 86400.0),
    "week": ("time", 604800.0),
    "weeks": ("time", 604800.0),
    "year": ("time", 31536000.0),
    "years": ("time", 31536000.0),
    # currency (relative, not real conversion)
    "$": ("currency", 1.0),
    "dollar": ("currency", 1.0),
    "dollars": ("currency", 1.0),
    "usd": ("currency", 1.0),
    # temperature (special handling needed)
    "c": ("temp_c", 1.0),
    "celsius": ("temp_c", 1.0),
    "f": ("temp_f", 1.0),
    "fahrenheit": ("temp_f", 1.0),
    "degrees": ("temp_generic", 1.0),
    "°": ("temp_generic", 1.0),
    # percentage
    "%": ("percent", 1.0),
    "percent": ("percent", 1.0),
}


@dataclass
class Proposition:
    """Normalized semantic representation extracted from text."""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    negated: bool = False
    modality: str = "factual"  # factual | possible | necessary | conditional
    tense: str = "present"  # past | present | future
    time_anchor: str | None = None
    quantifier: str = "bare"  # universal | existential | bare | numeric
    quantity: float | None = None
    unit: str | None = None
    comparator: str | None = None  # gt | lt | eq | gte | lte | approx
    entity_id: str | None = None
    source_span: str = ""
    extraction_confidence: float = 1.0


@dataclass
class ContradictionResult:
    """Output from semantic contradiction check."""
    label: str = "unknown"  # contradiction | not_contradiction | unknown
    confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    propositions_a: list[Proposition] = field(default_factory=list)
    propositions_b: list[Proposition] = field(default_factory=list)
    fallback_used: bool = False
    rule_trace: list[dict] = field(default_factory=list)


class SemanticContradictionDetector(Protocol):
    """Interface for contradiction detection."""

    def check(self, text_a: str, text_b: str) -> ContradictionResult:
        """Check if two texts contradict. Must be deterministic."""
        ...

    def check_batch(self, pairs: list[tuple[str, str]]) -> list[ContradictionResult]:
        """Batch check for efficiency."""
        ...


def _extract_numbers(text: str) -> list[tuple[float, str, str]]:
    """Extract numbers with units and comparators.

    Returns list of (value, unit, comparator) tuples.
    """
    results = []
    # pattern: optional comparator, number, optional unit
    pattern = r'(?:(more than|less than|over|under|about|approximately|around|exactly|at least|at most|>|<|>=|<=|~)\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|degrees?|°|[fFcC]|mph|dollars?|\$|minutes?|hours?|days?|years?|feet|ft|miles?|lbs?|kg|cm|mm|km|m|meters?|inches?|in|pounds?|grams?|g)?'

    for match in re.finditer(pattern, text, re.IGNORECASE):
        comp_str, num_str, unit = match.groups()
        try:
            # remove commas
            num = float(num_str.replace(",", ""))
            unit = (unit or "").lower().strip()

            # normalize comparator
            comparator = None
            if comp_str:
                comp_lower = comp_str.lower()
                if comp_lower in ("more than", "over", ">"):
                    comparator = "gt"
                elif comp_lower in ("less than", "under", "<"):
                    comparator = "lt"
                elif comp_lower in (">=", "at least"):
                    comparator = "gte"
                elif comp_lower in ("<=", "at most"):
                    comparator = "lte"
                elif comp_lower in ("about", "approximately", "around", "~"):
                    comparator = "approx"
                elif comp_lower == "exactly":
                    comparator = "eq"

            results.append((num, unit, comparator))
        except ValueError:
            pass

    return results


def _normalize_unit_value(value: float, unit: str) -> tuple[float, str] | None:
    """Convert value to base unit. Returns (normalized_value, unit_type) or None."""
    unit_lower = unit.lower().strip()
    if unit_lower in UNIT_CONVERSIONS:
        unit_type, multiplier = UNIT_CONVERSIONS[unit_lower]
        return (value * multiplier, unit_type)
    return None


def _detect_modality(text: str) -> str:
    """Detect modality from text."""
    text_lower = text.lower()
    words = set(text_lower.split())

    if words & MODALITY_IMPOSSIBLE:
        return "impossible"
    if words & MODALITY_NECESSARY:
        return "necessary"
    if words & MODALITY_POSSIBLE:
        return "possible"
    return "factual"


def _detect_quantifier(text: str) -> str:
    """Detect quantifier type from text."""
    text_lower = text.lower()
    words = set(text_lower.split())

    if words & QUANTIFIER_UNIVERSAL:
        return "universal"
    if words & QUANTIFIER_NONE:
        return "none"
    if words & QUANTIFIER_EXISTENTIAL:
        return "existential"
    return "bare"


def _detect_tense(doc) -> str:
    """Detect tense from spacy doc."""
    for token in doc:
        if token.pos_ == "VERB":
            if "Past" in token.morph.get("Tense", []):
                return "past"
            if "Fut" in str(token.morph):
                return "future"
    # check for will/shall
    for token in doc:
        if token.lemma_ in ("will", "shall") and token.pos_ == "AUX":
            return "future"
    return "present"


def _has_negation(doc) -> bool:
    """Check if doc contains negation modifying the main verb."""
    for token in doc:
        if token.dep_ == "neg":
            return True
        if token.text.lower() in ("not", "n't", "never", "no"):
            return True
    return False


def _extract_subject_predicate_object(doc) -> tuple[str, str, str]:
    """Extract core triple from spacy doc."""
    subject = ""
    predicate = ""
    obj = ""

    # find root verb
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        # fallback: first verb or noun
        for token in doc:
            if token.pos_ == "VERB":
                root = token
                break

    if root:
        predicate = root.lemma_

        # find subject
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = " ".join(t.text for t in child.subtree)
                break

        # find object
        for child in root.children:
            if child.dep_ in ("dobj", "attr", "acomp", "pobj"):
                obj = " ".join(t.text for t in child.subtree)
                break
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        obj = " ".join(t.text for t in grandchild.subtree)
                        break

    # fallback for copular sentences
    if not predicate:
        for token in doc:
            if token.pos_ == "AUX" and token.lemma_ == "be":
                predicate = "be"
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = " ".join(t.text for t in child.subtree)
                    if child.dep_ in ("attr", "acomp"):
                        obj = " ".join(t.text for t in child.subtree)
                break

    return subject.strip(), predicate.strip(), obj.strip()


def _extract_propositions(text: str) -> list[Proposition]:
    """Extract propositions from text using spacy."""
    nlp = _get_nlp()
    if nlp is None:
        # fallback: create minimal proposition from text
        return [Proposition(
            source_span=text,
            extraction_confidence=0.3
        )]

    doc = nlp(text)
    props = []

    # extract main proposition
    subject, predicate, obj = _extract_subject_predicate_object(doc)

    # detect modifiers
    negated = _has_negation(doc)
    modality = _detect_modality(text)
    tense = _detect_tense(doc)
    quantifier = _detect_quantifier(text)

    # extract numbers
    numbers = _extract_numbers(text)
    quantity = None
    unit = None
    comparator = None
    if numbers:
        quantity, unit, comparator = numbers[0]

    confidence = 1.0
    if not subject:
        confidence -= 0.3
    if not predicate:
        confidence -= 0.3

    props.append(Proposition(
        subject=subject.lower() if subject else "",
        predicate=predicate.lower() if predicate else "",
        object=obj.lower() if obj else "",
        negated=negated,
        modality=modality,
        tense=tense,
        quantifier=quantifier,
        quantity=quantity,
        unit=unit,
        comparator=comparator,
        source_span=text,
        extraction_confidence=max(0.3, confidence)
    ))

    return props


def _check_negation_rules(
    props_a: list[Proposition],
    props_b: list[Proposition]
) -> list[dict]:
    """Check negation-based contradiction rules."""
    traces = []

    for i, pa in enumerate(props_a):
        for j, pb in enumerate(props_b):
            # skip if either has low confidence
            if pa.extraction_confidence < 0.5 or pb.extraction_confidence < 0.5:
                continue

            # NEG_DIRECT: same subject+predicate+object, one negated
            if (pa.subject and pb.subject and
                pa.predicate and pb.predicate):

                subj_match = pa.subject == pb.subject or _subjects_overlap(pa.subject, pb.subject)
                pred_match = pa.predicate == pb.predicate
                obj_match = pa.object == pb.object or _objects_overlap(pa.object, pb.object)

                if subj_match and pred_match and obj_match:
                    if pa.negated != pb.negated:
                        traces.append({
                            "rule_code": "NEG_DIRECT",
                            "prop_a_index": i,
                            "prop_b_index": j,
                            "matched_fields": ["subject", "predicate", "object"],
                            "conflict_detail": {
                                "negated_a": pa.negated,
                                "negated_b": pb.negated
                            },
                            "contribution": RULE_WEIGHTS["NEG_DIRECT"]
                        })

            # NEG_PRED_FLIP: predicate implies opposite when negated
            if (pa.subject and pb.subject and
                _subjects_overlap(pa.subject, pb.subject)):

                if pa.predicate and pb.predicate:
                    # check if predicates are antonyms
                    if _are_antonyms(pa.predicate, pb.predicate):
                        if not (pa.negated and pb.negated):
                            traces.append({
                                "rule_code": "NEG_PRED_FLIP",
                                "prop_a_index": i,
                                "prop_b_index": j,
                                "matched_fields": ["subject"],
                                "conflict_detail": {
                                    "pred_a": pa.predicate,
                                    "pred_b": pb.predicate
                                },
                                "contribution": RULE_WEIGHTS["NEG_PRED_FLIP"]
                            })

    return traces


def _check_modality_rules(
    props_a: list[Proposition],
    props_b: list[Proposition]
) -> list[dict]:
    """Check modality-based contradiction rules."""
    traces = []

    for i, pa in enumerate(props_a):
        for j, pb in enumerate(props_b):
            if pa.extraction_confidence < 0.5 or pb.extraction_confidence < 0.5:
                continue

            if not _subjects_overlap(pa.subject, pb.subject):
                continue

            # MOD_NECESSARY_VS_IMPOSSIBLE
            if ((pa.modality == "necessary" and pb.modality == "impossible") or
                (pa.modality == "impossible" and pb.modality == "necessary")):
                traces.append({
                    "rule_code": "MOD_NECESSARY_VS_IMPOSSIBLE",
                    "prop_a_index": i,
                    "prop_b_index": j,
                    "matched_fields": ["subject"],
                    "conflict_detail": {
                        "modality_a": pa.modality,
                        "modality_b": pb.modality
                    },
                    "contribution": RULE_WEIGHTS["MOD_NECESSARY_VS_IMPOSSIBLE"]
                })

            # MOD_FACTUAL_VS_POSSIBLE with opposite predicate/negation
            if pa.modality == "factual" and pb.modality == "possible":
                if pa.negated != pb.negated and pa.predicate == pb.predicate:
                    traces.append({
                        "rule_code": "MOD_FACTUAL_VS_POSSIBLE",
                        "prop_a_index": i,
                        "prop_b_index": j,
                        "matched_fields": ["subject", "predicate"],
                        "conflict_detail": {
                            "modality_a": pa.modality,
                            "modality_b": pb.modality
                        },
                        "contribution": RULE_WEIGHTS["MOD_FACTUAL_VS_POSSIBLE"]
                    })

    return traces


def _check_temporal_rules(
    props_a: list[Proposition],
    props_b: list[Proposition]
) -> list[dict]:
    """Check temporal contradiction rules."""
    traces = []

    for i, pa in enumerate(props_a):
        for j, pb in enumerate(props_b):
            if pa.extraction_confidence < 0.5 or pb.extraction_confidence < 0.5:
                continue

            if not _subjects_overlap(pa.subject, pb.subject):
                continue

            # TEMP_SAME_ANCHOR_CONFLICT: same tense, conflicting predicates/objects
            if pa.tense == pb.tense and pa.tense == "present":
                # check for attribute conflict in objects
                if pa.predicate == pb.predicate or (pa.predicate == "be" and pb.predicate == "be"):
                    if pa.object and pb.object and _are_antonyms(pa.object, pb.object):
                        traces.append({
                            "rule_code": "TEMP_SAME_ANCHOR_CONFLICT",
                            "prop_a_index": i,
                            "prop_b_index": j,
                            "matched_fields": ["subject", "predicate", "tense"],
                            "conflict_detail": {
                                "object_a": pa.object,
                                "object_b": pb.object
                            },
                            "contribution": RULE_WEIGHTS["TEMP_SAME_ANCHOR_CONFLICT"]
                        })

    return traces


def _check_numeric_rules(
    props_a: list[Proposition],
    props_b: list[Proposition]
) -> list[dict]:
    """Check numeric/quantity contradiction rules."""
    traces = []

    for i, pa in enumerate(props_a):
        for j, pb in enumerate(props_b):
            if pa.quantity is None or pb.quantity is None:
                continue

            if not _subjects_overlap(pa.subject, pb.subject):
                # still check if predicates match for general numeric conflicts
                if not (pa.predicate and pb.predicate and pa.predicate == pb.predicate):
                    continue

            # try unit conversion
            val_a = pa.quantity
            val_b = pb.quantity
            unit_type_a = None
            unit_type_b = None

            if pa.unit:
                norm_a = _normalize_unit_value(pa.quantity, pa.unit)
                if norm_a:
                    val_a, unit_type_a = norm_a

            if pb.unit:
                norm_b = _normalize_unit_value(pb.quantity, pb.unit)
                if norm_b:
                    val_b, unit_type_b = norm_b

            # check if units are compatible
            if unit_type_a and unit_type_b and unit_type_a != unit_type_b:
                continue  # incompatible units, skip

            # NUM_COMPARATOR_CONFLICT
            if pa.comparator and pb.comparator:
                conflict = False
                if pa.comparator == "gt" and pb.comparator == "lt" and val_a >= val_b:
                    conflict = True
                elif pa.comparator == "lt" and pb.comparator == "gt" and val_a <= val_b:
                    conflict = True
                elif pa.comparator == "gte" and pb.comparator == "lt" and val_a >= val_b:
                    conflict = True
                elif pa.comparator == "lte" and pb.comparator == "gt" and val_a <= val_b:
                    conflict = True

                if conflict:
                    traces.append({
                        "rule_code": "NUM_COMPARATOR_CONFLICT",
                        "prop_a_index": i,
                        "prop_b_index": j,
                        "matched_fields": ["subject"] if pa.subject else [],
                        "conflict_detail": {
                            "value_a": val_a,
                            "value_b": val_b,
                            "comp_a": pa.comparator,
                            "comp_b": pb.comparator
                        },
                        "contribution": RULE_WEIGHTS["NUM_COMPARATOR_CONFLICT"]
                    })
                    continue

            # NUM_VALUE_CONFLICT / NUM_UNIT_CONVERTED
            if val_a != 0 or val_b != 0:
                max_val = max(abs(val_a), abs(val_b))
                if max_val > 0:
                    diff_pct = abs(val_a - val_b) / max_val
                    if diff_pct > 0.2:  # more than 20% different
                        rule_code = "NUM_UNIT_CONVERTED" if (unit_type_a or unit_type_b) else "NUM_VALUE_CONFLICT"
                        traces.append({
                            "rule_code": rule_code,
                            "prop_a_index": i,
                            "prop_b_index": j,
                            "matched_fields": ["subject"] if pa.subject else [],
                            "conflict_detail": {
                                "value_a": pa.quantity,
                                "value_b": pb.quantity,
                                "unit_a": pa.unit,
                                "unit_b": pb.unit,
                                "normalized_a": val_a,
                                "normalized_b": val_b
                            },
                            "contribution": RULE_WEIGHTS[rule_code]
                        })

    return traces


def _check_quantifier_rules(
    props_a: list[Proposition],
    props_b: list[Proposition]
) -> list[dict]:
    """Check quantifier contradiction rules."""
    traces = []

    for i, pa in enumerate(props_a):
        for j, pb in enumerate(props_b):
            if pa.extraction_confidence < 0.5 or pb.extraction_confidence < 0.5:
                continue

            # predicates should match or be about same topic
            pred_match = (pa.predicate == pb.predicate) or (pa.predicate and pb.predicate)
            if not pred_match:
                continue

            # QUANT_UNIVERSAL_VS_NONE
            if ((pa.quantifier == "universal" and pb.quantifier == "none") or
                (pa.quantifier == "none" and pb.quantifier == "universal")):
                traces.append({
                    "rule_code": "QUANT_UNIVERSAL_VS_NONE",
                    "prop_a_index": i,
                    "prop_b_index": j,
                    "matched_fields": ["predicate"],
                    "conflict_detail": {
                        "quant_a": pa.quantifier,
                        "quant_b": pb.quantifier
                    },
                    "contribution": RULE_WEIGHTS["QUANT_UNIVERSAL_VS_NONE"]
                })

            # QUANT_UNIVERSAL_VS_EXISTENTIAL_NEG
            if pa.quantifier == "universal" and pb.quantifier == "existential":
                # "all X" vs "some not X"
                if pb.negated and not pa.negated:
                    traces.append({
                        "rule_code": "QUANT_UNIVERSAL_VS_EXISTENTIAL_NEG",
                        "prop_a_index": i,
                        "prop_b_index": j,
                        "matched_fields": ["predicate"],
                        "conflict_detail": {
                            "quant_a": pa.quantifier,
                            "quant_b": pb.quantifier,
                            "negated_b": pb.negated
                        },
                        "contribution": RULE_WEIGHTS["QUANT_UNIVERSAL_VS_EXISTENTIAL_NEG"]
                    })

    return traces


def _check_entity_rules(
    props_a: list[Proposition],
    props_b: list[Proposition]
) -> list[dict]:
    """Check entity/attribute contradiction rules."""
    traces = []

    for i, pa in enumerate(props_a):
        for j, pb in enumerate(props_b):
            if not _subjects_overlap(pa.subject, pb.subject):
                continue

            # ENT_SAME_ATTRIBUTE_CONFLICT
            if pa.predicate == pb.predicate or (pa.predicate == "be" and pb.predicate == "be"):
                if pa.object and pb.object:
                    if _are_antonyms(pa.object, pb.object):
                        traces.append({
                            "rule_code": "ENT_SAME_ATTRIBUTE_CONFLICT",
                            "prop_a_index": i,
                            "prop_b_index": j,
                            "matched_fields": ["subject", "predicate"],
                            "conflict_detail": {
                                "object_a": pa.object,
                                "object_b": pb.object
                            },
                            "contribution": RULE_WEIGHTS["ENT_SAME_ATTRIBUTE_CONFLICT"]
                        })

            # ENT_EXCLUSIVE_CATEGORY
            obj_a = pa.object.lower() if pa.object else ""
            obj_b = pb.object.lower() if pb.object else ""

            for cat1, cat2 in EXCLUSIVE_CATEGORIES:
                a_in_1 = any(w in obj_a for w in cat1)
                a_in_2 = any(w in obj_a for w in cat2)
                b_in_1 = any(w in obj_b for w in cat1)
                b_in_2 = any(w in obj_b for w in cat2)

                if (a_in_1 and b_in_2) or (a_in_2 and b_in_1):
                    traces.append({
                        "rule_code": "ENT_EXCLUSIVE_CATEGORY",
                        "prop_a_index": i,
                        "prop_b_index": j,
                        "matched_fields": ["subject"],
                        "conflict_detail": {
                            "object_a": pa.object,
                            "object_b": pb.object,
                            "categories": [list(cat1), list(cat2)]
                        },
                        "contribution": RULE_WEIGHTS["ENT_EXCLUSIVE_CATEGORY"]
                    })
                    break

    return traces


def _subjects_overlap(subj_a: str, subj_b: str) -> bool:
    """Check if two subjects refer to the same entity (heuristic)."""
    if not subj_a or not subj_b:
        return False

    a = subj_a.lower().strip()
    b = subj_b.lower().strip()

    if a == b:
        return True

    # pronoun matching heuristic
    pronouns = {"i", "he", "she", "it", "they", "we", "you"}
    if a in pronouns or b in pronouns:
        return True  # assume pronouns may refer to same entity

    # check word overlap
    words_a = set(a.split())
    words_b = set(b.split())
    # remove articles
    words_a -= {"the", "a", "an"}
    words_b -= {"the", "a", "an"}

    if words_a and words_b and words_a & words_b:
        return True

    return False


def _objects_overlap(obj_a: str, obj_b: str) -> bool:
    """Check if two objects are semantically similar."""
    if not obj_a or not obj_b:
        return False

    a = obj_a.lower().strip()
    b = obj_b.lower().strip()

    if a == b:
        return True

    # word overlap check
    words_a = set(a.split())
    words_b = set(b.split())
    words_a -= {"the", "a", "an"}
    words_b -= {"the", "a", "an"}

    if words_a and words_b:
        overlap = len(words_a & words_b)
        total = len(words_a | words_b)
        if total > 0 and overlap / total > 0.5:
            return True

    return False


def _are_antonyms(word_a: str, word_b: str) -> bool:
    """Check if two words are antonyms."""
    a = word_a.lower().strip()
    b = word_b.lower().strip()

    for w1, w2 in ANTONYM_PAIRS:
        if (a == w1 and b == w2) or (a == w2 and b == w1):
            return True
        # partial match for phrases
        if (w1 in a and w2 in b) or (w2 in a and w1 in b):
            return True

    return False


def _legacy_negation_check(text_a: str, text_b: str) -> dict | None:
    """Legacy negation word asymmetry check."""
    t1 = text_a.lower()
    t2 = text_b.lower()

    def count_negations(text: str) -> int:
        return sum(1 for neg in NEGATION_WORDS if f" {neg} " in f" {text} ")

    neg1 = count_negations(t1)
    neg2 = count_negations(t2)

    if (neg1 > 0) != (neg2 > 0):
        return {
            "rule_code": "LEGACY_NEGATION_WORD",
            "prop_a_index": 0,
            "prop_b_index": 0,
            "matched_fields": [],
            "conflict_detail": {"neg_count_a": neg1, "neg_count_b": neg2},
            "contribution": RULE_WEIGHTS["LEGACY_NEGATION_WORD"]
        }
    return None


def _legacy_antonym_check(text_a: str, text_b: str) -> dict | None:
    """Legacy antonym list check."""
    t1 = text_a.lower()
    t2 = text_b.lower()

    def contains_word(text: str, word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", text) is not None

    for w1, w2 in ANTONYM_PAIRS:
        if (contains_word(t1, w1) and contains_word(t2, w2)) or \
           (contains_word(t1, w2) and contains_word(t2, w1)):
            return {
                "rule_code": "LEGACY_ANTONYM",
                "prop_a_index": 0,
                "prop_b_index": 0,
                "matched_fields": [],
                "conflict_detail": {"antonym_pair": [w1, w2]},
                "contribution": RULE_WEIGHTS["LEGACY_ANTONYM"]
            }
    return None


def _legacy_numeric_check(text_a: str, text_b: str) -> dict | None:
    """Legacy numeric conflict check."""
    nums_a = _extract_numbers(text_a)
    nums_b = _extract_numbers(text_b)

    if not nums_a or not nums_b:
        return None

    for n1, u1, _ in nums_a:
        for n2, u2, _ in nums_b:
            # units should match or both empty
            if u1 != u2 and u1 and u2:
                continue

            if n1 == 0 and n2 == 0:
                continue

            max_val = max(abs(n1), abs(n2))
            if max_val > 0:
                diff_pct = abs(n1 - n2) / max_val
                if diff_pct > 0.2:
                    return {
                        "rule_code": "LEGACY_NUMERIC",
                        "prop_a_index": 0,
                        "prop_b_index": 0,
                        "matched_fields": [],
                        "conflict_detail": {
                            "value_a": n1,
                            "value_b": n2,
                            "unit": u1 or u2
                        },
                        "contribution": RULE_WEIGHTS["LEGACY_NUMERIC"]
                    }
    return None


class RuleBasedContradictionDetector:
    """Rule-based semantic contradiction detector with fallback to legacy heuristics."""

    def __init__(self):
        self._nlp_available = None

    def check(self, text_a: str, text_b: str) -> ContradictionResult:
        """Check if two texts contradict."""
        result = ContradictionResult()

        # try semantic extraction
        props_a = _extract_propositions(text_a)
        props_b = _extract_propositions(text_b)

        result.propositions_a = props_a
        result.propositions_b = props_b

        # check if extraction succeeded
        extraction_failed = (
            (len(props_a) == 1 and props_a[0].extraction_confidence < 0.5) or
            (len(props_b) == 1 and props_b[0].extraction_confidence < 0.5)
        )

        all_traces = []

        if not extraction_failed:
            # run semantic rules in fixed order
            all_traces.extend(_check_negation_rules(props_a, props_b))
            all_traces.extend(_check_modality_rules(props_a, props_b))
            all_traces.extend(_check_temporal_rules(props_a, props_b))
            all_traces.extend(_check_numeric_rules(props_a, props_b))
            all_traces.extend(_check_quantifier_rules(props_a, props_b))
            all_traces.extend(_check_entity_rules(props_a, props_b))

        # if no semantic rules fired or extraction failed, try legacy
        if not all_traces:
            result.fallback_used = True

            legacy_neg = _legacy_negation_check(text_a, text_b)
            if legacy_neg:
                all_traces.append(legacy_neg)

            legacy_ant = _legacy_antonym_check(text_a, text_b)
            if legacy_ant:
                all_traces.append(legacy_ant)

            legacy_num = _legacy_numeric_check(text_a, text_b)
            if legacy_num:
                all_traces.append(legacy_num)

        result.rule_trace = all_traces

        # deduplicate reason codes while preserving order
        seen_codes = set()
        reason_codes = []
        for trace in all_traces:
            code = trace["rule_code"]
            if code not in seen_codes:
                seen_codes.add(code)
                reason_codes.append(code)
        result.reason_codes = reason_codes

        # compute confidence
        base_score = sum(trace["contribution"] for trace in all_traces)
        confidence = min(1.0, base_score)

        # apply extraction confidence penalty
        min_extraction_conf = min(
            min((p.extraction_confidence for p in props_a), default=1.0),
            min((p.extraction_confidence for p in props_b), default=1.0)
        )
        if min_extraction_conf < 0.8:
            confidence *= 0.7

        # apply fallback penalty
        if result.fallback_used:
            confidence *= 0.5

        result.confidence = round(confidence, 4)

        # determine label
        if confidence >= 0.5:
            result.label = "contradiction"
        elif confidence < 0.3 and all_traces:
            result.label = "not_contradiction"
        else:
            result.label = "unknown"

        # NLI fallback: if unknown or low confidence, try NLI model
        if result.label == "unknown" or (result.label == "contradiction" and confidence < 0.6):
            nli_result = _try_nli_fallback(text_a, text_b)
            if nli_result is not None:
                is_contra, nli_conf = nli_result
                if is_contra and nli_conf >= 0.6:
                    # NLI says contradiction - trust it proportionally
                    result.label = "contradiction"
                    # if NLI is highly confident and rules found nothing, NLI drives the score
                    if nli_conf >= 0.9 and confidence < 0.3:
                        result.confidence = round(nli_conf * 0.85, 4)
                    else:
                        result.confidence = round(min(1.0, confidence + nli_conf * 0.5), 4)
                    result.reason_codes.append("NLI_MODEL")
                    result.rule_trace.append({
                        "rule_code": "NLI_MODEL",
                        "triggered": True,
                        "prop_a_index": -1,
                        "prop_b_index": -1,
                        "matched_fields": ["nli_inference"],
                        "conflict_detail": {"nli_confidence": nli_conf},
                        "contribution": nli_conf * 0.6
                    })
                elif not is_contra and nli_conf >= 0.8:
                    # NLI says NOT contradiction with high confidence
                    if result.label == "unknown":
                        result.label = "not_contradiction"
                        result.confidence = round(nli_conf * 0.3, 4)

        return result

    def check_batch(self, pairs: list[tuple[str, str]]) -> list[ContradictionResult]:
        """Batch check for efficiency."""
        return [self.check(a, b) for a, b in pairs]


# module-level singleton for convenience
_detector = None


def get_detector() -> RuleBasedContradictionDetector:
    """Get or create the singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = RuleBasedContradictionDetector()
    return _detector


def check_contradiction(text_a: str, text_b: str) -> ContradictionResult:
    """Convenience function to check contradiction between two texts."""
    return get_detector().check(text_a, text_b)


__all__ = [
    "Proposition",
    "ContradictionResult",
    "SemanticContradictionDetector",
    "RuleBasedContradictionDetector",
    "check_contradiction",
    "get_detector",
]
