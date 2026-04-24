#!/usr/bin/env python3
# Author: Bradley R. Kinnard
"""
Contradiction Detection Benchmark

Compares legacy heuristic detector vs semantic rule-based detector against:
1. ABES curated corpus (inspired by SNLI, MultiNLI, SICK)
2. Selected SNLI test cases for reproducibility

Outputs JSON artifact to results/contradiction_benchmark.json
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkCase:
    id: str
    category: str
    text_a: str
    text_b: str
    expected_label: str
    source: str


@dataclass
class DetectorResult:
    label: str
    confidence: float
    reason_codes: list[str]
    fallback_used: bool
    latency_ms: float


@dataclass
class CaseResult:
    case_id: str
    category: str
    expected: str
    legacy_result: DetectorResult
    semantic_result: DetectorResult
    legacy_correct: bool
    semantic_correct: bool


@dataclass
class CategoryMetrics:
    category: str
    total_cases: int
    legacy_correct: int
    semantic_correct: int
    legacy_accuracy: float
    semantic_accuracy: float
    improvement: float


@dataclass
class BenchmarkReport:
    timestamp: str
    corpus_version: str
    total_cases: int
    legacy_accuracy: float
    semantic_accuracy: float
    improvement: float
    category_metrics: list[CategoryMetrics]
    case_results: list[CaseResult]
    snli_sample_results: list[dict]


def legacy_detector(text_a: str, text_b: str) -> DetectorResult:
    """
    Legacy detector: embedding similarity + antonym/negation heuristics.
    Reimplemented here to isolate from new semantic logic.
    """
    import re
    start = time.perf_counter()

    # negation words
    negation_words = {"not", "no", "never", "don't", "doesn't", "didn't",
                      "isn't", "aren't", "wasn't", "weren't", "won't", "can't", "cannot"}

    # antonym pairs
    antonym_pairs = [
        ("true", "false"), ("yes", "no"), ("always", "never"),
        ("good", "bad"), ("like", "dislike"), ("love", "hate"),
        ("hot", "cold"), ("warm", "cold"), ("bright", "dark"),
        ("sunny", "cloudy"), ("dry", "wet"),
        ("big", "small"), ("large", "small"), ("many", "few"),
        ("fast", "slow"), ("up", "down"), ("high", "low"),
        ("open", "closed"), ("on", "off"), ("alive", "dead"),
        ("new", "old"), ("happy", "sad"),
        ("red", "blue"), ("tall", "short"),
    ]

    t1, t2 = text_a.lower(), text_b.lower()
    reason_codes = []

    # check negation asymmetry
    def count_neg(text):
        return sum(1 for neg in negation_words if f" {neg} " in f" {text} ")

    neg1, neg2 = count_neg(t1), count_neg(t2)
    if (neg1 > 0) != (neg2 > 0):
        reason_codes.append("LEGACY_NEGATION_WORD")

    # check antonyms
    def contains_word(text, word):
        return re.search(rf"\b{re.escape(word)}\b", text) is not None

    for w1, w2 in antonym_pairs:
        if (contains_word(t1, w1) and contains_word(t2, w2)) or \
           (contains_word(t1, w2) and contains_word(t2, w1)):
            reason_codes.append("LEGACY_ANTONYM")
            break

    # check numeric conflict
    pattern = r'(\d+(?:\.\d+)?)\s*(%|degrees?|°|dollars?|\$|minutes?|hours?|days?|years?|feet|ft|miles?|km|meters?|kg|lbs?)?'
    nums1 = [(float(m[0]), m[1].lower() if m[1] else "") for m in re.findall(pattern, t1)]
    nums2 = [(float(m[0]), m[1].lower() if m[1] else "") for m in re.findall(pattern, t2)]

    for n1, u1 in nums1:
        for n2, u2 in nums2:
            if u1 != u2 and u1 and u2:
                continue
            if n1 == 0 and n2 == 0:
                continue
            max_val = max(abs(n1), abs(n2))
            if max_val > 0:
                diff_pct = abs(n1 - n2) / max_val
                if diff_pct > 0.2:
                    reason_codes.append("LEGACY_NUMERIC")
                    break
        if "LEGACY_NUMERIC" in reason_codes:
            break

    latency = (time.perf_counter() - start) * 1000

    confidence = 0.3 * len(reason_codes) if reason_codes else 0.0
    confidence = min(1.0, confidence)

    label = "contradiction" if confidence >= 0.3 else "not_contradiction"

    return DetectorResult(
        label=label,
        confidence=round(confidence, 4),
        reason_codes=reason_codes,
        fallback_used=False,
        latency_ms=round(latency, 3)
    )


def semantic_detector(text_a: str, text_b: str) -> DetectorResult:
    """Run new semantic detector."""
    from backend.core.bel.semantic_contradiction import check_contradiction

    start = time.perf_counter()
    result = check_contradiction(text_a, text_b)
    latency = (time.perf_counter() - start) * 1000

    return DetectorResult(
        label=result.label,
        confidence=result.confidence,
        reason_codes=result.reason_codes,
        fallback_used=result.fallback_used,
        latency_ms=round(latency, 3)
    )


def load_corpus() -> list[BenchmarkCase]:
    """Load curated corpus."""
    corpus_path = PROJECT_ROOT / "data" / "contradiction_corpus.json"
    with open(corpus_path) as f:
        data = json.load(f)

    cases = []
    for c in data["cases"]:
        cases.append(BenchmarkCase(
            id=c["id"],
            category=c["category"],
            text_a=c["text_a"],
            text_b=c["text_b"],
            expected_label=c["expected_label"],
            source=c.get("source", "unknown")
        ))

    return cases


# Selected SNLI test cases for external validation
# These are real examples from the SNLI dataset (Bowman et al. 2015)
SNLI_SAMPLE = [
    {
        "premise": "A man inspects the uniform of a figure in some East Asian country.",
        "hypothesis": "The man is sleeping.",
        "gold_label": "contradiction",
        "source": "SNLI"
    },
    {
        "premise": "An older and younger man smiling.",
        "hypothesis": "Two men are smiling and laughing at the cats playing on the floor.",
        "gold_label": "neutral",  # not contradiction
        "source": "SNLI"
    },
    {
        "premise": "A black race car starts up in front of a crowd of people.",
        "hypothesis": "A man is driving down a lonely road.",
        "gold_label": "contradiction",
        "source": "SNLI"
    },
    {
        "premise": "A soccer game with multiple males playing.",
        "hypothesis": "Some men are playing a sport.",
        "gold_label": "entailment",  # not contradiction
        "source": "SNLI"
    },
    {
        "premise": "A smiling costumed woman is holding an umbrella.",
        "hypothesis": "A happy woman in a fairy costume holds an umbrella.",
        "gold_label": "neutral",
        "source": "SNLI"
    },
    {
        "premise": "A person on a horse jumps over a broken down airplane.",
        "hypothesis": "A person is at a diner, deciding between a soup and salad.",
        "gold_label": "contradiction",
        "source": "SNLI"
    },
    {
        "premise": "Children smiling and waving at camera.",
        "hypothesis": "They are smiling at their parents.",
        "gold_label": "neutral",
        "source": "SNLI"
    },
    {
        "premise": "A boy is jumping on skateboard in the middle of a red bridge.",
        "hypothesis": "The boy skates down the sidewalk.",
        "gold_label": "contradiction",
        "source": "SNLI"
    },
    {
        "premise": "An older man sits with his orange juice at a small table in a coffee shop.",
        "hypothesis": "A man is sleeping on the couch.",
        "gold_label": "contradiction",
        "source": "SNLI"
    },
    {
        "premise": "Two women are embracing while holding to go packages.",
        "hypothesis": "The sisters are hugging goodbye while holding to go packages.",
        "gold_label": "neutral",
        "source": "SNLI"
    },
    # MultiNLI inspired cases
    {
        "premise": "The cat is on the mat.",
        "hypothesis": "The cat is not on the mat.",
        "gold_label": "contradiction",
        "source": "MultiNLI-style"
    },
    {
        "premise": "John went to the store and bought milk.",
        "hypothesis": "John stayed home all day.",
        "gold_label": "contradiction",
        "source": "MultiNLI-style"
    },
    # SICK inspired cases
    {
        "premise": "A man is playing a piano.",
        "hypothesis": "A man is playing a keyboard.",
        "gold_label": "neutral",
        "source": "SICK-style"
    },
    {
        "premise": "There is no dog in the yard.",
        "hypothesis": "A dog is playing in the yard.",
        "gold_label": "contradiction",
        "source": "SICK-style"
    },
    {
        "premise": "The woman is not dancing.",
        "hypothesis": "A woman is dancing.",
        "gold_label": "contradiction",
        "source": "SICK-style"
    },
]


def run_snli_sample() -> list[dict]:
    """Evaluate on SNLI sample."""
    results = []
    for case in SNLI_SAMPLE:
        legacy = legacy_detector(case["premise"], case["hypothesis"])
        semantic = semantic_detector(case["premise"], case["hypothesis"])

        # for SNLI, we predict contradiction or not
        legacy_pred = "contradiction" if legacy.label == "contradiction" else "other"
        semantic_pred = "contradiction" if semantic.label == "contradiction" else "other"
        expected = "contradiction" if case["gold_label"] == "contradiction" else "other"

        results.append({
            "premise": case["premise"],
            "hypothesis": case["hypothesis"],
            "gold_label": case["gold_label"],
            "source": case["source"],
            "legacy_pred": legacy_pred,
            "semantic_pred": semantic_pred,
            "legacy_correct": legacy_pred == expected,
            "semantic_correct": semantic_pred == expected,
            "semantic_reason_codes": semantic.reason_codes,
            "semantic_confidence": semantic.confidence
        })

    return results


def run_benchmark() -> BenchmarkReport:
    """Run full benchmark."""
    print("Loading corpus...")
    cases = load_corpus()
    print(f"Loaded {len(cases)} cases")

    case_results = []
    categories = {}

    print("Running evaluations...")
    for case in cases:
        legacy = legacy_detector(case.text_a, case.text_b)
        semantic = semantic_detector(case.text_a, case.text_b)

        legacy_correct = legacy.label == case.expected_label
        semantic_correct = semantic.label == case.expected_label

        result = CaseResult(
            case_id=case.id,
            category=case.category,
            expected=case.expected_label,
            legacy_result=legacy,
            semantic_result=semantic,
            legacy_correct=legacy_correct,
            semantic_correct=semantic_correct
        )
        case_results.append(result)

        # track by category
        if case.category not in categories:
            categories[case.category] = {"total": 0, "legacy": 0, "semantic": 0}
        categories[case.category]["total"] += 1
        if legacy_correct:
            categories[case.category]["legacy"] += 1
        if semantic_correct:
            categories[case.category]["semantic"] += 1

    # compute category metrics
    category_metrics = []
    for cat, counts in sorted(categories.items()):
        legacy_acc = counts["legacy"] / counts["total"] if counts["total"] > 0 else 0
        semantic_acc = counts["semantic"] / counts["total"] if counts["total"] > 0 else 0
        improvement = semantic_acc - legacy_acc

        category_metrics.append(CategoryMetrics(
            category=cat,
            total_cases=counts["total"],
            legacy_correct=counts["legacy"],
            semantic_correct=counts["semantic"],
            legacy_accuracy=round(legacy_acc, 4),
            semantic_accuracy=round(semantic_acc, 4),
            improvement=round(improvement, 4)
        ))

    # overall metrics
    total = len(cases)
    legacy_total = sum(1 for r in case_results if r.legacy_correct)
    semantic_total = sum(1 for r in case_results if r.semantic_correct)

    print("Running SNLI sample...")
    snli_results = run_snli_sample()

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        corpus_version="1.0.0",
        total_cases=total,
        legacy_accuracy=round(legacy_total / total, 4) if total > 0 else 0,
        semantic_accuracy=round(semantic_total / total, 4) if total > 0 else 0,
        improvement=round((semantic_total - legacy_total) / total, 4) if total > 0 else 0,
        category_metrics=category_metrics,
        case_results=case_results,
        snli_sample_results=snli_results
    )

    return report


def serialize_report(report: BenchmarkReport) -> dict:
    """Convert report to JSON-serializable dict."""
    def convert(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        else:
            return obj

    return convert(report)


def print_summary(report: BenchmarkReport):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("CONTRADICTION DETECTION BENCHMARK RESULTS")
    print("="*60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Corpus version: {report.corpus_version}")
    print(f"Total cases: {report.total_cases}")
    print()
    print("OVERALL ACCURACY")
    print(f"  Legacy detector:   {report.legacy_accuracy*100:.1f}%")
    print(f"  Semantic detector: {report.semantic_accuracy*100:.1f}%")
    print(f"  Improvement:       {report.improvement*100:+.1f}%")
    print()
    print("ACCURACY BY CATEGORY")
    print("-"*60)
    print(f"{'Category':<20} {'Legacy':>10} {'Semantic':>10} {'Δ':>10}")
    print("-"*60)
    for m in report.category_metrics:
        print(f"{m.category:<20} {m.legacy_accuracy*100:>9.1f}% {m.semantic_accuracy*100:>9.1f}% {m.improvement*100:>+9.1f}%")
    print("-"*60)

    # SNLI sample summary
    snli_legacy_correct = sum(1 for r in report.snli_sample_results if r["legacy_correct"])
    snli_semantic_correct = sum(1 for r in report.snli_sample_results if r["semantic_correct"])
    snli_total = len(report.snli_sample_results)

    print()
    print("SNLI/MultiNLI/SICK SAMPLE RESULTS")
    print(f"  Total cases: {snli_total}")
    print(f"  Legacy correct:   {snli_legacy_correct}/{snli_total} ({snli_legacy_correct/snli_total*100:.1f}%)")
    print(f"  Semantic correct: {snli_semantic_correct}/{snli_total} ({snli_semantic_correct/snli_total*100:.1f}%)")
    print()


def main():
    report = run_benchmark()

    # save artifact
    output_path = PROJECT_ROOT / "results" / "contradiction_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(serialize_report(report), f, indent=2)

    print(f"Results saved to {output_path}")
    print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
