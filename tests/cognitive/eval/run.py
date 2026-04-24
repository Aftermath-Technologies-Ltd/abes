# Author: Bradley R. Kinnard
"""
CLI entry point for running the cognitive evaluation suite.

Usage:
    python -m tests.cognitive.eval.run               # full 1000-prompt run
    python -m tests.cognitive.eval.run --domains episodic_memory reasoning
    python -m tests.cognitive.eval.run --max 50       # quick smoke test
    python -m tests.cognitive.eval.run --dry-run      # just validate prompts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .harness import EvalHarness
from .prompt_bank import build_prompts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ABES 1000-prompt cognitive evaluation suite"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="ABES backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Restrict to specific domains (e.g., episodic_memory reasoning)",
    )
    parser.add_argument(
        "--constructs", nargs="+", default=None,
        help="Restrict to specific constructs (e.g., modus_ponens false_belief)",
    )
    parser.add_argument(
        "--max", type=int, default=None, dest="max_prompts",
        help="Max prompts to run (useful for smoke tests)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Cosine similarity pass threshold (default: 0.70)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent session groups (default: 4)",
    )
    parser.add_argument(
        "--decay-cycles", type=int, default=5,
        help="Decay iterations between long-horizon prompts (default: 5)",
    )
    parser.add_argument(
        "--output-dir", default="results/cognitive_eval",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate prompt bank without running against the API",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.dry_run:
        return _dry_run()

    harness = EvalHarness(
        base_url=args.base_url,
        threshold=args.threshold,
        concurrency=args.concurrency,
        decay_cycles=args.decay_cycles,
    )

    result = asyncio.run(harness.run(
        domains=args.domains,
        constructs=args.constructs,
        max_prompts=args.max_prompts,
        output_dir=args.output_dir,
    ))

    stats = result["stats"]
    return 0 if stats.pass_rate >= 0.70 else 1


def _dry_run() -> int:
    """Validate prompt bank structure without hitting the API."""
    from collections import Counter

    prompts = build_prompts()
    domains = Counter(p.domain for p in prompts)
    constructs = Counter(p.construct for p in prompts)
    ids = [p.id for p in prompts]

    errors: list[str] = []

    if len(prompts) != 1000:
        errors.append(f"Expected 1000 prompts, got {len(prompts)}")

    if len(set(ids)) != len(ids):
        errors.append(f"Duplicate IDs: {len(ids)} total, {len(set(ids))} unique")

    for d, count in domains.items():
        if count != 125:
            errors.append(f"Domain '{d}' has {count} prompts (expected 125)")

    # check all probes have gold answers
    probes_without_gold = [
        p for p in prompts if not p.is_setup and not p.gold_answer
    ]
    if probes_without_gold:
        errors.append(
            f"{len(probes_without_gold)} probe(s) missing gold answers: "
            f"IDs {[p.id for p in probes_without_gold[:5]]}"
        )

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"Prompt bank validated: {len(prompts)} prompts OK")
    print(f"  Domains ({len(domains)}): {', '.join(sorted(domains))}")
    print(f"  Constructs ({len(constructs)}): {len(constructs)} total")
    print(f"  Setup prompts: {sum(1 for p in prompts if p.is_setup)}")
    print(f"  Probe prompts: {sum(1 for p in prompts if not p.is_setup)}")
    print(f"  Long-horizon: {sum(1 for p in prompts if p.horizon == 'long_horizon')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
