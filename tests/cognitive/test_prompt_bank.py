# Author: Bradley R. Kinnard
"""
Pytest integration for the cognitive evaluation suite.

Runs the full 1000-prompt eval as a single pytest test (marks slow).
Also provides smaller parametrized test for prompt bank validation.
"""

from __future__ import annotations

import pytest

from tests.cognitive.eval.prompt_bank import build_prompts


class TestPromptBank:
    """Validate prompt bank structure without hitting the API."""

    def test_total_count(self):
        prompts = build_prompts()
        assert len(prompts) == 1000, f"Expected 1000, got {len(prompts)}"

    def test_unique_ids(self):
        prompts = build_prompts()
        ids = [p.id for p in prompts]
        assert len(set(ids)) == len(ids), "Duplicate prompt IDs"

    def test_domain_balance(self):
        from collections import Counter
        prompts = build_prompts()
        domains = Counter(p.domain for p in prompts)
        for d, count in domains.items():
            assert count == 125, f"Domain '{d}' has {count} (expected 125)"

    def test_eight_domains(self):
        prompts = build_prompts()
        domains = set(p.domain for p in prompts)
        expected = {
            "episodic_memory", "semantic_memory", "working_memory",
            "selective_attention", "language_comprehension", "reasoning",
            "social_cognition", "self_correction",
        }
        assert domains == expected

    def test_probes_have_gold_answers(self):
        prompts = build_prompts()
        probes = [p for p in prompts if not p.is_setup]
        missing = [p.id for p in probes if not p.gold_answer]
        assert not missing, f"Probes missing gold answers: {missing[:10]}"

    def test_long_horizon_prompts_exist(self):
        prompts = build_prompts()
        lh = [p for p in prompts if p.horizon == "long_horizon"]
        assert len(lh) >= 30, f"Expected >=30 long-horizon prompts, got {len(lh)}"
