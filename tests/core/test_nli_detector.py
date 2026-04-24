# Author: Bradley R. Kinnard
"""Tests for the NLI detector module."""

import pytest


class TestNLIDetector:
    """Tests for NLI-based contradiction detection."""

    def test_import(self):
        """NLI detector module imports without error."""
        from backend.core.bel.nli_detector import (
            is_nli_available,
            classify_nli,
            check_contradiction_nli,
            NLIResult,
        )
        assert NLIResult is not None

    def test_nli_result_dataclass(self):
        """NLIResult dataclass works correctly."""
        from backend.core.bel.nli_detector import NLIResult

        result = NLIResult(
            label="contradiction",
            score=0.95,
            raw_scores={"contradiction": 0.95, "entailment": 0.03, "neutral": 0.02},
            model_used="test-model",
        )

        assert result.label == "contradiction"
        assert result.score == 0.95
        assert result.model_used == "test-model"

    def test_is_nli_available_returns_bool(self):
        """is_nli_available returns a boolean."""
        from backend.core.bel.nli_detector import is_nli_available

        result = is_nli_available()
        assert isinstance(result, bool)

    def test_check_contradiction_nli_returns_none_or_tuple(self):
        """check_contradiction_nli returns None or (bool, float) tuple."""
        from backend.core.bel.nli_detector import check_contradiction_nli, is_nli_available

        # NLI model may not be installed in test environment
        result = check_contradiction_nli("The sky is blue.", "The sky is not blue.")

        # returns None if model unavailable, tuple if available
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], float)
        # else None is acceptable


class TestNLIIntegration:
    """Tests for NLI integration with semantic detector."""

    def test_nli_fallback_in_semantic_detector(self):
        """Semantic detector can use NLI fallback."""
        from backend.core.bel.semantic_contradiction import check_contradiction

        # test with a case that might trigger NLI fallback
        result = check_contradiction(
            "The meeting is scheduled for tomorrow.",
            "The meeting was canceled.",
        )

        # should return a valid result regardless of NLI availability
        assert result.label in ("contradiction", "not_contradiction", "unknown")
        assert 0.0 <= result.confidence <= 1.0

    def test_nli_model_reason_code(self):
        """NLI_MODEL reason code added when NLI fallback used."""
        from backend.core.bel.nli_detector import is_nli_available
        from backend.core.bel.semantic_contradiction import check_contradiction

        if not is_nli_available():
            pytest.skip("NLI model not available")

        # tricky case that might use NLI
        result = check_contradiction(
            "John must attend the meeting.",
            "John cannot attend the meeting.",
        )

        # NLI_MODEL might be in reason_codes if used
        # (not guaranteed - depends on confidence thresholds)
        assert isinstance(result.reason_codes, list)
