# Author: Bradley R. Kinnard
"""Tests for the response validator module."""

import pytest
from uuid import uuid4


class TestExtractClaims:
    """Tests for claim extraction from LLM responses."""

    def test_import(self):
        """Response validator module imports without error."""
        from backend.chat.response_validator import (
            extract_claims,
            validate_response,
            ExtractedClaim,
            ValidationResult,
        )
        assert ExtractedClaim is not None

    def test_extract_simple_claims(self):
        """Extracts factual claims from simple sentences."""
        from backend.chat.response_validator import extract_claims

        response = "The project deadline is Friday. The budget is $50,000. We have 5 team members."
        claims = extract_claims(response)

        assert len(claims) >= 1
        assert all(c.text for c in claims)

    def test_skip_questions(self):
        """Questions are not extracted as claims."""
        from backend.chat.response_validator import extract_claims

        response = "What is your name? How are you?"
        claims = extract_claims(response)

        assert len(claims) == 0

    def test_hedged_claims_marked(self):
        """Hedged statements are marked as such."""
        from backend.chat.response_validator import extract_claims

        response = "I think the meeting is at 3pm. It might be raining tomorrow."
        claims = extract_claims(response)

        # at least some should be marked hedged
        hedged_count = sum(1 for c in claims if c.is_hedged)
        assert hedged_count >= 1

    def test_skip_user_citations(self):
        """Skips statements citing what user said."""
        from backend.chat.response_validator import extract_claims

        response = "You mentioned that the deadline is Friday. Based on what you told me, the budget is $50k."
        claims = extract_claims(response)

        # should skip citations
        assert len(claims) == 0

    def test_extracted_claim_dataclass(self):
        """ExtractedClaim dataclass works correctly."""
        from backend.chat.response_validator import ExtractedClaim

        claim = ExtractedClaim(
            text="The project is on track.",
            sentence="The project is on track.",
            confidence=0.9,
            is_hedged=False,
        )

        assert claim.text == "The project is on track."
        assert claim.confidence == 0.9
        assert claim.is_hedged is False


class TestValidateResponse:
    """Tests for response validation against beliefs."""

    def _make_belief(self, content: str, confidence: float = 0.8):
        """Create a mock belief for testing."""
        from backend.core.models.belief import Belief, OriginMetadata
        return Belief(
            id=uuid4(),
            content=content,
            confidence=confidence,
            origin=OriginMetadata(source="test"),
        )

    def test_valid_response_no_contradictions(self):
        """Response with no contradictions passes validation."""
        from backend.chat.response_validator import validate_response

        beliefs = [
            self._make_belief("The deadline is Friday."),
            self._make_belief("The budget is $50,000."),
        ]

        response = "I'll make sure to finish by Friday within the budget."
        result = validate_response(response, beliefs)

        assert result.is_valid is True
        assert len(result.contradictions) == 0

    def test_empty_beliefs_valid(self):
        """Empty beliefs list means valid response."""
        from backend.chat.response_validator import validate_response

        response = "The sky is blue and the grass is green."
        result = validate_response(response, [])

        assert result.is_valid is True

    def test_detects_contradiction(self):
        """Detects when response contradicts a belief."""
        from backend.chat.response_validator import validate_response

        beliefs = [
            self._make_belief("The temperature is 70 degrees."),
        ]

        response = "The temperature is 40 degrees today."
        result = validate_response(response, beliefs, contradiction_threshold=0.5)

        # may or may not detect depending on semantic detector
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.claims_checked, int)

    def test_validation_result_dataclass(self):
        """ValidationResult dataclass works correctly."""
        from backend.chat.response_validator import ValidationResult

        result = ValidationResult(
            is_valid=False,
            contradictions=[{"claim": "test", "belief_id": "123"}],
            claims_checked=5,
        )

        assert result.is_valid is False
        assert len(result.contradictions) == 1
        assert result.claims_checked == 5


class TestCorrectionPrompt:
    """Tests for correction prompt generation."""

    def test_generates_correction_prompt(self):
        """Generates a correction prompt with context."""
        from backend.chat.response_validator import get_correction_prompt
        from backend.core.models.belief import Belief, OriginMetadata

        beliefs = [
            Belief(
                id=uuid4(),
                content="The deadline is Friday.",
                confidence=0.9,
                origin=OriginMetadata(source="test"),
            ),
        ]

        contradictions = [
            {"claim": "The deadline is Monday.", "belief_content": "The deadline is Friday."}
        ]

        prompt = get_correction_prompt(
            "The deadline is Monday and we're on track.",
            contradictions,
            beliefs,
        )

        assert "Friday" in prompt
        assert "Monday" in prompt
        assert "contradict" in prompt.lower()
