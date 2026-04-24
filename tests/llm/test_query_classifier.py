# Author: Bradley R. Kinnard
"""Tests for the query classifier module."""

import pytest


class TestQueryClassifier:
    """Tests for zero-shot query classification."""

    def test_import(self):
        """Query classifier module imports without error."""
        from backend.llm.query_classifier import (
            classify_query,
            needs_real_time_info,
            is_classifier_available,
            RouteType,
        )
        assert RouteType is not None

    def test_classify_returns_tuple(self):
        """classify_query returns (route_type, confidence) tuple."""
        from backend.llm.query_classifier import classify_query

        result = classify_query("What is the weather like today?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        route, confidence = result
        assert route in ("real-time", "belief-grounded", "general")
        assert 0.0 <= confidence <= 1.0

    def test_weather_query_real_time(self):
        """Weather queries classified as real-time."""
        from backend.llm.query_classifier import classify_query

        route, _ = classify_query("What's the weather in New York right now?")

        # should be real-time (regex fallback at minimum)
        assert route == "real-time"

    def test_stock_query_real_time(self):
        """Stock price queries classified as real-time."""
        from backend.llm.query_classifier import classify_query

        route, _ = classify_query("What is the current stock price of Apple?")

        assert route == "real-time"

    def test_general_question(self):
        """General questions not classified as real-time."""
        from backend.llm.query_classifier import classify_query

        route, _ = classify_query("What is the capital of France?")

        # should NOT be real-time
        assert route in ("general", "belief-grounded")

    def test_needs_real_time_info(self):
        """needs_real_time_info convenience function."""
        from backend.llm.query_classifier import needs_real_time_info

        assert needs_real_time_info("What's the weather today?") is True
        assert needs_real_time_info("Tell me about your capabilities.") is False

    def test_caching(self):
        """Repeated queries use cache."""
        from backend.llm.query_classifier import classify_query

        query = "What is 2 + 2?"

        # call twice
        result1 = classify_query(query)
        result2 = classify_query(query)

        # should be identical (cached)
        assert result1 == result2

    def test_is_classifier_available(self):
        """is_classifier_available returns bool."""
        from backend.llm.query_classifier import is_classifier_available

        result = is_classifier_available()
        assert isinstance(result, bool)


class TestRegexFallback:
    """Tests for regex fallback when classifier unavailable."""

    def test_regex_patterns_work(self):
        """Regex patterns detect real-time queries."""
        from backend.llm.query_classifier import _regex_fallback

        # real-time patterns
        assert _regex_fallback("What's the weather today?") == "real-time"
        assert _regex_fallback("Look up the latest news") == "real-time"
        assert _regex_fallback("What time is it in Tokyo?") == "real-time"

        # general
        assert _regex_fallback("Hello, how are you?") == "general"
        assert _regex_fallback("Tell me a joke.") == "general"
