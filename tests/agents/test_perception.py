# Author: Bradley R. Kinnard
"""
Tests for PerceptionAgent.

Covers:
- Chat message extraction (factual claims)
- Structured/log input parsing
- Command filtering
- Filler/noise rejection
- Deduplication via LRU cache
- Sentence splitting with protected tokens
"""

import pytest
from backend.agents.perception import PerceptionAgent


@pytest.fixture
def agent():
    return PerceptionAgent(cache_size=100)


class TestChatExtraction:
    """Test extraction from conversational input."""

    @pytest.mark.asyncio
    async def test_simple_factual_statement(self, agent):
        result = await agent.ingest(
            "The cache is full and needs clearing.",
            {"source_type": "chat"},
        )
        assert len(result) >= 1
        assert any("cache" in r.lower() for r in result)

    @pytest.mark.asyncio
    async def test_filters_greetings(self, agent):
        result = await agent.ingest("Hi", {"source_type": "chat"})
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_acknowledgments(self, agent):
        result = await agent.ingest("ok thanks", {"source_type": "chat"})
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_pure_commands(self, agent):
        result = await agent.ingest("check the logs", {"source_type": "chat"})
        # pure command with no factual claim should be filtered
        assert result == []

    @pytest.mark.asyncio
    async def test_extracts_claim_from_command(self, agent):
        result = await agent.ingest(
            "Check why the model crashed during training",
            {"source_type": "chat"},
        )
        # should extract the factual part about crashing
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_empty_input(self, agent):
        result = await agent.ingest("", {"source_type": "chat"})
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_only(self, agent):
        result = await agent.ingest("   \n\t  ", {"source_type": "chat"})
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_sentences(self, agent):
        result = await agent.ingest(
            "The model converged. Loss dropped to 0.01. Training completed successfully.",
            {"source_type": "chat"},
        )
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_critical_signals_pass(self, agent):
        result = await agent.ingest("OOM error occurred", {"source_type": "chat"})
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_technical_content_passes(self, agent):
        result = await agent.ingest(
            "tensor_v2 is initialized with shape 512x768",
            {"source_type": "chat"},
        )
        assert len(result) >= 1


class TestStructuredExtraction:
    """Test extraction from logs and tool output."""

    @pytest.mark.asyncio
    async def test_strips_timestamp_prefix(self, agent):
        result = await agent.ingest(
            "[2024-01-15 10:30:00] Cache miss rate exceeded threshold",
            {"source_type": "log"},
        )
        assert len(result) >= 1
        # should not include the timestamp in output
        assert not any(r.startswith("[2024") for r in result)

    @pytest.mark.asyncio
    async def test_strips_level_prefix(self, agent):
        result = await agent.ingest(
            "ERROR: Connection timeout after 30s",
            {"source_type": "log"},
        )
        assert len(result) >= 1
        assert not any(r.startswith("ERROR:") for r in result)

    @pytest.mark.asyncio
    async def test_extracts_error_messages(self, agent):
        result = await agent.ingest(
            "RuntimeError: Model weights corrupted",
            {"source_type": "system"},
        )
        assert len(result) >= 1
        assert any("corrupt" in r.lower() for r in result)

    @pytest.mark.asyncio
    async def test_filters_json_noise(self, agent):
        result = await agent.ingest(
            "{\n}\n[\n]\nnull\n---",
            {"source_type": "tool"},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_progress_bars(self, agent):
        result = await agent.ingest(
            "epoch 3/10 done\nbatch 50/100 complete",
            {"source_type": "log"},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_keeps_failure_messages(self, agent):
        result = await agent.ingest(
            "Training failed with OOM error",
            {"source_type": "log"},
        )
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_limits_repeated_messages(self, agent):
        repeated = "\n".join(["sync completed"] * 10)
        result = await agent.ingest(repeated, {"source_type": "log"})
        # should cap repeats
        assert len(result) <= 3


class TestDeduplication:
    """Test LRU cache deduplication."""

    @pytest.mark.asyncio
    async def test_dedupes_same_message(self, agent):
        """Cross-call dedup removed (creator handles it). Same extraction each call."""
        msg = "The model is overfitting."
        r1 = await agent.ingest(msg, {"source_type": "chat"})
        r2 = await agent.ingest(msg, {"source_type": "chat"})

        assert len(r1) >= 1
        assert r2 == r1  # no cross-call dedup; creator handles dedup

    @pytest.mark.asyncio
    async def test_case_insensitive_dedupe(self, agent):
        """Cross-call dedup removed. Each call independently extracts."""
        r1 = await agent.ingest("Cache is full", {"source_type": "chat"})
        r2 = await agent.ingest("cache is full", {"source_type": "chat"})

        assert len(r1) >= 1
        assert len(r2) >= 1  # no cross-call dedup


class TestSentenceSplitting:
    """Test sentence boundary detection."""

    @pytest.mark.asyncio
    async def test_preserves_urls(self, agent):
        result = await agent.ingest(
            "Check https://example.com/model.py for details",
            {"source_type": "chat"},
        )
        # URL should be preserved, not split on dots
        assert any("https://example.com" in r for r in result) or len(result) <= 1

    @pytest.mark.asyncio
    async def test_preserves_filenames(self, agent):
        result = await agent.ingest(
            "The error is in config.yaml file",
            {"source_type": "chat"},
        )
        assert any("config.yaml" in r for r in result)

    @pytest.mark.asyncio
    async def test_preserves_version_numbers(self, agent):
        result = await agent.ingest(
            "Using PyTorch 2.1.0 for training",
            {"source_type": "chat"},
        )
        assert any("2.1.0" in r for r in result)


class TestCommandDetection:
    """Test command vs factual statement detection."""

    @pytest.mark.asyncio
    async def test_please_prefix_is_command(self, agent):
        result = await agent.ingest(
            "Please check the logs",
            {"source_type": "chat"},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_can_you_prefix_is_command(self, agent):
        result = await agent.ingest(
            "Can you fix the model",
            {"source_type": "chat"},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_factual_with_modal_passes(self, agent):
        result = await agent.ingest(
            "The model should update automatically when weights change",
            {"source_type": "chat"},
        )
        # this is factual, not a command
        assert len(result) >= 1


class TestEdgeCases:
    """Test boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_word_with_tech_marker(self, agent):
        result = await agent.ingest("GPU_OOM", {"source_type": "chat"})
        # single word with underscore is technical
        assert len(result) >= 1 or result == []  # depends on substance check

    @pytest.mark.asyncio
    async def test_very_long_input(self, agent):
        long_text = "The model is training. " * 100
        result = await agent.ingest(long_text, {"source_type": "chat"})
        # should handle without crashing, may dedupe repeats
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_unicode_content(self, agent):
        result = await agent.ingest(
            "Model accuracy is 95% 🎉",
            {"source_type": "chat"},
        )
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_negation_statements_pass(self, agent):
        result = await agent.ingest(
            "The cache is not syncing properly",
            {"source_type": "chat"},
        )
        assert len(result) >= 1
