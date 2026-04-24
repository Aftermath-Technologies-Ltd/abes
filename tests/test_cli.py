# Author: Bradley R. Kinnard
"""Tests for the ABES CLI entrypoint."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from backend.cli import cli


runner = CliRunner()


def test_cli_version():
    """Version flag prints version string."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    """Help lists all commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("demo", "chat", "seed", "inspect", "verify-quick", "verify-determinism"):
        assert cmd in result.output


def test_demo_help():
    """Demo subcommand has correct options."""
    result = runner.invoke(cli, ["demo", "--help"])
    assert result.exit_code == 0
    assert "--headless" in result.output
    assert "--script" in result.output
    assert "--no-pause" in result.output
    assert "--with-decay" in result.output
    assert "--decay-hours" in result.output
    assert "--keep-alive" in result.output


def test_seed_help():
    """Seed subcommand has --file option."""
    result = runner.invoke(cli, ["seed", "--help"])
    assert result.exit_code == 0
    assert "--file" in result.output


def test_inspect_help():
    """Inspect subcommand has correct options."""
    result = runner.invoke(cli, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "--top" in result.output
    assert "--json-out" in result.output


def test_verify_quick_help():
    """Verify-quick subcommand has --prompts option."""
    result = runner.invoke(cli, ["verify-quick", "--help"])
    assert result.exit_code == 0
    assert "--prompts" in result.output


def test_verify_determinism_help():
    """Verify-determinism subcommand has correct options."""
    result = runner.invoke(cli, ["verify-determinism", "--help"])
    assert result.exit_code == 0
    assert "--prompts" in result.output
    assert "--runs" in result.output


def test_seed_missing_file():
    """Seed fails cleanly when given nonexistent file."""
    result = runner.invoke(cli, ["seed", "--file", "/tmp/nonexistent_seed.json"])
    assert result.exit_code != 0


def test_demo_missing_script():
    """Demo fails cleanly when given nonexistent script."""
    result = runner.invoke(cli, ["demo", "--script", "/tmp/nonexistent.json"])
    assert result.exit_code != 0


def test_demo_conversation_file_valid():
    """Demo conversation JSON is well-formed and has required fields."""
    path = Path(__file__).resolve().parent.parent / "examples" / "demo_conversation.json"
    assert path.exists(), f"Missing {path}"

    with open(path) as f:
        turns = json.load(f)

    assert isinstance(turns, list)
    assert len(turns) >= 10, "Demo needs at least 10 turns"

    for i, turn in enumerate(turns):
        assert "message" in turn, f"Turn {i} missing 'message'"
        assert isinstance(turn["message"], str)
        assert len(turn["message"]) > 0


def test_seed_beliefs_file_valid():
    """Seed beliefs JSON is well-formed and has required fields."""
    path = Path(__file__).resolve().parent.parent / "examples" / "seed_beliefs.json"
    assert path.exists(), f"Missing {path}"

    with open(path) as f:
        beliefs = json.load(f)

    assert isinstance(beliefs, list)
    assert len(beliefs) >= 3, "Seed file needs at least 3 beliefs"

    for i, b in enumerate(beliefs):
        assert "content" in b, f"Belief {i} missing 'content'"
        assert isinstance(b["content"], str)
        assert len(b["content"]) > 0
        if "confidence" in b:
            assert 0.0 <= b["confidence"] <= 1.0


def test_seed_no_backend(monkeypatch):
    """Seed exits cleanly when backend is not running."""
    monkeypatch.setattr("backend.cli._backend_running", lambda *a: False)
    result = runner.invoke(cli, ["seed"])
    assert result.exit_code != 0
    assert "not running" in result.output.lower()


def test_inspect_no_backend(monkeypatch):
    """Inspect exits cleanly when backend is not running."""
    monkeypatch.setattr("backend.cli._backend_running", lambda *a: False)
    result = runner.invoke(cli, ["inspect"])
    assert result.exit_code != 0
    assert "not running" in result.output.lower()


def test_verify_quick_no_backend(monkeypatch):
    """Verify-quick exits cleanly when backend is not running."""
    monkeypatch.setattr("backend.cli._backend_running", lambda *a: False)
    result = runner.invoke(cli, ["verify-quick"])
    assert result.exit_code != 0
    assert "not running" in result.output.lower()
