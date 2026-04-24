# Contributing to ABES

## Development Setup

```bash
git clone https://github.com/Aftermath-Technologies-Ltd/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

First run will download `all-MiniLM-L6-v2` (~80MB) for embeddings.

## Running Tests

```bash
# All tests
pytest tests/ -q

# Specific module
pytest tests/agents/test_perception.py -v

# With coverage (if installed)
pytest tests/ --cov=backend --cov-report=term-missing
```

All 821 tests should pass. If any fail, check Python version (requires 3.10+).

## License

By contributing, you agree that contributions are licensed under AGPL-3.0.
Derivative works must retain attribution described in [NOTICE](NOTICE).

## Code Style

This project follows strict conventions from [abes_spec_md.md](abes_spec_md.md) section 0.

### Naming

- `snake_case` for functions and variables
- `PascalCase` for classes
- Descriptive names: `belief_candidates` not `items`
- Verbs for actions: `compute_tension`, `apply_decay`
- Boolean functions: `is_active`, `has_conflict`

### Structure

- Functions under 50 lines
- Nesting under 3 levels
- Early returns to flatten logic
- Type hints on all public functions

### Docstrings

Concise, not novels:

```python
def apply_decay(self, belief: Belief) -> Optional[DecayEvent]:
    """
    Apply time-based decay to belief confidence.
    Returns DecayEvent if anything changed, None otherwise.
    """
```

### What NOT to do

- No `Any` types except at true API boundaries
- No `except Exception` blanket catches
- No commented-out code
- No TODO comments without tickets
- No "Manager" or "Handler" class names

## Adding Tests

Tests live in `tests/` mirroring the backend structure:

```
backend/agents/perception.py  →  tests/agents/test_perception.py
backend/rl/policy.py          →  tests/rl/test_policy.py
```

Use pytest fixtures. One concept per test. Clear names:

```python
@pytest.mark.asyncio
async def test_filters_pure_commands(self, agent):
    result = await agent.ingest("check the logs", {"source_type": "chat"})
    assert result == []
```

## Pull Request Process

1. Open an issue first to discuss
2. Branch from `main`
3. Run full test suite before submitting
4. One logical change per PR
5. Update CHANGELOG.md

## What's Not Implemented

Before working on something, check if it exists:

- `backend/api/` - empty, no HTTP routes
- `backend/benchmark/` - empty, no scenarios
- `backend/metrics/` - empty, no collection
- `frontend/` - placeholder stubs only
- `configs/` - empty
- `experiments/` - empty

The `BeliefEcologyLoop` class exists but has no tests.
