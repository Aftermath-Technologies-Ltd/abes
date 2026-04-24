# ABES: Structured Belief Revision & Dynamic Epistemic Memory for Autonomous Agents

<p align="center">
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+" /></a>
  <a href="docs/EVALUATIONS.md"><img src="https://img.shields.io/badge/cognitive%20eval-825%2F1000-blue.svg" alt="Cognitive Eval" /></a>
  <a href="docs/side_by_side_eval.md"><img src="https://img.shields.io/badge/side--by--side%20eval-14%2F15-blue.svg" alt="Side-by-Side Eval" /></a>
</p>

ABES is a structured belief revision backend for autonomous agents. It ingests natural language inputs, extracts and stores discrete beliefs, and manages them through a 15-phase agent pipeline covering confidence scoring, contradiction detection, salience decay, and evidence tracking.

## Quick Navigation

- [Install & Quickstart](#install--quickstart)
- [Architecture Overview](#architecture-overview)
- [Usage](#usage)
- [API Reference Summary](#api-reference-summary)
- [Eval Results Summary](#eval-results-summary)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Install & Quickstart

### Requirements

- Python 3.10+
- Node.js 18+ (optional debug UI only)
- Ollama (optional, for LLM-backed mutation and chat)

External models used:
- `cross-encoder/nli-deberta-v3-base` — contradiction detection via NLI
- `sentence-transformers/all-MiniLM-L6-v2` — embedding similarity and scoring
- Ollama or compatible LLM provider — mutation, narrative, and chat generation

### Local setup

```bash
git clone https://github.com/Aftermath-Technologies-Ltd/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Start the backend

```bash
# Terminal 1
PYTHONPATH=$PWD uvicorn backend.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — add a belief
curl -X POST http://localhost:8000/beliefs \
  -H "Content-Type: application/json" \
  -d '{"content": "System target is alpha-node-4", "confidence": 0.9, "source": "agent"}'

# Retrieve all beliefs
curl http://localhost:8000/beliefs | python3 -m json.tool
```

### Optional debug UI

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose up
docker compose up --profile ui
docker compose up --profile llm --profile ui
STORAGE_BACKEND=sqlite docker compose up
```

---

## Architecture Overview

ABES is a backend-first service with an optional TypeScript debug UI. The core is a belief store (in-memory or SQLite) managed by a 15-phase agent pipeline.

Core flow:
1. Agents or chat clients send payloads via REST or WebSocket.
2. The scheduler runs 15 ordered phases over the current belief set.
3. Beliefs are updated in the store.
4. The relevance stack is selected and ranked for response generation.
5. Safety and response validation run before output is returned.

### Belief Pipeline (15 Phases)

| Phase | Agent | What it does |
|-------|-------|-------------|
| 1 | Perception | Extracts candidate belief statements from raw input |
| 2 | Creation | Creates new Belief records from extracted candidates |
| 3 | Reinforcement | Increases confidence and salience on corroborating evidence |
| 4 | Decay | Applies time-based confidence and salience decay |
| 5 | Contradiction | Detects contradicting pairs via rule-based analysis + NLI fallback |
| 6 | Mutation | Proposes revised beliefs for high-tension, low-confidence entries |
| 7 | Resolution | Applies resolution strategies to contradiction pairs |
| 8 | Relevance | Scores and ranks beliefs by cosine similarity to current context |
| 9 | RL Policy | Applies learned parameter adjustments to control variables |
| 10 | Consistency | Checks belief state invariants |
| 11 | Safety | Detects prompt injection and system prompt extraction attempts |
| 12 | Baseline | Bridges to baseline memory implementations for comparison |
| 13 | Narrative | Generates summaries of belief state changes |
| 14 | Experiment | Collects telemetry for evaluation runs |
| 15 | Consolidation | Merges near-duplicate beliefs; prunes low-salience orphans |

### Belief Model Fields

- `id`, `content`, `confidence`, `tension`, `salience`
- `status`: `active`, `decaying`, `dormant`, `mutated`, `deprecated`
- `is_axiom`: immutable, immune to decay and deprecation
- `memory_tier`: `L1` (working), `L2` (episodic), `L3` (deep)
- `half_life_days`, `evidence_for`, `evidence_against`, `evidence_balance`
- `links`, `parent_id`, `user_id`, `session_id`, `origin`

Key formulas:
- Salience decay: `s(t) = s0 * 0.5^(elapsed_hours / (half_life_days * 24))`
- Confidence update: `posterior = 0.7 * evidence_weight + 0.3 * prior_confidence`
- Stack ranking: weighted score over confidence, relevance, salience, recency, and tension

---

## Usage

### CLI commands

| Command | Description |
|---------|-------------|
| `abes demo` | Run scripted ingestion demo (12-turn sequence) |
| `abes chat` | Launch backend |
| `abes seed` | Load seed beliefs from JSON |
| `abes inspect` | Show current belief state |
| `abes verify-quick` | Run cognitive smoke test |
| `abes verify-determinism` | Compare repeated runs for reproducibility |

```bash
abes demo --headless
abes demo --headless --with-decay --decay-hours 12
abes inspect --json-out | jq .
abes verify-quick --prompts 200
abes verify-determinism --runs 5
```

---

## API Reference Summary

| Route | Method | Description |
|-------|--------|-------------|
| `/beliefs` | GET | List beliefs with optional status/tag filtering |
| `/beliefs` | POST | Create a new belief |
| `/beliefs/{id}` | GET | Get single belief |
| `/beliefs/{id}/ecology` | GET | Full belief state including evidence ledger and links |
| `/beliefs/fitness` | GET | Beliefs ranked by fitness with tier utilization stats |
| `/beliefs/clusters` | GET | Cluster summaries with tension and fitness metrics |
| `/chat/` | POST | Ingest a message through the full pipeline |
| `/clusters` | GET | List semantic clusters |
| `/rl/status` | GET | RL policy status and training metrics |
| `/bel/stack` | GET | Current ranked belief stack for a context query |
| `/snapshots` | GET | List belief state snapshots |
| `/docs` | GET | Interactive OpenAPI documentation |

---

## Eval Results Summary

### 1000-Prompt Cognitive Evaluation

**Result: 825/1000 (82.5%), 95% CI [0.800, 0.848]** — tested on Llama 3.1 8B, single model, self-reported. See [docs/EVALUATIONS.md](docs/EVALUATIONS.md) for full breakdown, failure analysis, and planned baselines.

| Domain | Score |
|--------|-------|
| Episodic Memory | 96.8% |
| Working Memory | 94.4% |
| Semantic Memory | 92.8% |
| Selective Attention | 85.6% |
| Self-Correction | 82.4% |
| Reasoning | 79.2% |
| Language Comprehension | 70.4% |
| Social Cognition | 58.4% |

### Side-by-Side: Structured Memory Layer vs. Stateless LLM

This comparison demonstrates the value of a structured memory layer over a stateless LLM. ABES and raw Ollama receive identical prompts; the score difference reflects what persistent belief management adds, not model quality.

| Metric | ABES (structured memory) | Ollama (stateless) |
|--------|--------------------------|---------------------|
| Blocks passed | 14/15 | 6/15 |
| Contradiction detection | Rule-based + NLI | Context window only |
| Session isolation | Zero cross-session leakage | No memory |
| Safety (prompt injection) | 0 leaks across 5 vectors | 3 leaks |

Full protocol: [docs/side_by_side_eval.md](docs/side_by_side_eval.md) | Results: [results/side_by_side_eval.json](results/side_by_side_eval.json)

---

## Limitations

- Modality and temporal contradiction detection are weaker than quantifier and numeric rules.
- Social cognition and moral reasoning scores are primarily limited by LLM refusal behavior in the underlying model, not belief-layer mechanics.
- In-memory storage is the default; use SQLite (`STORAGE_BACKEND=sqlite`) for persistence across restarts.
- Multi-agent concurrency has not been load-tested.
- All evaluations used a single model (Llama 3.1 8B). No external baselines (Mem0, LangGraph MemoryStore, vector DB + reranker) have been benchmarked yet.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

Copyright (C) 2026 Bradley R. Kinnard. All Rights Reserved.

Attribution requirements: [NOTICE](NOTICE). Citation metadata: [CITATION.cff](CITATION.cff).
