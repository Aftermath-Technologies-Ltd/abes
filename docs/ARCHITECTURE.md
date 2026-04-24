# ABES Architecture

ABES (Adaptive Belief Evolution System) is a FastAPI backend that maintains a structured set of beliefs, updates them in response to conversation input, and uses a reinforcement learning policy to guide mutation decisions. This document covers the belief lifecycle pipeline, external dependencies, and explicit non-goals.

---

## Belief Lifecycle Pipeline

Each conversation turn triggers one complete scheduler iteration. The 15 phases run in the fixed order defined in `AgentScheduler` (`backend/agents/scheduler.py`).

| # | Phase | What it does |
|---|-------|--------------|
| 1 | **Perception** | Parses raw text input and extracts candidate propositions for downstream creation. |
| 2 | **Creation** | Prompts the LLM to generate new `Belief` objects from the perceived candidates; assigns initial confidence and salience. |
| 3 | **Reinforcement** | Raises confidence and use-count on beliefs that align with new evidence; records supporting `EvidenceRef` entries. |
| 4 | **Decay** | Applies time-based confidence decay to all non-axiom beliefs; salience drops proportionally to idle duration. |
| 5 | **Contradiction** | Computes pairwise semantic-tension scores via NLI; populates `tension_map` and `contradiction_pairs` in the context. |
| 6 | **Mutation** | Clusters high-tension beliefs by token overlap; generates LLM-proposed revisions to reduce detected contradictions. |
| 7 | **Resolution** | Selects among competing resolutions using the `ResolutionStrategist`; applies the highest-scoring candidate. |
| 8 | **Relevance** | Scores each belief against the current turn using sentence embeddings; prunes beliefs whose relevance falls below threshold. |
| 9 | **RLPolicy** | Queries the trained MLP policy for an action recommendation (e.g., consolidate, mutate, retain); stores result in context. |
| 10 | **Consistency** | Cross-checks the updated belief set for logical consistency violations using the `ConsistencyChecker`. |
| 11 | **Safety** | Runs the `SafetyEnforcer` to flag or reject beliefs that breach content or factual-accuracy constraints. |
| 12 | **Baseline** | Optionally evaluates the same input through a plain LLM or append-only memory baseline for benchmarking. |
| 13 | **Narrative** | Produces a natural-language explanation of what changed this iteration and why, via `NarrativeExplainer`. |
| 14 | **Experiment** | Runs any configured experiment hooks (e.g., drift benchmarks, decay sweeps) and records results. |
| 15 | **Consolidation** | Computes fitness for all beliefs, runs population selection, calls `rebalance_tiers()` on the belief store to enforce tier caps, and deprecates overflow beliefs. |

---

## Belief Fitness and Tier Storage

Beliefs are scored by:

```
fitness = confidence * salience * max(use_count, 1)
```

Axiom-tagged beliefs receive a `1e6` bonus to make them eviction-resistant. The in-memory store maintains three tiers (L1 / L2 / L3) with configurable caps. When a tier overflows, the lowest-fitness beliefs are demoted to the next tier; L3 overflow results in `BeliefStatus.Deprecated`.

Default caps: L1=50, L2=200, L3=1000.

---

## RL Training

The policy is a small tanh MLP (`backend/rl/policy.py`). Training uses REINFORCE with discounted returns:

```
G_t = sum(gamma^k * r_{t+k})
```

Gradients are computed by manual backpropagation through the tanh layers and applied via gradient ascent on the flattened parameter vector. Training data comes from a `TrajectoryBuffer` (FIFO deque, max 10,000 transitions). The `train-rl` CLI command loads a saved buffer, runs N epochs, and optionally saves the updated policy.

---

## External Dependencies

| Dependency | Purpose |
|------------|---------|
| **DeBERTa NLI** (via `transformers`) | Contradiction detection in the Contradiction phase; produces entailment/neutral/contradiction logits for belief pairs. |
| **sentence-transformers** | Semantic similarity scoring in the Relevance phase; embeds beliefs and turn text into dense vectors for cosine comparison. |
| **Ollama / local LLM** | Belief creation in the Creation phase and mutation proposal in the Mutation phase; requires a running Ollama instance with a configured model. |
| **Gymnasium** | Defines the RL environment interface (`backend/rl/environment.py`); belief-state observation spaces and action spaces follow the Gymnasium API. |
| **NumPy** | Array operations throughout: fitness scoring, return discounting, gradient accumulation, trajectory buffer serialization. |
| **FastAPI + Uvicorn** | HTTP API layer; exposes belief CRUD, analytics, RL status, and chat endpoints. |
| **aiosqlite** | Async SQLite backend used by the snapshot store for persisting belief state to disk. |

---

## What ABES Is Not

**Not a standalone LLM.** ABES requires an external LLM (via Ollama) for belief creation and mutation. It does not include any language model weights.

**Not a replacement for a vector database.** Belief storage is in-memory with tier-capped eviction. It is not designed for billions of embeddings or nearest-neighbor retrieval at scale.

**Not a biological or cognitive simulation.** Decay, mutation, and fitness are engineering mechanisms borrowed loosely from formal belief revision theory (AGM postulates). There is no claim that the system models human cognition.

**Not a general-purpose knowledge base.** Beliefs are scoped to a single conversation session unless explicitly snapshotted. Cross-session persistence requires integrating the snapshot store.

---

## API Surface

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Submit a turn; triggers the full 15-phase scheduler iteration. |
| GET | `/beliefs` | List all active beliefs with confidence, salience, and tier. |
| POST | `/beliefs` | Manually inject a belief (bypasses the pipeline). |
| GET | `/beliefs/{id}` | Fetch a single belief by UUID. |
| DELETE | `/beliefs/{id}` | Remove a belief from the store. |
| GET | `/beliefs/fitness` | Sorted belief list with fitness scores and tier utilization stats. |
| GET | `/beliefs/clusters` | Tension-based belief clusters with mean tension, mean fitness, and member IDs. |
| GET | `/rl/status` | Current RL policy state: trained flag, trajectory count, last loss, action means. |
| GET | `/metrics` | Aggregate stats: belief counts by tier, decay rates, contradiction counts. |

---

## Directory Layout

```
backend/
  agents/         one module per AgentPhase, plus scheduler.py
  api/
    routes/       one file per route group (beliefs, chat, analytics, rl)
  core/
    bel/          belief revision logic: contradiction, decay, relevance, loop
    models/       Pydantic models: Belief, EvidenceRef, BeliefStatus
    config.py     settings loaded from environment
    deps.py       FastAPI dependency providers (belief store, RL state singletons)
  rl/
    policy.py     tanh MLP definition and parameter serialization
    training.py   REINFORCE train loop and metrics
    environment.py Gymnasium environment wrapping the belief store
    trajectory_buffer.py FIFO buffer with JSON persistence
  storage/
    in_memory.py  tiered belief store with fitness-based eviction
    snapshot.py   aiosqlite-backed persistence
  llm/            Ollama client and prompt builders
  metrics/        decay and drift metric collectors
tests/
  agents/         unit tests per agent module
  api/            FastAPI route tests using TestClient
  storage/        tier cap and eviction tests
  rl/             trajectory buffer and REINFORCE training tests
  test_agm_properties.py  AGM postulate verification (Success, Consistency, Minimal Change, Recovery)
docs/
  ARCHITECTURE.md  this file
  architecture.md  older diagram-centric overview (retained for diagrams)
  agents.md        per-agent behavior reference
  EVALUATIONS.md   benchmark methodology and results
```
