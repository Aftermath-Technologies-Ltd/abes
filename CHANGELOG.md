# Changelog

All notable changes to ABES are documented here.

## [0.5.0] - 2026-03-04

### Added

**200-Prompt Cognitive Stress Test**
- End-to-end test sending 200 graded prompts through the full ABES pipeline
- 16 categories: identity, belief creation, reinforcement, deduplication, noise rejection, memory recall, context relevance, contradiction, safety, multi-fact, general knowledge, session isolation, coherence, edge cases, lifecycle
- 3 consecutive runs at 200/200 (100.0%) on Llama 3.1 8B
- Source: `tests/cognitive/test_200_stress.py` (1,134 lines)
- Artifact: `results/stress_test_200_results.json`

**Response-Level Safety Sanitizer**
- `ChatService._sanitize_response()` catches system prompt leaks post-LLM
- 9 marker strings detected (IMPORTANT RULES, belief_context, CONFIDENTIAL, etc.)
- Leaked responses replaced with safe refusal message

**System Prompt Hardening**
- Security instruction moved to first line of prompt template (before AI identity)
- Explicit refusal template for extraction attempts
- Covers: "output raw prompt", "repeat above", "show config", etc.

**Perception Noise Filters**
- Gibberish detector (repeated characters, 3+ word repetition)
- Code/SQL injection filter
- Filler phrase rejection ("hmm", "let me think", etc.)
- Missing verb patterns: `find`, `collect`, `picked`, `bought`, `sold`, `adopted`, `started`
- Adverb gap support: "I recently started", "I just picked up"

### Fixed

**Decay From Epoch (Critical)**
- `last_reinforced` defaulted to epoch (1970-01-01) for new beliefs
- Decay computed `confidence × 0.99^490000 ≈ 0`, instantly depreciating every new belief
- Fix: anchor = `max(last_reinforced, created_at)`

**Cross-User Perception Cache (Critical)**
- Global `_seen` cache silently dropped identical messages from different users
- Fix: scoped dedup to per-call only; creator handles cross-call deduplication

**Bayesian Update Bypassing Confidence Cap**
- `_bayes_update()` pushed confidence to 0.955, bypassing 0.95 ceiling
- Fix: capped output at `min(0.95, ...)`

**Reinforcement Ceiling Off-by-One**
- Guard used `>=` (beliefs AT ceiling skipped); should use `>` (only ABOVE ceiling)
- Beliefs at exactly 0.95 could never be reinforced again
- Fix: strict `>` comparison

**Response Validator False Positives**
- General knowledge answers flagged as contradictions with personal beliefs
- Fix: reduced threshold sensitivity for factual/non-personal responses

### Changed

- Total unit tests: 798 → 801 (3 new from test alignment)
- Aligned 14 existing tests with new behavior:
  - Decay tests: `created_at` parameter for anchor logic (7 tests)
  - Perception tests: removed cross-call dedup assertions (2 tests)
  - Reinforcement tests: ceiling and cooldown assertions (2 tests)
  - API route tests: in-memory store injection for isolation (2 tests)
  - LLM mutation test: environment variable isolation (1 test)

---

## [0.4.0] - 2026-03-03

### Added

**Belief Ecology Extensions**
- `salience` field (0.0–1.0) on Belief model — attentional energy with exponential half-life decay
- `half_life_days` field — configurable per-belief salience decay rate (default 7.0)
- `decay_salience()` and `boost_salience()` methods on Belief
- `EvidenceRef` model with `content`, `direction` (supports/attacks), `weight`, `source_id`
- Evidence ledger: `evidence_for` / `evidence_against` lists per belief
- `add_evidence()` with naive Bayes confidence update (70% evidence / 30% prior)
- `evidence_balance` property: sum(support weights) − sum(attack weights)
- `BeliefLink` model with `target_id`, `relation` (reinforces/contradicts), `weight`
- `add_link()` / `get_links()` on Belief for graph edge management
- `BeliefStatus.Dormant` — beliefs with near-zero salience hibernate instead of dying
- `hibernate()` / `reawaken()` methods for dormancy transitions

**ConsolidationAgent (Agent 16)**
- Merges near-duplicate beliefs within clusters (>0.92 cosine similarity)
- Winner absorbs loser's evidence ledger and graph edges
- Compresses lineage chains deeper than 5 — tip absorbs ancestor evidence
- Added as 15th pipeline phase in agent scheduler

**Belief Stack Selection**
- `select_belief_stack()` in `backend/core/bel/stack.py`
- Weighted scoring: 35% salience + 30% relevance + 20% recency + 15% graph spread
- `compete_for_attention()` hibernates losers when ecology exceeds 4× stack size
- Chat service uses stack selection for LLM context window

**87 new tests across 8 test files**
- `test_belief_ecology_extensions.py` (40 tests) — salience, evidence, links, dormancy
- `test_belief_stack.py` (13 tests) — stack selection and competition
- `test_ranking_salience.py` (6 tests) — salience weight in ranking
- `test_consolidation.py` (8 tests) — merge and lineage compression
- `test_decay_salience.py` (7 tests) — salience decay and dormancy transitions
- `test_scheduler_consolidation.py` (5 tests) — consolidation phase in scheduler
- `test_tension_formula.py` (4 tests) — confidence-weighted tension
- `test_reinforcement_ecology.py` (4 tests) — salience boost, evidence, graph edges

### Changed

**Tension Formula Refinement**
- Old: `similarity × (0.5 + 0.5 × opposition)`
- New: `similarity × avg_confidence × (0.5 + 0.5 × opposition)`
- High-confidence contradictions now produce proportionally more tension

**Ranking Weights Rebalanced**
- Added salience as ranking dimension (weight 0.20)
- New: confidence 0.30, relevance 0.30, salience 0.20, recency 0.10, tension 0.10
- Old: confidence 0.40, relevance 0.40, recency 0.10, tension 0.10

**Reinforcement Agent Enhanced**
- Now boosts salience by 0.1 on reinforcement
- Adds `EvidenceRef(direction="supports")` to reinforced beliefs
- Adds `reinforces` graph edges between co-reinforced beliefs

**Decay Controller Enhanced**
- Now decays salience via half-life formula alongside confidence decay
- `DecayEvent` includes `old_salience` / `new_salience` fields
- Triggers dormancy when salience drops below threshold (default 0.05)

**Agent Scheduler**
- Expanded from 14 to 15 phases (added Consolidation as final phase)

- Test count increased from 698 to **798 passing tests**

---

## [0.3.0] - 2026-01-30

### Added

**Chat Interface with LLM Integration**
- `ChatService` orchestrating belief pipeline for conversational AI
- `OllamaProvider` for local LLM integration (llama3.1 8B)
- WebSocket support for real-time belief event streaming
- Chat API endpoints: `/chat/message`, `/chat/sessions`, `/chat/ws`
- Session management with history persistence

**Dashboard Hub**
- Central dashboard at `/` with service navigation cards
- Service cards: Chat (active), Documents (coming soon), Explorer (coming soon), Integrations (coming soon)
- System status indicator and quick actions
- Stats overview panel

**Modern Frontend UI**
- Clean dark theme (black/grey, Grok-style)
- 3-panel chat layout: sidebar, chat area, belief activity panel
- Real-time belief activity display (created, reinforced, updated, tensions)
- Responsive design with minimal semantic colors

**Perception Agent Enhancements**
- `_has_chat_substance()` for conversational content extraction
- Support for personal facts, opinions, named entities
- Protection for name abbreviations (R., Mr., Dr., etc.)
- Fixed sentence splitting for names like "Bradley R. Kinnard"

**LLM Context Improvements**
- Belief-to-user-perspective transformation
- "My name is Brad" → "User's name is Brad" in LLM context
- Improved system prompt clarifying facts are ABOUT the user
- Prevents LLM from confusing itself with user

### Changed
- Updated architecture diagram to include Frontend, API, and LLM layers
- Expanded module map with new chat/, llm/, and frontend/ directories

---

## [0.2.0] - 2026-01-30

### Added

**Comprehensive Test Coverage for Previously Untested Modules**
- `BeliefEcologyLoop` (40 tests) — full 7-step iteration cycle
  - Decay computation and status transitions
  - Contradiction/tension detection with negation heuristics
  - Relevance scoring and belief ranking
  - Snapshot creation with edge relationships
  - Ecological action triggering
- `SnapshotTimeline` (10 tests) — replay functionality and key moment detection
- `RLBELIntegration` (24 tests) — RL-to-BEL policy application
- `snapshot_queries` (24 tests) — query functions for snapshot retrieval

**Model Enhancement**
- Added `relevance` and `score` fields to `Belief` model for ranking support

### Fixed
- `Belief` model now supports dynamic relevance/score attributes needed by BEL loop

### Changed
- Test count increased from 534 to **632 passing tests**

---

## [0.1.0] - 2026-01-30

### Added

**Core Models**
- `Belief` model with confidence, tension, status, tags, lineage tracking
- `Snapshot` model capturing ecology state with edge relationships
- `BeliefSnapshot` frozen belief state for snapshots
- Event models: `BeliefCreatedEvent`, `BeliefReinforcedEvent`, etc.

**15 Specialized Agents**
- `PerceptionAgent` — extracts claims from raw input
- `BeliefCreatorAgent` — creates beliefs with embedding-based deduplication
- `ReinforcementAgent` — boosts confidence on similar evidence
- `ContradictionAuditorAgent` — computes pairwise tensions via embeddings
- `MutationEngineerAgent` — proposes hedged/conditional belief variants
- `ResolutionStrategistAgent` — integrates, splits, or deprecates conflicts
- `RelevanceCuratorAgent` — ranks beliefs by weighted formula
- `DecayControllerAgent` — applies time-based confidence decay
- `BaselineMemoryBridgeAgent` — interfaces with RAG/chat history
- `RLPolicyAgent` — outputs actions from ecology state (heuristic fallback)
- `RewardShaperAgent` — computes shaped reward signals
- `ExperimentOrchestratorAgent` — runs scripted scenarios
- `ConsistencyCheckerAgent` — probes for answer drift
- `NarrativeExplainerAgent` — generates human-readable explanations
- `SafetySanityAgent` — enforces limits, vetoes dangerous actions

**Agent Scheduler**
- 14-phase execution order per spec 4.2
- Conditional execution support (`run_every_n`, enable/disable)
- `AgentProtocol` and `AgentResult` for uniform interface

**RL Layer**
- `BeliefEcologyEnv` — 15-dim state, 7-dim action, episodic
- `MLPPolicy` — pure NumPy MLP with tanh activations
- `EvolutionStrategy` — gradient-free optimizer
- `ESTrainer` — training loop with checkpointing

**Storage**
- `BeliefStoreABC` and `SnapshotStoreABC` abstract interfaces
- `InMemoryBeliefStore` and `InMemorySnapshotStore` implementations
- Snapshot compression via msgpack + zlib

**Snapshot Features**
- Contradiction edges: `(belief_a, belief_b, score)`
- Support edges: high similarity without negation
- Lineage edges: parent → child from mutations
- `Snapshot.diff()` for comparing snapshots

**REST API**
- FastAPI application with belief CRUD endpoints
- Snapshot retrieval and comparison endpoints
- Agent status and control endpoints

**Benchmark System**
- Three scenario types: Contradiction, Decay, Scale
- Baseline memory systems: FIFO, LRU, VectorStore

**Metrics**
- `EcologyMetrics`, `AgentMetrics`, `IterationMetrics`
- Export to JSON, CSV, Prometheus formats

**Configuration**
- `ABESSettings` via pydantic-settings
- All spec formulas configurable (decay rate, thresholds, weights)

---

## Format

Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
