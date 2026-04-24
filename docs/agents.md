# Agent Reference

ABES uses 16 specialized agents orchestrated by a scheduler. Each agent has a single responsibility and operates on beliefs or ecology state.

---

## Execution Order

Agents run in a fixed 15-phase order per iteration:

1. Perception
2. Creation
3. Reinforcement
4. Decay
5. Contradiction
6. Mutation
7. Resolution
8. Relevance
9. RL Policy
10. Consistency
11. Safety
12. Baseline
13. Narrative
14. Experiment
15. Consolidation

Not all agents run every iteration. The scheduler supports `run_every_n` and conditional execution.

---

## Agent Details

### 1. PerceptionAgent

**Location:** `backend/agents/perception.py`

**Purpose:** Extracts candidate belief strings from raw input (chat messages, logs, tool output).

**Key method:**
```python
async def ingest(self, text: str, metadata: dict) -> list[str]
```

**Behavior:**
- Filters greetings, acknowledgments, pure commands
- Extracts factual claims via heuristics
- Strips log prefixes and timestamps
- Deduplicates via LRU cache

**Tests:** 20+ tests in `test_perception.py`

---

### 2. BeliefCreatorAgent

**Location:** `backend/agents/belief_creator.py`

**Purpose:** Creates `Belief` objects from candidate strings, deduplicating against existing beliefs.

**Key method:**
```python
async def create_beliefs(
    self, candidates: list[str], origin: OriginMetadata, store: BeliefStoreABC
) -> list[Belief]
```

**Behavior:**
- Embeds candidates via sentence-transformers
- Checks cosine similarity against store (threshold: 0.95)
- Assigns tags based on content patterns
- Saves embeddings if store supports it

**Tests:** 25+ tests in `test_belief_creator.py`

---

### 3. ReinforcementAgent

**Location:** `backend/agents/reinforcement.py`

**Purpose:** Boosts confidence and salience of existing beliefs when new input is similar. Populates evidence ledger and graph edges.

**Key method:**
```python
async def reinforce(
    self, incoming: str, beliefs: list[Belief], store: BeliefStoreABC
) -> list[Belief]
```

**Behavior:**
- Embeds incoming text
- Finds beliefs with similarity > 0.7
- Boosts confidence by 0.1 (capped at 0.95)
- Boosts salience by 0.1 (capped at 1.0)
- Adds `EvidenceRef(direction="supports")` to reinforced beliefs
- Adds `reinforces` graph edges between co-reinforced beliefs
- Updates `last_reinforced` timestamp
- Respects 60-second cooldown per belief

**Tests:** 15+ tests in `test_reinforcement.py`, 4 ecology tests in `test_reinforcement_ecology.py`

---

### 4. ContradictionAuditorAgent

**Location:** `backend/agents/contradiction_auditor.py`

**Purpose:** Detects conflicting belief pairs and computes per-belief tension. Populates graph edges.

**Key method:**
```python
async def audit(self, beliefs: list[Belief]) -> list[ContradictionDetectedEvent]
```

**Behavior:**
- Caches embeddings with LRU eviction
- Computes pairwise cosine similarity
- Applies negation heuristic (negation words, antonym pairs)
- Tension = similarity × avg_confidence × (0.5 + 0.5 × opposition)
- Adds `contradicts` graph edges on detected contradictions
- Emits events when tension crosses threshold

**Tests:** 25+ tests in `test_contradiction_auditor.py`, 4 tests in `test_tension_formula.py`

---

### 5. MutationEngineerAgent

**Location:** `backend/agents/mutation_engineer.py`

**Purpose:** Proposes mutated belief variants for high-tension, low-confidence beliefs.

**Key method:**
```python
def propose_mutation(
    self, belief: Belief, contradicting: Optional[Belief], all_beliefs: list[Belief]
) -> Optional[MutationProposal]
```

**Strategies:**
- **Hedge:** "It may be that {content}"
- **Condition:** "{content}, at least as of {date}"
- **Scope narrow:** Replace "always" → "usually", etc.
- **Source attribute:** "According to {source}, {content}"

**Behavior:**
- Triggers when tension ≥ 0.6 AND confidence < 0.5
- Respects max mutation depth (5)
- Preserves lineage via `parent_id`
- New belief starts at confidence 0.5

**Tests:** 20+ tests in `test_mutation_engineer.py`

---

### 6. ResolutionStrategistAgent

**Location:** `backend/agents/resolution_strategist.py`

**Purpose:** Resolves conflicts between high-confidence contradicting beliefs.

**Key method:**
```python
def resolve(
    self, belief_a: Belief, belief_b: Belief, contradiction_score: float
) -> Optional[ResolutionResult]
```

**Strategies:**
- **Integrate:** Merge into conditional belief (>70% token overlap)
- **Split:** Add scope tags when temporal markers present
- **Deprecate loser:** Lower-confidence belief deprecated (gap > 0.3)

**Triggers:** contradiction ≥ 0.7 AND both confidences ≥ 0.6

**Tests:** 25+ tests in `test_resolution_strategist.py`

---

### 7. RelevanceCuratorAgent

**Location:** `backend/agents/relevance_curator.py`

**Purpose:** Ranks beliefs by weighted formula against current context.

**Key method:**
```python
async def rank_beliefs(
    self, beliefs: list[Belief], context: str, ...
) -> list[RankedBelief]
```

**Formula:**
```
rank = 0.4×relevance + 0.3×confidence + 0.2×recency - 0.1×tension
```

**Behavior:**
- Filters beliefs with relevance < 0.3
- Recency: `1 - (hours_since_reinforced / 168)`
- Weights adjustable via RL

**Tests:** 15+ tests in `test_relevance_curator.py`

---

### 8. DecayControllerAgent

**Location:** `backend/agents/decay_controller.py`

**Purpose:** Applies time-based confidence and salience decay, manages status transitions including dormancy.

**Key method:**
```python
def apply_decay(self, belief: Belief) -> Optional[DecayEvent]
```

**Formulas:**
```
new_confidence = confidence × (decay_rate ^ hours_elapsed)
new_salience = salience × 0.5 ^ (hours_elapsed / (half_life_days × 24))
```

**Status transitions:**
- `active → decaying`: confidence < 0.3
- `decaying → deprecated`: confidence < 0.1
- `active/decaying → dormant`: salience < 0.05
- `any → deprecated`: use_count = 0 AND age > 30 days

**DecayEvent** now includes `old_salience` / `new_salience` fields.

Supports per-cluster and per-tag rate overrides.

**Tests:** 20+ tests in `test_decay_controller.py`, 7 tests in `test_decay_salience.py`

---

### 9. BaselineMemoryBridgeAgent

**Location:** `backend/agents/baseline_memory_bridge.py`

**Purpose:** Interfaces with non-ecological memory systems for comparison.

**Backends:**
- `RAGBackend` — vector similarity retrieval
- `ChatHistoryBackend` — recency-based retrieval
- `BeliefEcologyBackend` — adapter for belief store

**Key method:**
```python
async def compare_retrieval(self, query: str, top_k: int) -> ComparisonResult
```

**Tests:** 15+ tests in `test_baseline_memory_bridge.py`

---

### 10. RLPolicyAgent

**Location:** `backend/agents/rl_policy.py`

**Purpose:** Outputs control parameters from current ecology state.

**Key method:**
```python
def get_action(self, state: EcologyState) -> PolicyAction
```

**Output (`PolicyAction`):**
- `global_decay_rate`
- `mutation_threshold`, `resolution_threshold`, `deprecation_threshold`
- `ranking_weights`
- `beliefs_to_surface`

**Current implementation:** Heuristic fallback until trained policy loaded.

**Tests:** 10+ tests in `test_rl_policy.py`

---

### 11. RewardShaperAgent

**Location:** `backend/agents/reward_shaper.py`

**Purpose:** Computes shaped reward signals for RL training.

**Key method:**
```python
def compute_reward(
    self, task_success: float, beliefs: list[Belief], ...
) -> RewardSignal
```

**Components:**
- +task_success, +consistency, +efficiency, +stability
- −contradiction_penalty, −forgetting_penalty

**Tests:** 10+ tests in `test_reward_shaper.py`

---

### 12. ExperimentOrchestratorAgent

**Location:** `backend/agents/experiment_orchestrator.py`

**Purpose:** Runs scripted scenarios and logs experimental results.

**Key methods:**
```python
def register_scenario(self, scenario: Scenario) -> UUID
async def run_experiment(self, scenario_id: UUID, config: SystemConfig) -> ExperimentRun
```

**Supports:** step handlers, expected outcomes, metrics computation

**Tests:** 15+ tests in `test_experiment_orchestrator.py`

---

### 13. ConsistencyCheckerAgent

**Location:** `backend/agents/consistency_checker.py`

**Purpose:** Probes system with previous queries to detect answer drift.

**Key methods:**
```python
def record_probe(self, query: str, response: str, belief_ids: list[UUID]) -> ConsistencyProbe
def check_consistency(self, probe: ConsistencyProbe, current_response: str, ...) -> ConsistencyResult
```

**Metrics:**
- Text similarity (Jaccard)
- Belief overlap (Jaccard of IDs)
- Consistency rate across probes

**Tests:** 20+ tests in `test_consistency_checker.py`

---

### 14. NarrativeExplainerAgent

**Location:** `backend/agents/narrative_explainer.py`

**Purpose:** Generates human-readable explanations of ecology behavior.

**Key methods:**
```python
def explain_selection(self, context: ExplanationContext) -> Explanation
def explain_mutation(self, original: Belief, mutated: Belief) -> Explanation
def explain_resolution(self, belief_a: Belief, belief_b: Belief, strategy: str) -> Explanation
```

**Output:** Summary text + detail bullets + belief references

**Tests:** 20+ tests in `test_narrative_explainer.py`

---

### 15. SafetySanityAgent

**Location:** `backend/agents/safety_sanity.py`

**Purpose:** Enforces guardrails and vetoes dangerous actions.

**Checks:**
- Low-confidence belief usage
- Runaway mutation (depth > 5)
- Belief proliferation (> 10,000 active)
- Core belief forgetting
- Cluster overflow (> 500 per cluster)
- Content length (> 2000 chars)
- Deprecation spikes (> 30% in one pass)

**Actions:** Warn, Block, Override

**Tests:** 25+ tests in `test_safety_sanity.py`

---

### 16. ConsolidationAgent

**Location:** `backend/agents/consolidation.py`

**Purpose:** Merges near-duplicate beliefs within clusters and compresses deep lineage chains.

**Key method:**
```python
async def consolidate(
    self, beliefs: list[Belief]
) -> tuple[list[ConsolidationEvent], list[Belief], list[Belief]]
```

**Merge behavior:**
- Groups beliefs by `cluster_id`
- Computes pairwise cosine similarity within each cluster
- Merges pairs with similarity > 0.92 (configurable)
- Winner = higher `confidence × salience`
- Winner absorbs loser's evidence ledger and graph edges
- Loser gets deprecated

**Lineage compression:**
- Finds chains deeper than `max_depth` (default 5)
- Tip belief absorbs all ancestor evidence
- Ancestors get deprecated, tip's `parent_id` reset to None

**Tests:** 8 tests in `test_consolidation.py`

---

## AgentScheduler

**Location:** `backend/agents/scheduler.py`

**Purpose:** Orchestrates agent execution in defined order.

**Key methods:**
```python
def register(self, phase: AgentPhase, agent: Any, enabled: bool, run_every_n: int, condition: Callable)
async def run_iteration(self, context: SchedulerContext) -> list[AgentResult]
```

**Features:**
- 15 phases via `AgentPhase` enum
- Conditional execution
- Frequency control (`run_every_n`)
- Enable/disable per agent
- Result collection

**Tests:** 20 tests in `test_scheduler.py`, 5 tests in `test_scheduler_consolidation.py`
