# ABES Architecture

## Overview

ABES implements a **Belief Ecology** where beliefs are living entities that decay, conflict, mutate, and cluster. The system consists of multiple layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Dashboard │ │   Chat   │ │ Explorer │ │  Docs    │  ...  │
│  │   Hub    │ │Interface │ │ (soon)   │ │ (soon)   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │ REST + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ Chat API │ │Belief API│ │ Stats API│                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Layer (Ollama)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Chat completion with belief context injection        │  │
│  │ Transforms beliefs to user-perspective for clarity   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agent Scheduler                         │
│  (15 phases: Perception → Creation → ... → Consolidation)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    16 Specialized Agents                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Perception│ │ Creator  │ │ Auditor  │ │ Mutation │  ...  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌──────────────────┐    ┌───────────────────┐              │
│  │  Belief Store    │    │  Snapshot Store   │              │
│  │  (in-memory)     │    │  (compressed)     │              │
│  └──────────────────┘    └───────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      RL Layer                                │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐             │
│  │Environment│ ←→ │  Policy  │ ←→ │ Trainer  │             │
│  │(15d/7d)   │    │(NumPy MLP│    │(ES)      │             │
│  └───────────┘    └──────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## Chat Service Flow

The chat service orchestrates the belief pipeline for conversational AI:

```
User Message
     │
     ▼
┌─────────────┐     ┌──────────────┐
│ Perception  │ ──▶ │ Extract facts│
│   Agent     │     │ from message │
└─────────────┘     └──────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────┐
│   Belief    │ ──▶ │ Create new   │
│  Creator    │     │ beliefs      │
└─────────────┘     └──────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────┐
│Reinforcement│ ──▶ │ Boost similar│
│   Agent     │     │ beliefs      │
└─────────────┘     └──────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────┐
│ Relevance   │ ──▶ │ Rank beliefs │
│  Curator    │     │ for context  │
└─────────────┘     └──────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────┐
│    LLM      │ ──▶ │ Generate response with       │
│  Provider   │     │ user's belief context        │
└─────────────┘     └──────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────┐
│  WebSocket  │ ──▶ │ Stream events│
│  Broadcast  │     │ to frontend  │
└─────────────┘     └──────────────┘
```

## Data Flow

### Per-Iteration Flow

1. **Input arrives** → PerceptionAgent extracts candidate claims
2. **BeliefCreatorAgent** deduplicates via embedding similarity, creates Belief objects
3. **ReinforcementAgent** boosts existing beliefs matching new input
4. **DecayControllerAgent** applies time-based confidence decay
5. **ContradictionAuditorAgent** computes pairwise tensions
6. **MutationEngineerAgent** proposes hedged variants for high-tension beliefs
7. **ResolutionStrategistAgent** resolves high-confidence conflicts
8. **RelevanceCuratorAgent** ranks beliefs against current context
9. **RLPolicyAgent** outputs control parameters
10. **ConsistencyCheckerAgent** probes for drift
11. **SafetySanityAgent** enforces limits
12. **BaselineMemoryBridgeAgent** interfaces with RAG/chat
13. **NarrativeExplainerAgent** generates explanations
14. **ExperimentOrchestratorAgent** logs results
15. **ConsolidationAgent** merges near-duplicates, compresses lineage

### Snapshot Capture

After each iteration, the system captures:
- All belief states (frozen `BeliefSnapshot` objects)
- Contradiction edges: `[(uuid, uuid, score)]`
- Support edges: high similarity, no negation
- Lineage edges: parent → child mutations
- Agent actions taken
- RL state/action if training

## Core Models

### Belief

```python
class Belief:
    id: UUID
    content: str                    # natural language
    confidence: float               # 0.0 to 1.0
    salience: float                 # 0.0 to 1.0 (attentional energy)
    half_life_days: float           # salience half-life (default 7.0)
    origin: OriginMetadata          # source, timestamp, last_reinforced
    tags: list[str]
    tension: float                  # contradiction pressure
    cluster_id: Optional[UUID]
    status: BeliefStatus            # active, decaying, dormant, mutated, deprecated
    parent_id: Optional[UUID]       # lineage tracking
    use_count: int
    evidence_for: list[EvidenceRef] # supporting evidence ledger
    evidence_against: list[EvidenceRef]  # attacking evidence ledger
    links: list[BeliefLink]         # graph edges (reinforces/contradicts)
```

### Snapshot

```python
class Snapshot:
    id: UUID
    metadata: SnapshotMetadata      # iteration, timestamp
    beliefs: list[BeliefSnapshot]
    global_tension: float
    contradiction_edges: list[tuple[UUID, UUID, float]]
    support_edges: list[tuple[UUID, UUID, float]]
    lineage_edges: list[tuple[UUID, UUID]]
    agent_actions: list[dict]
    rl_state_action: Optional[dict]
```

## Formulas

All computations follow spec sections 3.4.1–3.4.10:

| Formula | Equation |
|---------|----------|
| Confidence Decay | `confidence *= decay_rate ^ hours_elapsed` |
| Salience Decay | `salience *= 0.5 ^ (hours_elapsed / (half_life_days * 24))` |
| Contradiction | `semantic_similarity × negation_signal` |
| Tension | `similarity × avg_confidence × (0.5 + 0.5 × opposition)` |
| Ranking | `0.30×confidence + 0.30×relevance + 0.20×salience + 0.10×recency - 0.10×tension` |
| Stack Selection | `0.35×salience + 0.30×relevance + 0.20×recency + 0.15×graph_spread` |
| Evidence Update | `conf = 0.7 × (support / total_evidence) + 0.3 × prior_conf` |

Thresholds:
- `active → decaying`: confidence < 0.3
- `decaying → deprecated`: confidence < 0.1
- `active → dormant`: salience < 0.05
- Mutation trigger: tension ≥ 0.6 AND confidence < 0.5
- Resolution trigger: contradiction ≥ 0.7 AND both confidences ≥ 0.6
- Consolidation merge: cosine similarity > 0.92 within same cluster

## RL Environment

**State (15 dimensions):**
- Confidence: mean, std, min, max
- Tension: mean, max, high_tension_count
- Counts: total_beliefs, active, decaying (normalized)
- Cluster count
- Recent: mutations, deprecations, reinforcements
- Episode progress

**Action (7 dimensions):**
- Global decay rate adjustment
- Mutation threshold adjustment
- Resolution threshold adjustment
- Deprecation threshold adjustment
- Ranking weight: relevance
- Ranking weight: confidence
- Beliefs to surface count

**Reward shaping:**
- +task_success, +consistency, +efficiency, +stability
- −contradiction_errors, −forgetting_penalty

## Storage

Abstract interfaces in `storage/base.py`:
- `BeliefStoreABC`: CRUD + embedding search + bulk update
- `SnapshotStoreABC`: save/get/list/compare snapshots

Current implementation: in-memory dicts. Snapshots are compressed via msgpack + zlib.

## Module Map

```
backend/
├── agents/                 # 16 agents + scheduler
│   ├── perception.py       # Extract facts from chat/logs
│   ├── belief_creator.py   # Create beliefs with deduplication
│   ├── reinforcement.py    # Boost confidence + salience + evidence + graph edges
│   ├── contradiction_auditor.py  # Detect conflicts, confidence-weighted tension
│   ├── mutation_engineer.py      # Propose hedged variants
│   ├── resolution_strategist.py  # Resolve high-confidence conflicts
│   ├── relevance_curator.py      # Rank by relevance to context
│   ├── decay_controller.py       # Confidence + salience half-life + dormancy
│   ├── consolidation.py          # Merge near-duplicates, compress lineage
│   ├── baseline_memory_bridge.py # RAG/chat interface
│   ├── rl_policy.py              # RL control parameters
│   ├── reward_shaper.py          # Reward computation
│   ├── experiment_orchestrator.py # Experiment logging
│   ├── consistency_checker.py    # Drift detection
│   ├── narrative_explainer.py    # Natural language explanations
│   ├── safety_sanity.py          # Safety limits
│   └── scheduler.py              # 15-phase orchestrator
├── api/
│   ├── app.py              # FastAPI application
│   ├── schemas.py          # Request/response models
│   └── routes/
│       ├── beliefs.py      # Belief CRUD endpoints
│       ├── bel.py          # System stats/health
│       └── chat.py         # Chat API + WebSocket
├── chat/
│   └── service.py          # Chat orchestration with agent pipeline
├── llm/
│   └── provider.py         # Ollama LLM integration
├── core/
│   ├── config.py           # ABESSettings
│   ├── deps.py             # Dependency injection
│   ├── events.py           # Event models
│   ├── models/
│   │   ├── belief.py       # Belief data model
│   │   └── snapshot.py     # Snapshot with edges
│   └── bel/
│       ├── loop.py         # BeliefEcologyLoop
│       ├── clustering.py   # Semantic clustering
│       ├── decay.py        # Decay logic
│       ├── contradiction.py # Contradiction detection
│       ├── ranking.py      # Belief ranking (with salience weight)
│       ├── stack.py        # Belief stack selection + competition
│       ├── rl_integration.py # RL-BEL bridge
│       └── timeline.py     # Snapshot timeline
├── rl/
│   ├── environment.py      # BeliefEcologyEnv (Gymnasium)
│   ├── policy.py           # MLPPolicy, EvolutionStrategy
│   └── training.py         # ESTrainer
└── storage/
    ├── base.py             # Abstract interfaces
    ├── in_memory.py        # Dict-based implementations
    └── snapshot_queries.py # Query utilities

frontend/
├── app/
│   ├── page.tsx            # Dashboard hub
│   ├── chat/page.tsx       # Chat interface
│   ├── layout.tsx          # Root layout
│   └── globals.css         # Dark theme styles
├── components/
│   ├── ChatInterface.tsx   # Main chat component
│   ├── BeliefActivityPanel.tsx  # Real-time belief events
│   ├── Sidebar.tsx         # Session management
│   └── ...
└── lib/
    ├── api.ts              # API client
    └── types.ts            # TypeScript types
```

## Embeddings

- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Library: sentence-transformers
- Usage: deduplication, contradiction detection, relevance scoring
- Cached per agent to avoid redundant encoding

## LLM Integration

ABES integrates with local LLMs via Ollama for chat completion:

### Provider Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_0` | Model to use |
| `LLM_TIMEOUT` | `120.0` | Request timeout in seconds |

### Belief Context Injection

The LLM provider transforms first-person beliefs to user-perspective for clarity:

| Original (stored) | Transformed (to LLM) |
|-------------------|----------------------|
| "My name is Brad" | "User's name is Brad" |
| "I have two dogs" | "User has two dogs" |
| "I love coffee" | "User loves coffee" |

This prevents the LLM from confusing itself with the user.

### System Prompt

The system prompt explains that the LLM has learned facts about the user from previous conversations. It instructs the LLM to:

1. Use stored facts to give personalized responses
2. Refer to user's information correctly ("You mentioned..." not "My...")
3. Acknowledge new information naturally
4. Summarize accurately when asked about user's profile

## WebSocket Events

Real-time belief events are broadcast via WebSocket at `/chat/ws`:

| Event Type | Description |
|------------|-------------|
| `created` | New belief extracted from user message |
| `reinforced` | Existing belief strengthened by similar input |
| `mutated` | Belief evolved due to tension |
| `deprecated` | Belief decayed below threshold |
| `tension_changed` | Contradiction detected |

Event payload:
```json
{
  "type": "belief_event",
  "data": {
    "event_type": "created",
    "belief_id": "uuid",
    "content": "User's name is Brad",
    "confidence": 0.8,
    "tension": 0.0,
    "details": {"source": "user_message"},
    "timestamp": "2026-01-30T12:00:00Z"
  }
}
```
