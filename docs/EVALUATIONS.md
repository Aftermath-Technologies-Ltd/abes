# ABES Evaluation Results

Detailed breakdowns of all cognitive evaluation suites. For a high-level summary, see the [README](../README.md#testing-and-verification).

---

## Cognitive Stress Test (200-Prompt Suite)

End-to-end test sending 200 graded prompts through the full ABES pipeline. Each prompt is evaluated against deterministic pass/fail criteria.

```bash
PYTHONPATH=$PWD python tests/cognitive/test_200_stress.py
```

**Result: 200/200 (100.0%) across 3 consecutive runs on Llama 3.1 8B**

| Category | Prompts | Score | What it tests |
|----------|---------|-------|---------------|
| Identity | 13 | 100% | Name recall, self-identification, persona consistency |
| Belief Creation (Personal) | 14 | 100% | Facts about the ingesting agent (job, location, family, pets) |
| Belief Creation (Preferences) | 12 | 100% | Likes, dislikes, favorites |
| Reinforcement | 14 | 100% | Repeated facts boost confidence, 3-step sequences |
| Deduplication | 12 | 100% | Near-identical messages don't create duplicate beliefs |
| Noise Rejection | 13 | 100% | Filler phrases, gibberish, injection attempts, code payloads |
| Memory Recall | 14 | 100% | Retrieve stored facts across topics |
| Context Relevance | 11 | 100% | Factual queries answered without injecting stored beliefs |
| Contradiction Handling | 15 | 100% | Contradicted facts update, tension mechanics |
| Safety | 14 | 100% | Prompt injection, system prompt extraction, jailbreak attempts |
| Multi-Fact Extraction | 11 | 100% | Multiple beliefs from compound sentences |
| General Knowledge | 12 | 100% | Non-personal questions answered accurately |
| Session Isolation | 10 | 100% | Beliefs don't leak between session scopes |
| Conversational Coherence | 11 | 100% | Natural dialogue flow, follow-up questions |
| Edge Cases | 13 | 100% | Empty messages, single words, unicode, URLs |
| Belief Lifecycle | 11 | 100% | Create, recall, reinforce, update, recall chain |

Source: [tests/cognitive/test_200_stress.py](../tests/cognitive/test_200_stress.py)
Artifact: [results/stress_test_200_results.json](../results/stress_test_200_results.json)

---

## Cognitive AI Battery (50-Prompt Suite)

Research-grade evaluation of human-like cognitive capabilities grounded in peer-reviewed literature. All 50 prompts target specific constructs. All assertions are deterministic boolean predicates.

```bash
PYTHONPATH=$PWD python tests/cognitive/test_cognitive_battery.py
```

**Result: 50/50 (100.0%) across 3 consecutive runs on Llama 3.1 8B**

| Domain | Prompts | Score | Framework |
|--------|---------|-------|-----------|
| Episodic Memory | 7 | 100% | Tulving 1972 |
| Semantic Memory | 6 | 100% | Collins & Quillian 1969 |
| Working Memory | 6 | 100% | Baddeley & Hitch 1974 |
| Selective Attention | 5 | 100% | Broadbent 1958, Posner 1980 |
| Language Comprehension | 7 | 100% | Grice 1975, Searle 1975 |
| Deductive & Inductive Reasoning | 7 | 100% | Wason 1966, Gentner 1983 |
| Social Cognition | 6 | 100% | Premack & Woodruff 1978, Kohlberg 1958 |
| Self-Correction | 6 | 100% | AGM 1985 |

Source: [tests/cognitive/test_cognitive_battery.py](../tests/cognitive/test_cognitive_battery.py)
Artifact: [results/cognitive_battery_50_results.json](../results/cognitive_battery_50_results.json)

---

## 1000-Prompt Cognitive Evaluation Suite

A research-grade evaluation with 1000 prompts across 8 cognitive domains, 40 constructs, and 35 long-horizon decay scenarios. Uses semantic cosine similarity scoring (all-MiniLM-L6-v2) instead of keyword matching, plus ecology invariant auditing to verify internal belief mechanics.

```bash
# Validate prompt bank structure (no API needed)
python -m tests.cognitive.eval.run --dry-run

# Full 1000-prompt run against live backend
python -m tests.cognitive.eval.run

# Quick smoke test (stratified across all domains)
python -m tests.cognitive.eval.run --max 24

# Target specific domains
python -m tests.cognitive.eval.run --domains reasoning social_cognition

# Verbose logging
python -m tests.cognitive.eval.run --max 50 -v
```

**Baseline Result: 825/1000 (82.5%), 95% CI [0.800, 0.848]**

Mean cosine similarity: 0.875. Zero ecology violations. | Mean Cosine | Constructs |
|--------|--------|------|-------------|------------|
| Episodic Memory | 121/125 | 96.8% | 0.93 | encoding, temporal/spatial retrieval, temporal ordering, source monitoring |
| Working Memory | 118/125 | 94.4% | 0.92 | multi-item encoding, item retrieval, interference resistance, feature binding, updating |
| Semantic Memory | 116/125 | 92.8% | 0.93 | fact encoding, categorical inference, property retrieval, source discrimination, knowledge update |
| Selective Attention | 107/125 | 85.6% | 0.89 | target encoding, distractor filtering, inhibition, relevance gating, focused retrieval |
| Self-Correction | 103/125 | 82.4% | 0.91 | belief revision, iterative revision, long-horizon decay, minimal change, revision verification |
| Reasoning | 99/125 | 79.2% | 0.85 | modus ponens, modus tollens, analogical, transitive inference, causal reasoning |
| Language Comprehension | 88/125 | 70.4% | 0.83 | disambiguation, figurative language, indirect speech, presupposition, scalar implicature |
| Social Cognition | 73/125 | 58.4% | 0.75 | false belief, perspective taking, emotional inference, intention attribution, moral reasoning |

The social_cognition shortfall is primarily from moral_reasoning (20 of 52 failures): the LLM deflects ethical dilemmas with refusal responses instead of reasoning through them. This is an LLM-level behavior, not a belief ecology defect.

**Top failure constructs**: moral_reasoning (20), scalar_implicature (12), disambiguation (11), focused_retrieval (9), false_belief (9), analogical_reasoning (8).

### Eval Architecture

The 1000-prompt suite lives in [tests/cognitive/eval/](../tests/cognitive/eval/) and consists of 7 modules:

| Module | Purpose |
|--------|---------|
| [prompt_bank.py](../tests/cognitive/eval/prompt_bank.py) | 1000 prompts: 8 domains, 40 constructs, gold answers, forbidden semantics, ecology checks |
| [scorer.py](../tests/cognitive/eval/scorer.py) | Semantic cosine similarity scoring via all-MiniLM-L6-v2 embeddings |
| [ecology_auditor.py](../tests/cognitive/eval/ecology_auditor.py) | 7 belief-state invariant checks: creation, reinforcement, tension, mutation, salience decay, dormancy, orphan links |
| [stats.py](../tests/cognitive/eval/stats.py) | Clopper-Pearson exact binomial CI (pure Python, no scipy), per-domain distributions, test-retest correlation |
| [reporter.py](../tests/cognitive/eval/reporter.py) | Terminal summary, Markdown report, JSON artifact generation |
| [harness.py](../tests/cognitive/eval/harness.py) | Async runner with session grouping, stratified sampling, long-horizon decay cycling |
| [run.py](../tests/cognitive/eval/run.py) | CLI entry point: `--dry-run`, `--max`, `--domains`, `--constructs`, `--concurrency`, `--threshold` |

Each prompt carries: `domain`, `construct`, `gold_answer`, `forbidden_semantics`, `ecology_checks`, `session_group`, `is_setup`, `horizon`, and an academic `reference`. Setup prompts establish context (470 total); probe prompts test recall and reasoning (530 total). Long-horizon prompts (35 total) inject decay cycles between setup and probe to test salience persistence.

Scoring uses cosine similarity between the LLM response and the gold answer embedding. Pass threshold: 0.70. Forbidden answers are checked with a 0.60 ceiling. The ecology auditor runs invariant checks against pre/post belief snapshots fetched via `GET /beliefs/{id}/ecology`.

Artifact: [results/cognitive_eval/baseline_1000.json](../results/cognitive_eval/baseline_1000.json)

---

## Limitations

- **Self-reported evals:** All results were produced by the authors using their own evaluation harness. No independent replication has been performed.
- **Single model tested:** All evaluations used Llama 3.1 8B via Ollama. Results with other models or sizes may differ significantly.
- **No external baselines:** Comparisons against Mem0, LangGraph MemoryStore, vector DB + reranker, or raw chat history have not been run. The side-by-side eval compares ABES against a stateless LLM baseline only.
- **Weak domains:** Social cognition (58.4%) and moral reasoning specifically are limited by LLM refusal behavior, not belief-layer mechanics. The belief revision system does not control how the LLM handles ethical questions.
- **Language comprehension (70.4%):** Scalar implicature and disambiguation failures are partially attributable to cosine similarity scoring thresholds rather than system correctness.

---

## Planned Baselines

The following external comparisons are planned but not yet completed:

- **Mem0** — commercial memory layer for LLM agents
- **Vector DB + reranker** — embedding search with cross-encoder reranking (no structured belief model)
- **LangGraph MemoryStore** — graph-based memory in the LangChain ecosystem
- **Raw chat history** — full conversation history passed as context, no compression or belief extraction

These baselines will test whether structured belief revision outperforms simpler approaches on the same evaluation suite.
