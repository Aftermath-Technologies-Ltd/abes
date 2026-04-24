# Author: Bradley R. Kinnard
"""
Drift and contradiction benchmark.

Runs a fixed conversation script against three systems:
1. Plain LLM (no memory)
2. Append-only memory (naive storage)
3. Belief ecology (full system)

Captures belief state after every turn and writes results to
results/drift_comparison.json.

The conversation script includes:
- Reinforcement (same fact repeated)
- Direct contradictions
- Repeated prompts

All operations are deterministic and seed-controlled.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.plain_llm_runner import PlainLLMRunner
from baselines.append_only_memory import AppendOnlyMemory
from metrics.drift_metrics import compute_belief_entropy, count_potential_contradictions


# Fixed conversation script - versioned and deterministic
CONVERSATION_SCRIPT = [
    # Initial facts
    "My name is Alex.",
    "I have a dog named Max.",
    "I work as a software engineer.",
    "I like coffee.",

    # Reinforcement - same facts repeated
    "My name is Alex.",
    "I really love coffee.",
    "I'm a software engineer.",

    # New facts
    "I live in Seattle.",
    "Max is a golden retriever.",
    "I have been coding for 10 years.",

    # Direct contradictions
    "Actually, my name is Sam.",  # Contradicts "My name is Alex"
    "I don't like coffee anymore.",  # Contradicts "I like coffee"
    "I work as a data scientist now.",  # Contradicts "software engineer"

    # More reinforcement
    "I live in Seattle.",
    "Max is my best friend.",

    # More contradictions
    "I've only been coding for 5 years.",  # Contradicts "10 years"
    "I moved to Portland.",  # Contradicts "Seattle"

    # Complex statements
    "I have a cat named Luna.",
    "Luna and Max get along well.",

    # Repeated prompts (exact duplicates)
    "My name is Alex.",
    "I have a dog named Max.",

    # Final contradictions
    "I actually hate dogs.",  # Strong contradiction
    "I never learned to code.",  # Strong contradiction
]

SCRIPT_VERSION = "1.0.0"
SEED = 12345


class BeliefEcologyAdapter:
    """
    Adapter to run belief ecology system in deterministic mode.

    Uses the actual ABES components but with:
    - No LLM calls (perception only)
    - Deterministic processing
    - State extraction for metrics
    """

    def __init__(self, seed: int = 12345):
        self._seed = seed
        self._turn_count = 0
        self._beliefs: list[dict] = []

        # Import actual components
        try:
            from backend.storage.in_memory import InMemoryBeliefStore
            from backend.agents.perception import PerceptionAgent
            from backend.agents.reinforcement import ReinforcementAgent
            from backend.agents.contradiction_auditor import ContradictionAuditorAgent
            from backend.agents.mutation_engineer import MutationEngineerAgent
            from backend.agents.decay_controller import DecayControllerAgent
            from backend.core.models.belief import Belief, OriginMetadata, BeliefStatus

            self._store = InMemoryBeliefStore()
            self._perception = PerceptionAgent()
            self._reinforcement = ReinforcementAgent()
            self._contradiction = ContradictionAuditorAgent()
            self._mutation = MutationEngineerAgent()
            self._decay = DecayControllerAgent()
            self._Belief = Belief
            self._OriginMetadata = OriginMetadata
            self._BeliefStatus = BeliefStatus
            self._available = True
        except ImportError as e:
            print(f"Warning: Could not import ABES components: {e}")
            self._available = False

    def reset(self) -> None:
        """Reset to initial state."""
        self._turn_count = 0
        self._beliefs = []
        if self._available:
            self._store = type(self._store)()

    async def process_turn(self, user_message: str) -> dict:
        """Process a single turn through the belief ecology."""
        self._turn_count += 1

        if not self._available:
            return {
                "turn": self._turn_count,
                "beliefs_active": 0,
                "contradictions_detected": 0,
                "belief_entropy": 0.0,
            }

        # Extract beliefs from message
        candidates = await self._perception.ingest(
            user_message,
            {"source_type": "benchmark"}
        )

        # Create beliefs
        for content in candidates:
            belief = self._Belief(
                content=content,
                confidence=0.8,
                origin=self._OriginMetadata(source="benchmark"),
            )
            await self._store.create(belief)

        # Run reinforcement
        all_beliefs = await self._store.list(status=self._BeliefStatus.Active, limit=1000)
        await self._reinforcement.reinforce(
            incoming=user_message,
            beliefs=all_beliefs,
            store=self._store,
        )

        # Run contradiction detection
        all_beliefs = await self._store.list(status=self._BeliefStatus.Active, limit=1000)
        events = await self._contradiction.audit(all_beliefs, store=self._store)

        # Get final state
        all_beliefs = await self._store.list(status=self._BeliefStatus.Active, limit=1000)
        belief_dicts = [{"content": b.content, "confidence": b.confidence} for b in all_beliefs]

        return {
            "turn": self._turn_count,
            "beliefs_active": len(all_beliefs),
            "contradictions_detected": len(events),
            "belief_entropy": compute_belief_entropy(belief_dicts),
        }

    def get_belief_count(self) -> int:
        return len(self._beliefs)

    async def get_state(self) -> dict:
        if not self._available:
            return {"turn_count": self._turn_count, "beliefs": []}

        all_beliefs = await self._store.list(status=self._BeliefStatus.Active, limit=1000)
        return {
            "turn_count": self._turn_count,
            "beliefs": [
                {"content": b.content, "confidence": b.confidence, "tension": b.tension}
                for b in all_beliefs
            ],
        }


async def run_benchmark(seed: int = SEED) -> dict:
    """
    Run the full drift benchmark against all three systems.

    Returns results dict matching required schema.
    """
    print(f"Running drift benchmark with seed={seed}, {len(CONVERSATION_SCRIPT)} turns")

    # Initialize systems
    plain_llm = PlainLLMRunner(seed=seed)
    append_only = AppendOnlyMemory(seed=seed)
    belief_ecology = BeliefEcologyAdapter(seed=seed)

    # Results storage
    results = {
        "run_metadata": {
            "seed": seed,
            "num_turns": len(CONVERSATION_SCRIPT),
            "script_version": SCRIPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "systems": {
            "plain_llm": {
                "belief_count_over_time": [],
                "contradiction_count_over_time": [],
                "belief_entropy_over_time": [],
            },
            "append_only": {
                "belief_count_over_time": [],
                "contradiction_count_over_time": [],
                "belief_entropy_over_time": [],
            },
            "belief_ecology": {
                "belief_count_over_time": [],
                "contradiction_count_over_time": [],
                "belief_entropy_over_time": [],
            },
        },
    }

    # Process each turn
    for i, message in enumerate(CONVERSATION_SCRIPT):
        print(f"  Turn {i+1}/{len(CONVERSATION_SCRIPT)}: {message[:40]}...")

        # Plain LLM
        plain_metrics = plain_llm.process_turn(message)
        results["systems"]["plain_llm"]["belief_count_over_time"].append(
            plain_llm.get_belief_count()
        )
        results["systems"]["plain_llm"]["contradiction_count_over_time"].append(
            plain_llm.get_contradiction_count()
        )
        results["systems"]["plain_llm"]["belief_entropy_over_time"].append(
            plain_llm.compute_entropy()
        )

        # Append-only
        append_metrics = append_only.process_turn(message)
        state = append_only.get_state()
        belief_dicts = [{"content": b.content} for b in state.beliefs]
        results["systems"]["append_only"]["belief_count_over_time"].append(
            append_only.get_belief_count()
        )
        results["systems"]["append_only"]["contradiction_count_over_time"].append(
            count_potential_contradictions(belief_dicts)
        )
        results["systems"]["append_only"]["belief_entropy_over_time"].append(
            append_only.compute_entropy()
        )

        # Belief ecology
        ecology_metrics = await belief_ecology.process_turn(message)
        results["systems"]["belief_ecology"]["belief_count_over_time"].append(
            ecology_metrics["beliefs_active"]
        )
        results["systems"]["belief_ecology"]["contradiction_count_over_time"].append(
            ecology_metrics["contradictions_detected"]
        )
        results["systems"]["belief_ecology"]["belief_entropy_over_time"].append(
            ecology_metrics["belief_entropy"]
        )

    return results


def main():
    """Run benchmark and write results."""
    results = asyncio.run(run_benchmark())

    # Write results
    output_path = PROJECT_ROOT / "results" / "drift_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for system_name, data in results["systems"].items():
        final_beliefs = data["belief_count_over_time"][-1] if data["belief_count_over_time"] else 0
        final_contradictions = data["contradiction_count_over_time"][-1] if data["contradiction_count_over_time"] else 0
        print(f"{system_name}: {final_beliefs} beliefs, {final_contradictions} contradictions")

    return 0


if __name__ == "__main__":
    sys.exit(main())
