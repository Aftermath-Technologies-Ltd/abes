# Author: Bradley R. Kinnard
"""
Decay sensitivity sweep experiment.

Tests the belief decay system with multiple decay factors to measure:
- Belief stability vs churn
- Memory retention characteristics
- Impact on contradiction accumulation

Runs identical belief streams with decay values:
[0.999, 0.995, 0.99, 0.97, 0.95]

Writes outputs to results/decay_sweep/decay_<value>.json
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.decay_metrics import summarize_decay_run

# Decay factors to test
DECAY_FACTORS = [0.999, 0.995, 0.99, 0.97, 0.95]

# Fixed seed for reproducibility
SEED = 12345

# Simulated duration (accelerated)
SIMULATED_HOURS = 48

# Fixed belief stream for testing
BELIEF_STREAM = [
    # Core identity facts - should persist
    {"content": "My name is Alex", "reinforcement_count": 5},
    {"content": "I work as an engineer", "reinforcement_count": 3},

    # Preferences - moderate persistence
    {"content": "I like coffee", "reinforcement_count": 2},
    {"content": "I prefer mornings", "reinforcement_count": 1},

    # Transient facts - should decay
    {"content": "It is sunny today", "reinforcement_count": 0},
    {"content": "I had pasta for lunch", "reinforcement_count": 0},
    {"content": "The meeting was at 3pm", "reinforcement_count": 0},

    # Contradicting facts - tension test
    {"content": "I live in Seattle", "reinforcement_count": 2},
    {"content": "I moved to Portland", "reinforcement_count": 1},  # Contradiction

    # More transient
    {"content": "The traffic was bad", "reinforcement_count": 0},
    {"content": "I need to buy milk", "reinforcement_count": 0},

    # Long-term preferences
    {"content": "I enjoy hiking", "reinforcement_count": 4},
    {"content": "Reading is my hobby", "reinforcement_count": 3},
]


class DecaySimulator:
    """
    Simulates belief decay over time without requiring full ABES stack.

    Models the core decay behavior:
    - Each turn, confidence *= decay_factor
    - Beliefs drop below threshold -> deprecated
    - Reinforcement resets confidence
    """

    DEPRECATION_THRESHOLD = 0.3

    def __init__(self, decay_factor: float, seed: int = SEED):
        self.decay_factor = decay_factor
        self.seed = seed
        self.beliefs: list[dict] = []
        self.belief_history: list[dict] = []
        self.turn = 0
        self.beliefs_created = 0
        self.beliefs_dropped = 0
        self.contradictions_detected = 0

    def reset(self) -> None:
        self.beliefs = []
        self.belief_history = []
        self.turn = 0
        self.beliefs_created = 0
        self.beliefs_dropped = 0
        self.contradictions_detected = 0

    def add_belief(self, content: str, confidence: float = 0.8) -> None:
        """Add a new belief."""
        self.beliefs.append({
            "content": content,
            "confidence": confidence,
            "created_turn": self.turn,
            "deprecated_turn": None,
            "reinforcement_count": 0,
        })
        self.beliefs_created += 1

    def reinforce(self, content: str, boost: float = 0.05) -> bool:
        """Reinforce a belief by content match."""
        for b in self.beliefs:
            if b["content"].lower() == content.lower() and b["deprecated_turn"] is None:
                b["confidence"] = min(0.95, b["confidence"] + boost)
                b["reinforcement_count"] += 1
                return True
        return False

    def apply_decay(self) -> int:
        """Apply decay to all beliefs. Returns count of dropped beliefs."""
        dropped = 0
        for b in self.beliefs:
            if b["deprecated_turn"] is not None:
                continue

            b["confidence"] *= self.decay_factor

            if b["confidence"] < self.DEPRECATION_THRESHOLD:
                b["deprecated_turn"] = self.turn
                self.belief_history.append(b.copy())
                dropped += 1

        self.beliefs_dropped += dropped
        return dropped

    def check_contradictions(self) -> int:
        """Simple contradiction check using antonym pairs."""
        antonyms = [
            ("seattle", "portland"),
            ("morning", "evening"),
            ("like", "hate"),
        ]

        active = [b for b in self.beliefs if b["deprecated_turn"] is None]
        count = 0

        for i, b1 in enumerate(active):
            for b2 in active[i+1:]:
                c1 = b1["content"].lower()
                c2 = b2["content"].lower()

                for w1, w2 in antonyms:
                    if (w1 in c1 and w2 in c2) or (w2 in c1 and w1 in c2):
                        count += 1
                        break

        return count

    def advance_turn(self) -> None:
        """Advance simulation by one turn."""
        self.turn += 1
        self.apply_decay()
        self.contradictions_detected = self.check_contradictions()

    def get_active_count(self) -> int:
        return sum(1 for b in self.beliefs if b["deprecated_turn"] is None)

    def get_retained_count(self) -> int:
        return self.get_active_count()


def simulate_decay(decay_factor: float, turns: int = 1000) -> dict:
    """
    Run a full decay simulation.

    Args:
        decay_factor: Decay multiplier per turn
        turns: Number of turns to simulate

    Returns:
        Results dict matching required schema
    """
    sim = DecaySimulator(decay_factor=decay_factor, seed=SEED)

    # Add initial beliefs
    for item in BELIEF_STREAM:
        sim.add_belief(item["content"])

    # Simulate turns with periodic reinforcement
    reinforcement_schedule = {
        0: ["My name is Alex", "I work as an engineer"],
        50: ["My name is Alex", "I like coffee"],
        100: ["My name is Alex", "I enjoy hiking"],
        200: ["My name is Alex"],
        300: ["My name is Alex"],
        400: ["My name is Alex"],
        500: ["I work as an engineer"],
    }

    for turn in range(turns):
        sim.advance_turn()

        # Apply reinforcement at scheduled times
        if turn in reinforcement_schedule:
            for content in reinforcement_schedule[turn]:
                sim.reinforce(content)

    # Compute duration in hours (simulated)
    duration_hours = (turns / 1000) * SIMULATED_HOURS

    return summarize_decay_run(
        decay_factor=decay_factor,
        seed=SEED,
        duration_hours=duration_hours,
        beliefs_created=sim.beliefs_created,
        beliefs_dropped=sim.beliefs_dropped,
        beliefs_retained=sim.get_retained_count(),
        contradictions_detected=sim.contradictions_detected,
        belief_history=sim.belief_history,
    )


def main() -> int:
    """Run decay sweep for all factors."""
    print(f"Running decay sweep experiment with {len(DECAY_FACTORS)} factors")
    print(f"Seed: {SEED}")

    output_dir = PROJECT_ROOT / "results" / "decay_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for factor in DECAY_FACTORS:
        print(f"\n  Testing decay_factor={factor}...")

        results = simulate_decay(decay_factor=factor, turns=1000)

        # Write individual result file
        output_path = output_dir / f"decay_{factor}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"    Retained: {results['metrics']['beliefs_retained']}")
        print(f"    Dropped: {results['metrics']['beliefs_dropped']}")
        print(f"    Churn rate: {results['metrics']['belief_churn_rate']:.4f}")

        all_results.append(results)

    print(f"\nResults written to: {output_dir}/")

    # Print comparison table
    print("\n=== Decay Factor Comparison ===")
    print(f"{'Factor':<10} {'Retained':<10} {'Dropped':<10} {'Churn':<10}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['decay_factor']:<10} "
              f"{r['metrics']['beliefs_retained']:<10} "
              f"{r['metrics']['beliefs_dropped']:<10} "
              f"{r['metrics']['belief_churn_rate']:<10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
