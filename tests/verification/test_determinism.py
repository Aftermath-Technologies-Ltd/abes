# Author: Bradley R. Kinnard
"""
Determinism verification test.

Verifies that identical inputs produce byte-for-byte identical outputs.
This is critical for reproducible research and debugging.

Writes results to results/determinism_check.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from interfaces.belief_ingest import DeterministicBeliefIngest


# Fixed seed for all tests
SEED = 12345

# Fixed input sequence
INPUT_SEQUENCE = [
    "My name is Alex.",
    "I work as a software engineer.",
    "I have a dog named Max.",
    "I like coffee and hiking.",
    "My favorite color is blue.",
    "I live in Seattle.",
    "I have been coding for 10 years.",
    "My name is Alex.",  # Duplicate for reinforcement test
]


def run_ingestion(seed: int, inputs: list[str]) -> dict:
    """
    Run a complete ingestion and return state hash.
    """
    ingest = DeterministicBeliefIngest(seed=seed)

    for text in inputs:
        ingest.ingest(text)

    return {
        "final_hash": ingest.get_state_hash(),
        "belief_count": len(ingest.get_beliefs()),
        "beliefs": ingest.get_beliefs(),
    }


def test_determinism() -> dict:
    """
    Test that two runs with identical inputs produce identical hashes.
    """
    print("Testing determinism...")

    runs = []

    # Run 1
    print("  Run 1...")
    result1 = run_ingestion(SEED, INPUT_SEQUENCE)
    runs.append({
        "run_id": 1,
        "state_hash": result1["final_hash"],
        "belief_count": result1["belief_count"],
    })

    # Run 2 (identical input)
    print("  Run 2...")
    result2 = run_ingestion(SEED, INPUT_SEQUENCE)
    runs.append({
        "run_id": 2,
        "state_hash": result2["final_hash"],
        "belief_count": result2["belief_count"],
    })

    # Compare
    deterministic = result1["final_hash"] == result2["final_hash"]

    if deterministic:
        print("  PASS: Hashes match")
    else:
        print(f"  FAIL: Hash mismatch")
        print(f"    Run 1: {result1['final_hash']}")
        print(f"    Run 2: {result2['final_hash']}")

    return {
        "seed": SEED,
        "input_count": len(INPUT_SEQUENCE),
        "runs": runs,
        "deterministic": deterministic,
    }


def test_different_seeds() -> dict:
    """
    Verify that different seeds produce different results.
    """
    print("\nTesting seed sensitivity...")

    result1 = run_ingestion(SEED, INPUT_SEQUENCE)
    result2 = run_ingestion(SEED + 1, INPUT_SEQUENCE)

    different = result1["final_hash"] != result2["final_hash"]

    if different:
        print("  PASS: Different seeds produce different hashes")
    else:
        print("  FAIL: Different seeds produced same hash")

    return {
        "seeds_tested": [SEED, SEED + 1],
        "hashes_differ": different,
    }


def main() -> int:
    """Run all determinism tests and write results."""
    print("=" * 50)
    print("Determinism Verification")
    print("=" * 50)

    # Run main determinism test
    det_result = test_determinism()

    # Run seed sensitivity test
    seed_result = test_different_seeds()

    # Combine results
    results = {
        "seed": SEED,
        "runs": det_result["runs"],
        "deterministic": det_result["deterministic"],
        "seed_sensitivity": seed_result,
    }

    # Write results
    output_path = PROJECT_ROOT / "results" / "determinism_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_path}")

    # Return exit code based on determinism
    if det_result["deterministic"]:
        print("\n✓ System is deterministic")
        return 0
    else:
        print("\n✗ System is NOT deterministic")
        return 1


if __name__ == "__main__":
    sys.exit(main())
