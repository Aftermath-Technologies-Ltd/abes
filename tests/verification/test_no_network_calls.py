# Author: Bradley R. Kinnard
"""
Offline operation verification test.

Blocks network sockets at runtime and verifies the experiment suite
can run without any network calls.

Writes results to results/offline_verification.json
"""

import json
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class NetworkBlocker:
    """
    Context manager that blocks all network socket operations.

    Records any attempted network calls for verification.
    """

    def __init__(self):
        self.network_calls: list[dict] = []
        self._original_socket = None
        self._original_connect = None
        self._original_create_connection = None

    def _blocked_socket(self, *args, **kwargs):
        """Replacement socket that records and blocks."""
        call_info = {
            "function": "socket.socket",
            "args": str(args),
            "kwargs": str(kwargs),
        }
        self.network_calls.append(call_info)
        raise OSError("Network access blocked for offline verification")

    def _blocked_connect(self, *args, **kwargs):
        """Replacement connect that records and blocks."""
        call_info = {
            "function": "socket.connect",
            "args": str(args),
            "kwargs": str(kwargs),
        }
        self.network_calls.append(call_info)
        raise OSError("Network access blocked for offline verification")

    def _blocked_create_connection(self, *args, **kwargs):
        """Replacement create_connection that records and blocks."""
        call_info = {
            "function": "socket.create_connection",
            "args": str(args),
            "kwargs": str(kwargs),
        }
        self.network_calls.append(call_info)
        raise OSError("Network access blocked for offline verification")

    def __enter__(self):
        # Store originals
        self._original_socket = socket.socket
        self._original_create_connection = socket.create_connection

        # Replace with blockers
        socket.socket = self._blocked_socket
        socket.create_connection = self._blocked_create_connection

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore originals
        socket.socket = self._original_socket
        socket.create_connection = self._original_create_connection

        return False


def run_offline_experiments(blocker: NetworkBlocker) -> dict:
    """
    Run experiments that should work offline.

    Returns dict with experiment results.
    """
    results = {
        "experiments_run": [],
        "experiments_failed": [],
    }

    # Test 1: Deterministic belief ingest
    print("  Testing belief ingest...", end=" ")
    try:
        from interfaces.belief_ingest import DeterministicBeliefIngest

        ingest = DeterministicBeliefIngest(seed=12345)
        ingest.ingest("My name is Alex")
        ingest.ingest("I like coffee")

        results["experiments_run"].append("belief_ingest")
        print("OK")
    except Exception as e:
        results["experiments_failed"].append({"name": "belief_ingest", "error": str(e)})
        print(f"FAIL: {e}")

    # Test 2: Conflict resolution
    print("  Testing conflict resolution...", end=" ")
    try:
        from beliefs.conflict_resolution import ConflictResolver, Belief
        from uuid import uuid4

        resolver = ConflictResolver()
        b1 = Belief(id=uuid4(), content="A", confidence=0.8, created_at_turn=0)
        b2 = Belief(id=uuid4(), content="B", confidence=0.7, created_at_turn=5)
        resolver.resolve(b1, b2, current_turn=10)

        results["experiments_run"].append("conflict_resolution")
        print("OK")
    except Exception as e:
        results["experiments_failed"].append({"name": "conflict_resolution", "error": str(e)})
        print(f"FAIL: {e}")

    # Test 3: Baseline systems
    print("  Testing baselines...", end=" ")
    try:
        from baselines.plain_llm_runner import PlainLLMRunner
        from baselines.append_only_memory import AppendOnlyMemory

        plain = PlainLLMRunner(seed=12345)
        plain.process_turn("Hello")

        append = AppendOnlyMemory(seed=12345)
        append.process_turn("My name is Alex")

        results["experiments_run"].append("baselines")
        print("OK")
    except Exception as e:
        results["experiments_failed"].append({"name": "baselines", "error": str(e)})
        print(f"FAIL: {e}")

    # Test 4: Metrics computation
    print("  Testing metrics...", end=" ")
    try:
        from metrics.drift_metrics import compute_belief_entropy, count_potential_contradictions

        beliefs = [{"content": "I like coffee"}, {"content": "I hate coffee"}]
        entropy = compute_belief_entropy(beliefs)
        contradictions = count_potential_contradictions(beliefs)

        results["experiments_run"].append("metrics")
        print("OK")
    except Exception as e:
        results["experiments_failed"].append({"name": "metrics", "error": str(e)})
        print(f"FAIL: {e}")

    # Test 5: Decay simulation
    print("  Testing decay simulation...", end=" ")
    try:
        from experiments.decay_sweep import DecaySimulator

        sim = DecaySimulator(decay_factor=0.995, seed=12345)
        sim.add_belief("Test belief")
        sim.advance_turn()

        results["experiments_run"].append("decay_simulation")
        print("OK")
    except Exception as e:
        results["experiments_failed"].append({"name": "decay_simulation", "error": str(e)})
        print(f"FAIL: {e}")

    return results


def main() -> int:
    """Run offline verification and write results."""
    print("=" * 50)
    print("Offline Operation Verification")
    print("=" * 50)
    print("\nBlocking network access...")

    with NetworkBlocker() as blocker:
        print(f"\nRunning experiments with network blocked...")
        experiment_results = run_offline_experiments(blocker)
        network_calls = len(blocker.network_calls)

    # Build results
    results = {
        "network_calls_detected": network_calls,
        "network_call_details": blocker.network_calls if network_calls > 0 else [],
        "experiments_run": experiment_results["experiments_run"],
        "experiments_failed": experiment_results["experiments_failed"],
        "status": "pass" if network_calls == 0 else "fail",
    }

    # Write results
    output_path = PROJECT_ROOT / "results" / "offline_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_path}")

    # Summary
    print("\n" + "=" * 50)
    if network_calls == 0:
        print("✓ PASS: No network calls detected")
        print(f"  Experiments run: {len(experiment_results['experiments_run'])}")
        return 0
    else:
        print(f"✗ FAIL: {network_calls} network call(s) detected")
        for call in blocker.network_calls:
            print(f"  - {call['function']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
