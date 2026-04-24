# Author: Bradley R. Kinnard
"""
Run all verification experiments.

Executes the complete verification suite and writes results to results/.
Exits non-zero on any failure.

Usage:
    python experiments/run_all.py
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_python_script(script_path: Path) -> tuple[int, str]:
    """
    Run a Python script and capture output.

    Returns (exit_code, output).
    """
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    output = result.stdout + result.stderr
    return result.returncode, output


def run_drift_benchmark() -> tuple[bool, str]:
    """Run the drift comparison benchmark."""
    print("\n" + "=" * 60)
    print("Running: Drift Benchmark")
    print("=" * 60)

    try:
        # Import and run directly for better error handling
        from experiments.drift_benchmark import run_benchmark

        results = asyncio.run(run_benchmark())

        # Write results
        output_path = PROJECT_ROOT / "results" / "drift_comparison.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results written to: {output_path}")
        return True, "Success"

    except Exception as e:
        return False, str(e)


def run_decay_sweep() -> tuple[bool, str]:
    """Run the decay sensitivity sweep."""
    print("\n" + "=" * 60)
    print("Running: Decay Sweep")
    print("=" * 60)

    script_path = PROJECT_ROOT / "experiments" / "decay_sweep.py"
    exit_code, output = run_python_script(script_path)

    print(output)
    return exit_code == 0, output if exit_code != 0 else "Success"


def run_conflict_resolution() -> tuple[bool, str]:
    """Run conflict resolution tests."""
    print("\n" + "=" * 60)
    print("Running: Conflict Resolution Tests")
    print("=" * 60)

    script_path = PROJECT_ROOT / "tests" / "verification" / "test_conflict_resolution_cases.py"
    exit_code, output = run_python_script(script_path)

    print(output)
    return exit_code == 0, output if exit_code != 0 else "Success"


def run_determinism_check() -> tuple[bool, str]:
    """Run determinism verification."""
    print("\n" + "=" * 60)
    print("Running: Determinism Check")
    print("=" * 60)

    script_path = PROJECT_ROOT / "tests" / "verification" / "test_determinism.py"
    exit_code, output = run_python_script(script_path)

    print(output)
    return exit_code == 0, output if exit_code != 0 else "Success"


def run_offline_verification() -> tuple[bool, str]:
    """Run offline operation verification."""
    print("\n" + "=" * 60)
    print("Running: Offline Verification")
    print("=" * 60)

    script_path = PROJECT_ROOT / "tests" / "verification" / "test_no_network_calls.py"
    exit_code, output = run_python_script(script_path)

    print(output)
    return exit_code == 0, output if exit_code != 0 else "Success"


def verify_results_exist() -> dict:
    """Verify all expected result files exist."""
    expected_files = [
        "drift_comparison.json",
        "determinism_check.json",
        "offline_verification.json",
        "conflict_resolution_log.json",
        "decay_sweep/decay_0.999.json",
        "decay_sweep/decay_0.995.json",
        "decay_sweep/decay_0.99.json",
        "decay_sweep/decay_0.97.json",
        "decay_sweep/decay_0.95.json",
    ]

    results_dir = PROJECT_ROOT / "results"
    status = {}

    for f in expected_files:
        path = results_dir / f
        status[f] = path.exists()

    return status


def main() -> int:
    """Run all experiments and report results."""
    start_time = time.time()

    print("=" * 60)
    print("ABES Verification Suite")
    print("=" * 60)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")

    # Track results
    experiments = []
    all_passed = True

    # Run each experiment
    experiment_runners = [
        ("drift_benchmark", run_drift_benchmark),
        ("decay_sweep", run_decay_sweep),
        ("conflict_resolution", run_conflict_resolution),
        ("determinism_check", run_determinism_check),
        ("offline_verification", run_offline_verification),
    ]

    for name, runner in experiment_runners:
        try:
            passed, message = runner()
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        experiments.append({
            "name": name,
            "passed": passed,
            "message": message if not passed else None,
        })

        if not passed:
            all_passed = False
            print(f"\n⚠ {name} FAILED: {message}")

    # Verify all result files exist
    print("\n" + "=" * 60)
    print("Verifying Result Files")
    print("=" * 60)

    file_status = verify_results_exist()
    for f, exists in file_status.items():
        status = "✓" if exists else "✗"
        print(f"  {status} results/{f}")
        if not exists:
            all_passed = False

    # Write summary
    elapsed = time.time() - start_time
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "all_passed": all_passed,
        "experiments": experiments,
        "result_files": file_status,
    }

    summary_path = PROJECT_ROOT / "results" / "verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Duration: {elapsed:.1f}s")
    print(f"Experiments: {sum(1 for e in experiments if e['passed'])}/{len(experiments)} passed")
    print(f"Result files: {sum(file_status.values())}/{len(file_status)} generated")

    if all_passed:
        print("\n✓ All verifications PASSED")
        return 0
    else:
        print("\n✗ Some verifications FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
