# Author: Bradley R. Kinnard
"""
Test conflict resolution cases.

Verifies that the conflict resolver behaves consistently and
demonstrates consistency enforcement (not truth inference).

Writes results to results/conflict_resolution_log.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from beliefs.conflict_resolution import (
    Belief,
    ConflictResolver,
    ResolutionAction,
    generate_test_cases,
)
from uuid import uuid4


def test_equal_strength_contradiction():
    """Test that equal-strength contradictions are deferred."""
    resolver = ConflictResolver()

    b1 = Belief(id=uuid4(), content="A is true", confidence=0.8, created_at_turn=0)
    b2 = Belief(id=uuid4(), content="A is false", confidence=0.8, created_at_turn=5)

    result = resolver.resolve(b1, b2, current_turn=10)

    assert result.resolution_action == ResolutionAction.DEFER, \
        f"Expected DEFER for equal strength, got {result.resolution_action}"
    assert len(result.resulting_beliefs) == 2, \
        "Both beliefs should be retained when deferred"

    return result


def test_clear_winner_contradiction():
    """Test that clear confidence gap leads to WEAKEN."""
    resolver = ConflictResolver()

    b1 = Belief(id=uuid4(), content="I like X", confidence=0.9, created_at_turn=0)
    b2 = Belief(id=uuid4(), content="I hate X", confidence=0.6, created_at_turn=5)

    result = resolver.resolve(b1, b2, current_turn=10)

    assert result.resolution_action == ResolutionAction.WEAKEN, \
        f"Expected WEAKEN for confidence gap, got {result.resolution_action}"

    # Stronger belief should remain unchanged
    strong_belief = next(b for b in result.resulting_beliefs if b.confidence == 0.9)
    assert strong_belief.content == "I like X"

    # Weaker belief should be further weakened
    weak_belief = next(b for b in result.resulting_beliefs if b.content == "I hate X")
    assert weak_belief.confidence < 0.6, \
        f"Weaker belief should be further weakened, got {weak_belief.confidence}"

    return result


def test_age_difference():
    """Test that significant age difference leads to WEAKEN of older."""
    resolver = ConflictResolver()

    b1 = Belief(id=uuid4(), content="Old fact", confidence=0.8, created_at_turn=0)
    b2 = Belief(id=uuid4(), content="New fact", confidence=0.75, created_at_turn=50)

    result = resolver.resolve(b1, b2, current_turn=60)

    assert result.resolution_action == ResolutionAction.WEAKEN, \
        f"Expected WEAKEN for age difference, got {result.resolution_action}"

    # Old belief should be weakened
    old_belief = next(b for b in result.resulting_beliefs if b.content == "Old fact")
    assert old_belief.confidence < 0.8, \
        f"Older belief should be weakened, got {old_belief.confidence}"

    return result


def test_determinism():
    """Test that resolution is deterministic."""
    resolver = ConflictResolver()

    # Run same case twice
    b1 = Belief(id=uuid4(), content="Test A", confidence=0.85, created_at_turn=0)
    b2 = Belief(id=uuid4(), content="Test B", confidence=0.7, created_at_turn=5)

    result1 = resolver.resolve(b1, b2, current_turn=10)
    result2 = resolver.resolve(b1, b2, current_turn=10)

    assert result1.resolution_action == result2.resolution_action, \
        "Resolution action should be deterministic"
    assert result1.notes == result2.notes, \
        "Resolution notes should be deterministic"

    return result1


def run_all_tests() -> dict:
    """Run all tests and return results log."""
    cases = []

    print("Running conflict resolution tests...")

    # Run individual tests
    tests = [
        ("equal_strength_contradiction", test_equal_strength_contradiction),
        ("clear_winner_contradiction", test_clear_winner_contradiction),
        ("age_difference", test_age_difference),
        ("determinism", test_determinism),
    ]

    for name, test_fn in tests:
        print(f"  {name}...", end=" ")
        try:
            result = test_fn()
            result_dict = result.to_dict()
            result_dict["test_name"] = name
            result_dict["status"] = "pass"
            cases.append(result_dict)
            print("PASS")
        except AssertionError as e:
            print(f"FAIL: {e}")
            cases.append({
                "test_name": name,
                "status": "fail",
                "error": str(e),
            })

    # Add generated test cases
    print("\n  Generating additional test cases...")
    for case in generate_test_cases():
        case_dict = case.to_dict()
        case_dict["test_name"] = f"generated_{case.case_id[:8]}"
        case_dict["status"] = "generated"
        cases.append(case_dict)

    return {"cases": cases}


def main() -> int:
    """Run tests and write results."""
    results = run_all_tests()

    # Write results
    output_path = PROJECT_ROOT / "results" / "conflict_resolution_log.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {output_path}")

    # Count results
    passed = sum(1 for c in results["cases"] if c.get("status") == "pass")
    failed = sum(1 for c in results["cases"] if c.get("status") == "fail")
    generated = sum(1 for c in results["cases"] if c.get("status") == "generated")

    print(f"\nSummary: {passed} passed, {failed} failed, {generated} generated")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
