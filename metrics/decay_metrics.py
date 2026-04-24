# Author: Bradley R. Kinnard
"""
Metrics calculation for decay sweep experiments.
"""


def compute_belief_lifetime(belief_history: list[dict]) -> float:
    """
    Compute average belief lifetime in turns.

    Args:
        belief_history: List of belief snapshots with 'created_turn' and 'deprecated_turn'

    Returns:
        Average lifetime in turns
    """
    if not belief_history:
        return 0.0

    lifetimes = []
    for b in belief_history:
        created = b.get("created_turn", 0)
        deprecated = b.get("deprecated_turn")

        if deprecated is not None:
            lifetimes.append(deprecated - created)

    if not lifetimes:
        return float("inf")  # No beliefs deprecated = infinite lifetime

    return sum(lifetimes) / len(lifetimes)


def compute_churn_rate(
    beliefs_created: int,
    beliefs_dropped: int,
    total_turns: int
) -> float:
    """
    Compute belief churn rate.

    Churn = (created + dropped) / (2 * turns)

    Range 0-1 where:
    - 0 = no change
    - 1 = complete turnover every turn
    """
    if total_turns == 0:
        return 0.0

    return (beliefs_created + beliefs_dropped) / (2 * total_turns)


def summarize_decay_run(
    decay_factor: float,
    seed: int,
    duration_hours: float,
    beliefs_created: int,
    beliefs_dropped: int,
    beliefs_retained: int,
    contradictions_detected: int,
    belief_history: list[dict],
) -> dict:
    """
    Summarize a decay sweep run into the required schema.
    """
    total_turns = beliefs_created  # Approximation

    return {
        "decay_factor": decay_factor,
        "run_metadata": {
            "seed": seed,
            "duration_hours": duration_hours,
        },
        "metrics": {
            "average_belief_lifetime": compute_belief_lifetime(belief_history),
            "belief_churn_rate": compute_churn_rate(
                beliefs_created, beliefs_dropped, total_turns
            ),
            "contradictions_detected": contradictions_detected,
            "beliefs_dropped": beliefs_dropped,
            "beliefs_retained": beliefs_retained,
        },
    }


__all__ = [
    "compute_belief_lifetime",
    "compute_churn_rate",
    "summarize_decay_run",
]
