# Author: Bradley R. Kinnard
"""
Ecology auditor — inspects internal belief state via GET /beliefs/{id}/ecology.

Verifies that the belief ecology's internal dynamics (tension, salience, evidence,
mutation, dormancy) work as claimed. This is what separates "the LLM answered correctly"
from "the cognitive architecture is actually doing its job."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BeliefState:
    """Snapshot of one belief's ecology internals."""
    id: str
    content: str
    confidence: float
    tension: float
    salience: float
    half_life_days: float
    evidence_for_count: int
    evidence_against_count: int
    evidence_balance: float
    link_count: int
    links: list[dict]
    status: str


@dataclass
class EcologySnapshot:
    """Collection of belief states at a point in time."""
    beliefs: dict[str, BeliefState] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tension(self) -> float:
        return sum(b.tension for b in self.beliefs.values())

    @property
    def avg_salience(self) -> float:
        if not self.beliefs:
            return 0.0
        return sum(b.salience for b in self.beliefs.values()) / len(self.beliefs)


@dataclass
class Violation:
    """A failed ecology invariant check."""
    invariant: str
    belief_id: str
    message: str
    severity: str = "error"    # "error" or "warning"


class EcologyAuditor:
    """Inspects and audits belief ecology internals via the API."""

    async def snapshot_beliefs(self, client, belief_ids: list[str]) -> EcologySnapshot:
        """Fetch ecology state for specific beliefs."""
        snap = EcologySnapshot()
        for bid in belief_ids:
            resp = await client.get(f"/beliefs/{bid}/ecology")
            if resp.status_code == 200:
                data = resp.json()
                snap.beliefs[bid] = BeliefState(
                    id=data["id"],
                    content=data["content"],
                    confidence=data["confidence"],
                    tension=data["tension"],
                    salience=data["salience"],
                    half_life_days=data["half_life_days"],
                    evidence_for_count=data["evidence_for_count"],
                    evidence_against_count=data["evidence_against_count"],
                    evidence_balance=data["evidence_balance"],
                    link_count=data["link_count"],
                    links=data.get("links", []),
                    status=data["status"],
                )
        return snap

    async def snapshot_stats(self, client) -> dict:
        """Fetch aggregate ecology stats."""
        resp = await client.get("/bel/stats")
        if resp.status_code == 200:
            return resp.json()
        return {}

    def audit_invariants(
        self,
        pre: EcologySnapshot,
        post: EcologySnapshot,
        checks: list[str],
        response_data: dict | None = None,
    ) -> list[Violation]:
        """Run invariant checks between pre/post snapshots."""
        violations: list[Violation] = []

        for check in checks:
            handler = _INVARIANT_HANDLERS.get(check)
            if handler:
                violations.extend(handler(pre, post, response_data or {}))

        return violations


# ========= Invariant handlers =========

def _check_belief_created(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """New beliefs should appear with evidence_for_count >= 1."""
    violations = []
    created_ids = resp.get("beliefs_created", [])
    for bid in created_ids:
        if bid in post.beliefs:
            b = post.beliefs[bid]
            if b.evidence_for_count < 1:
                violations.append(Violation(
                    invariant="belief_created",
                    belief_id=bid,
                    message=f"New belief has {b.evidence_for_count} supporting evidence (expected >= 1)",
                ))
    return violations


def _check_reinforcement_boost(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """Reinforced beliefs: confidence and salience should increase."""
    violations = []
    reinforced_ids = resp.get("beliefs_reinforced", [])
    for bid in reinforced_ids:
        if bid in pre.beliefs and bid in post.beliefs:
            before, after = pre.beliefs[bid], post.beliefs[bid]
            if after.confidence < before.confidence - 0.001:
                violations.append(Violation(
                    invariant="reinforcement_boost",
                    belief_id=bid,
                    message=f"Confidence dropped {before.confidence:.3f} -> {after.confidence:.3f} on reinforcement",
                ))
            if after.salience < before.salience - 0.001:
                violations.append(Violation(
                    invariant="reinforcement_boost",
                    belief_id=bid,
                    message=f"Salience dropped {before.salience:.3f} -> {after.salience:.3f} on reinforcement",
                    severity="warning",
                ))
    return violations


def _check_contradiction_tension(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """Contradictions should increase tension on at least one involved belief."""
    violations = []
    events = resp.get("events", [])
    tension_events = [e for e in events if e.get("event_type") == "tension_changed"]
    if not tension_events:
        return violations

    # at least one belief should have higher tension
    any_tension_increased = False
    for bid in post.beliefs:
        if bid in pre.beliefs:
            if post.beliefs[bid].tension > pre.beliefs[bid].tension + 0.001:
                any_tension_increased = True
                break

    if not any_tension_increased:
        violations.append(Violation(
            invariant="contradiction_tension",
            belief_id="aggregate",
            message="Tension event fired but no belief's tension increased",
        ))
    return violations


def _check_mutation_on_high_tension(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """High-tension beliefs (> 0.5) should trigger mutation."""
    violations = []
    mutated_ids = set(resp.get("beliefs_mutated", []))
    for bid, b in post.beliefs.items():
        if b.tension > 0.5 and bid not in mutated_ids:
            violations.append(Violation(
                invariant="mutation_on_high_tension",
                belief_id=bid,
                message=f"Tension {b.tension:.3f} > 0.5 but no mutation triggered",
                severity="warning",
            ))
    return violations


def _check_salience_decay(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """Salience should not increase without reinforcement."""
    violations = []
    reinforced_ids = set(resp.get("beliefs_reinforced", []))
    for bid in pre.beliefs:
        if bid in post.beliefs and bid not in reinforced_ids:
            before, after = pre.beliefs[bid], post.beliefs[bid]
            if after.salience > before.salience + 0.01:
                violations.append(Violation(
                    invariant="salience_decay",
                    belief_id=bid,
                    message=f"Salience increased {before.salience:.3f} -> {after.salience:.3f} without reinforcement",
                ))
    return violations


def _check_dormancy_threshold(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """Beliefs with salience near zero should be dormant."""
    violations = []
    for bid, b in post.beliefs.items():
        if b.salience < 0.05 and b.status == "active":
            violations.append(Violation(
                invariant="dormancy_threshold",
                belief_id=bid,
                message=f"Salience {b.salience:.4f} < 0.05 but status is '{b.status}' (expected dormant)",
                severity="warning",
            ))
    return violations


def _check_no_orphan_links(
    pre: EcologySnapshot, post: EcologySnapshot, resp: dict
) -> list[Violation]:
    """All link target_ids should reference existing beliefs."""
    violations = []
    all_ids = set(post.beliefs.keys())
    for bid, b in post.beliefs.items():
        for link in b.links:
            target = link.get("target_id", "")
            if target and target not in all_ids:
                violations.append(Violation(
                    invariant="no_orphan_links",
                    belief_id=bid,
                    message=f"Link to {target} but that belief not in snapshot",
                    severity="warning",
                ))
    return violations


# registry of invariant name -> handler
_INVARIANT_HANDLERS = {
    "belief_created": _check_belief_created,
    "reinforcement_boost": _check_reinforcement_boost,
    "contradiction_tension": _check_contradiction_tension,
    "mutation_on_high_tension": _check_mutation_on_high_tension,
    "salience_decay": _check_salience_decay,
    "dormancy_threshold": _check_dormancy_threshold,
    "no_orphan_links": _check_no_orphan_links,
}
