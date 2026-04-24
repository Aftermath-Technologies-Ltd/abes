# Author: Bradley R. Kinnard
"""Analytics routes: belief fitness rankings and tension-based cluster summaries."""

from fastapi import APIRouter

from ...core.deps import get_belief_store
from ...core.models.belief import BeliefStatus
from ...agents.consolidation import compute_belief_fitness
from ...agents.mutation_engineer import cluster_high_tension_beliefs

router = APIRouter(prefix="/beliefs", tags=["analytics"])


@router.get("/fitness")
async def get_belief_fitness():
    """Return all active beliefs sorted by fitness descending, with tier stats.

    Fitness per belief = confidence * salience * max(use_count, 1).
    Axioms receive a dominant bonus so they always appear first.

    Returns:
        JSON with ``beliefs`` (sorted list) and ``stats`` (aggregate metrics
        and per-tier utilization from the store's get_tier_stats()).
    """
    store = get_belief_store()
    beliefs = await store.list(limit=10000)
    active = [b for b in beliefs if b.status in (BeliefStatus.Active, BeliefStatus.Decaying)]

    scored = sorted(
        [{"belief": b, "fitness": compute_belief_fitness(b)} for b in active],
        key=lambda x: x["fitness"],
        reverse=True,
    )

    belief_list = [
        {
            "id": str(item["belief"].id),
            "content_summary": item["belief"].content[:100],
            "fitness": round(item["fitness"], 6),
            "tier": item["belief"].memory_tier,
            "tension": item["belief"].tension,
        }
        for item in scored
    ]

    mean_fitness = float(
        sum(x["fitness"] for x in scored) / len(scored)
    ) if scored else 0.0

    tier_stats: dict = {}
    if hasattr(store, "get_tier_stats"):
        tier_stats = await store.get_tier_stats()

    return {
        "beliefs": belief_list,
        "stats": {
            "total": len(belief_list),
            "mean_fitness": round(mean_fitness, 6),
            "tiers": tier_stats.get("tiers", {}),
        },
    }


@router.get("/clusters")
async def get_belief_clusters():
    """Group active beliefs into tension-based clusters.

    Uses token-overlap clustering on beliefs whose ``tension`` field is above
    the threshold. For each cluster, returns aggregate tension, fitness, and
    the full list of belief IDs.

    Returns:
        JSON with a ``clusters`` key containing a list of cluster summaries.
    """
    store = get_belief_store()
    beliefs = await store.list(limit=10000)
    active = [b for b in beliefs if b.status in (BeliefStatus.Active, BeliefStatus.Decaying)]

    tension_scores = {b.id: b.tension for b in active}
    raw_clusters = cluster_high_tension_beliefs(
        active, tension_scores, tension_threshold=0.1, overlap_threshold=0.2
    )

    result = []
    for i, cluster in enumerate(raw_clusters):
        mean_tension = sum(b.tension for b in cluster) / len(cluster)
        mean_fitness = sum(compute_belief_fitness(b) for b in cluster) / len(cluster)
        top = max(cluster, key=lambda b: compute_belief_fitness(b))
        result.append({
            "cluster_id": str(i),
            "size": len(cluster),
            "mean_tension": round(mean_tension, 6),
            "mean_fitness": round(mean_fitness, 6),
            "top_belief_summary": top.content[:100],
            "belief_ids": [str(b.id) for b in cluster],
        })

    return {"clusters": result}
