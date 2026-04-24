# Author: Bradley R. Kinnard
"""
ConsolidationAgent - merges/compresses belief lineages and clusters.
Runs as the final phase to prevent belief store bloat.
Combines highly-similar beliefs within a cluster, summarizes long lineage chains.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus, EvidenceRef, OriginMetadata

logger = logging.getLogger(__name__)


# similarity threshold for merging beliefs in the same cluster
MERGE_SIMILARITY_THRESHOLD = 0.92

# max lineage depth before forcing a consolidation
MAX_LINEAGE_DEPTH = 5

# GC thresholds: beliefs below these and with no links get pruned
GC_SALIENCE_FLOOR = 0.02
GC_CONFIDENCE_FLOOR = 0.05


@dataclass
class ConsolidationEvent:
    """What happened during consolidation."""

    event_type: str  # "merged" | "compressed" | "pruned"
    affected_ids: list[UUID]
    result_id: Optional[UUID]
    timestamp: datetime


def _lineage_depth(belief: Belief, belief_map: dict[UUID, Belief]) -> int:
    """Walk parent chain to compute depth."""
    depth = 0
    current = belief
    seen: set[UUID] = set()
    while current.parent_id and current.parent_id not in seen:
        seen.add(current.id)
        depth += 1
        parent = belief_map.get(current.parent_id)
        if parent is None:
            break
        current = parent
    return depth


class ConsolidationAgent:
    """
    Post-loop consolidation: merges duplicate-ish beliefs within clusters
    and compresses deep lineage chains into summary beliefs.
    """

    def __init__(
        self,
        merge_threshold: float = MERGE_SIMILARITY_THRESHOLD,
        max_depth: int = MAX_LINEAGE_DEPTH,
    ):
        self._merge_threshold = merge_threshold
        self._max_depth = max_depth
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    async def consolidate(
        self,
        beliefs: List[Belief],
    ) -> tuple[List[ConsolidationEvent], List[Belief], List[Belief]]:
        """
        Run consolidation on the belief set.

        Returns:
            (events, new_beliefs_to_create, beliefs_to_deprecate)
        """
        events: list[ConsolidationEvent] = []
        to_create: list[Belief] = []
        to_deprecate: list[Belief] = []

        # phase 1: merge near-duplicates within clusters
        merge_events, merged_new, merged_old = await self._merge_duplicates(beliefs)
        events.extend(merge_events)
        to_create.extend(merged_new)
        to_deprecate.extend(merged_old)

        # phase 2: compress deep lineage chains
        belief_map = {b.id: b for b in beliefs}
        compress_events, compressed_new, compressed_old = self._compress_lineages(
            beliefs, belief_map
        )
        events.extend(compress_events)
        to_create.extend(compressed_new)
        to_deprecate.extend(compressed_old)

        # phase 3: garbage collection - prune orphaned low-value beliefs
        gc_events, gc_pruned = self._garbage_collect(beliefs)
        events.extend(gc_events)
        to_deprecate.extend(gc_pruned)

        if events:
            logger.info(
                "consolidation: %d merges, %d compressions, %d gc_pruned, %d retired",
                len(merge_events), len(compress_events), len(gc_events), len(to_deprecate),
            )

        return events, to_create, to_deprecate

    async def _merge_duplicates(
        self, beliefs: List[Belief]
    ) -> tuple[list[ConsolidationEvent], list[Belief], list[Belief]]:
        """Find near-duplicate beliefs in the same cluster and merge them."""
        import numpy as np

        events: list[ConsolidationEvent] = []
        new_beliefs: list[Belief] = []
        deprecated: list[Belief] = []

        # group by cluster
        clusters: dict[Optional[UUID], list[Belief]] = {}
        for b in beliefs:
            if b.status not in (BeliefStatus.Active, BeliefStatus.Decaying):
                continue
            clusters.setdefault(b.cluster_id, []).append(b)

        model = self._get_model()

        for cluster_id, members in clusters.items():
            if len(members) < 2 or cluster_id is None:
                continue

            # embed all members
            texts = [b.content for b in members]
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            # find merge candidates (greedy)
            merged_set: set[int] = set()
            for i in range(len(members)):
                if i in merged_set:
                    continue
                merge_group = [i]
                for j in range(i + 1, len(members)):
                    if j in merged_set:
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim >= self._merge_threshold:
                        merge_group.append(j)
                        merged_set.add(j)

                if len(merge_group) > 1:
                    # pick the strongest belief as the survivor
                    group_beliefs = [members[idx] for idx in merge_group]
                    winner = max(group_beliefs, key=lambda b: b.confidence * b.salience)

                    # absorb evidence from losers
                    for b in group_beliefs:
                        if b.id == winner.id:
                            continue
                        winner.evidence_for.extend(b.evidence_for)
                        winner.evidence_against.extend(b.evidence_against)
                        # absorb links
                        for lnk in b.links:
                            winner.add_link(lnk.target_id, lnk.relation, lnk.weight)
                        b.status = BeliefStatus.Deprecated
                        deprecated.append(b)

                    # boost winner slightly
                    winner.confidence = min(0.99, winner.confidence + 0.02 * (len(merge_group) - 1))
                    merged_set.add(merge_group[0])

                    events.append(ConsolidationEvent(
                        event_type="merged",
                        affected_ids=[members[idx].id for idx in merge_group],
                        result_id=winner.id,
                        timestamp=datetime.now(timezone.utc),
                    ))

        return events, new_beliefs, deprecated

    def _compress_lineages(
        self,
        beliefs: List[Belief],
        belief_map: dict[UUID, Belief],
    ) -> tuple[list[ConsolidationEvent], list[Belief], list[Belief]]:
        """Compress lineage chains that exceed max depth."""
        events: list[ConsolidationEvent] = []
        new_beliefs: list[Belief] = []
        deprecated: list[Belief] = []
        seen_chains: set[UUID] = set()

        for b in beliefs:
            if b.id in seen_chains:
                continue
            if b.status not in (BeliefStatus.Active, BeliefStatus.Decaying):
                continue

            depth = _lineage_depth(b, belief_map)
            if depth < self._max_depth:
                continue

            # collect the chain
            chain: list[Belief] = [b]
            current = b
            while current.parent_id and current.parent_id in belief_map:
                parent = belief_map[current.parent_id]
                chain.append(parent)
                current = parent

            for link in chain:
                seen_chains.add(link.id)

            # the tip (newest) belief survives; ancestors get deprecated
            tip = chain[0]
            ancestors = chain[1:]

            # absorb ancestor evidence into the tip
            for ancestor in ancestors:
                tip.evidence_for.extend(ancestor.evidence_for)
                tip.evidence_against.extend(ancestor.evidence_against)
                for lnk in ancestor.links:
                    tip.add_link(lnk.target_id, lnk.relation, lnk.weight)

                if ancestor.status in (BeliefStatus.Active, BeliefStatus.Decaying):
                    ancestor.status = BeliefStatus.Deprecated
                    deprecated.append(ancestor)

            # clear parent pointer (it's now the root of a fresh lineage)
            tip.parent_id = None

            events.append(ConsolidationEvent(
                event_type="compressed",
                affected_ids=[link.id for link in chain],
                result_id=tip.id,
                timestamp=datetime.now(timezone.utc),
            ))

        return events, new_beliefs, deprecated

    def _garbage_collect(
        self, beliefs: List[Belief]
    ) -> tuple[list[ConsolidationEvent], list[Belief]]:
        """
        Prune beliefs that are effectively dead weight:
        low salience + low confidence + no graph links + not an axiom.
        Keeps the ecology lean by removing orphaned, forgotten beliefs.
        """
        events: list[ConsolidationEvent] = []
        pruned: list[Belief] = []

        # build a set of all belief IDs that are linked-to by other beliefs
        linked_ids: set[UUID] = set()
        for b in beliefs:
            for lnk in b.links:
                linked_ids.add(lnk.target_id)
            if b.parent_id:
                linked_ids.add(b.parent_id)

        for b in beliefs:
            # never GC axioms
            if b.is_axiom:
                continue
            # only GC beliefs that are already decaying, dormant, or deprecated
            if b.status == BeliefStatus.Active and b.confidence > GC_CONFIDENCE_FLOOR:
                continue
            # skip if this belief has any outgoing or incoming links
            has_links = len(b.links) > 0 or b.id in linked_ids
            if has_links:
                continue
            # final check: truly abandoned (low salience AND low confidence)
            if b.salience <= GC_SALIENCE_FLOOR and b.confidence <= GC_CONFIDENCE_FLOOR:
                b.status = BeliefStatus.Deprecated
                pruned.append(b)
                events.append(ConsolidationEvent(
                    event_type="pruned",
                    affected_ids=[b.id],
                    result_id=None,
                    timestamp=datetime.now(timezone.utc),
                ))

        if pruned:
            logger.info("GC pruned %d orphaned beliefs", len(pruned))

        return events, pruned


__all__ = ["ConsolidationAgent", "ConsolidationEvent"]
