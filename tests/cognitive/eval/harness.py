# Author: Bradley R. Kinnard
"""
Async evaluation harness — runs 1000 prompts against the live ABES API.

Orchestrates: prompt delivery → response capture → semantic scoring →
ecology auditing → stats → reporting. Supports both full and subset runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import httpx

from .ecology_auditor import EcologyAuditor, Violation
from .prompt_bank import EvalPrompt, build_prompts
from .reporter import terminal_summary, write_reports
from .scorer import SemanticScorer, ScoringResult
from .stats import compute_suite_stats

logger = logging.getLogger("abes.eval")


class EvalHarness:
    """Async evaluation harness for the ABES cognitive suite."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        threshold: float = 0.70,
        forbidden_threshold: float = 0.60,
        concurrency: int = 4,
        decay_cycles: int = 5,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.threshold = threshold
        self.forbidden_threshold = forbidden_threshold
        self.concurrency = concurrency
        self.decay_cycles = decay_cycles
        self.timeout = timeout
        self._scorer = SemanticScorer()
        self._auditor = EcologyAuditor()

    async def run(
        self,
        *,
        domains: list[str] | None = None,
        constructs: list[str] | None = None,
        max_prompts: int | None = None,
        output_dir: str = "results/cognitive_eval",
    ) -> dict[str, Any]:
        """Execute the full evaluation pipeline."""
        all_prompts = build_prompts()

        # optional filters
        prompts = _filter_prompts(all_prompts, domains, constructs, max_prompts)
        logger.info("Running %d prompts (of %d total)", len(prompts), len(all_prompts))

        # group prompts by session_group for multi-turn coherence
        groups = _group_prompts(prompts)

        results: list[dict] = []
        sem = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        ) as client:
            # register a test user + get token
            token = await _register_test_user(client)
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            # clear beliefs for a clean slate
            await client.post("/beliefs/clear")

            tasks = []
            for group_key, group_prompts in groups.items():
                tasks.append(
                    self._run_group(client, headers, group_key, group_prompts, sem)
                )

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for gr in group_results:
                if isinstance(gr, Exception):
                    logger.error("Group failed: %s", gr)
                    continue
                results.extend(gr)

        # compute stats
        stats = compute_suite_stats(results)

        # generate reports
        run_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": self.base_url,
            "total_prompts": len(prompts),
            "threshold": self.threshold,
            "forbidden_threshold": self.forbidden_threshold,
            "decay_cycles": self.decay_cycles,
        }

        # terminal output
        print(terminal_summary(stats))

        # disk output
        paths = write_reports(stats, results, output_dir, run_meta)
        for fmt, path in paths.items():
            logger.info("Wrote %s report: %s", fmt, path)

        return {
            "stats": stats,
            "results": results,
            "paths": paths,
        }

    async def _run_group(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        group_key: str,
        prompts: list[EvalPrompt],
        sem: asyncio.Semaphore,
    ) -> list[dict]:
        """Run a group of prompts sharing a session_group sequentially."""
        async with sem:
            results: list[dict] = []
            session_id = None
            created_belief_ids: list[str] = []

            # create session for this group
            resp = await client.post("/chat/sessions")
            if resp.status_code == 200:
                session_id = resp.json().get("id")

            is_long_horizon = any(p.horizon == "long_horizon" for p in prompts)

            for prompt in prompts:
                result = await self._run_prompt(
                    client, headers, prompt, session_id, created_belief_ids
                )
                results.append(result)

                # track created beliefs for ecology auditing
                if "beliefs_created" in result.get("raw_response", {}):
                    created_belief_ids.extend(
                        result["raw_response"]["beliefs_created"]
                    )

                # for long-horizon prompts, run decay cycles between setups
                if is_long_horizon and prompt.is_setup:
                    for _ in range(self.decay_cycles):
                        await client.post("/bel/iterate", json={"context": ""})
                        await asyncio.sleep(0.05)

            return results

    async def _run_prompt(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        prompt: EvalPrompt,
        session_id: str | None,
        known_belief_ids: list[str],
    ) -> dict:
        """Send one prompt, score the response, audit ecology."""
        t0 = time.monotonic()

        # pre-snapshot for ecology checks
        pre_snap = await self._auditor.snapshot_beliefs(client, known_belief_ids)

        # send the message
        payload: dict[str, Any] = {"message": prompt.message}
        if session_id:
            payload["session_id"] = session_id

        try:
            resp = await client.post("/chat/message", json=payload, headers=headers)
            resp_data = resp.json() if resp.status_code == 200 else {}
        except Exception as exc:
            logger.warning("Prompt %d failed: %s", prompt.id, exc)
            resp_data = {}

        assistant_msg = resp_data.get("assistant_message", "")
        elapsed = time.monotonic() - t0

        # track any newly created beliefs
        new_ids = resp_data.get("beliefs_created", [])
        all_ids = list(set(known_belief_ids + new_ids))

        # post-snapshot
        post_snap = await self._auditor.snapshot_beliefs(client, all_ids)

        # score response (skip for setup messages with no gold answer)
        scoring = ScoringResult(
            passed=True, similarity=1.0, forbidden_max=0.0,
            gold_answer="", details=["setup — no scoring"]
        )
        if not prompt.is_setup and prompt.gold_answer:
            scoring = self._scorer.check(
                assistant_msg,
                prompt.gold_answer,
                threshold=self.threshold,
                forbidden=prompt.forbidden_semantics or None,
                forbidden_threshold=self.forbidden_threshold,
            )

        # ecology audit
        violations: list[Violation] = []
        if prompt.ecology_checks:
            violations = self._auditor.audit_invariants(
                pre_snap, post_snap, prompt.ecology_checks, resp_data
            )

        return {
            "id": prompt.id,
            "domain": prompt.domain,
            "construct": prompt.construct,
            "message": prompt.message,
            "gold_answer": prompt.gold_answer,
            "response": assistant_msg,
            "passed": scoring.passed and len(violations) == 0,
            "similarity": scoring.similarity,
            "forbidden_max": scoring.forbidden_max,
            "details": scoring.details,
            "ecology_violations": [asdict(v) for v in violations],
            "is_setup": prompt.is_setup,
            "session_group": prompt.session_group,
            "elapsed_s": round(elapsed, 3),
            "raw_response": resp_data,
        }


# ---- helpers ----

def _filter_prompts(
    prompts: list[EvalPrompt],
    domains: list[str] | None,
    constructs: list[str] | None,
    max_prompts: int | None,
) -> list[EvalPrompt]:
    """Filter by domain/construct with stratified sampling when capped."""
    filtered = prompts
    if domains:
        filtered = [p for p in filtered if p.domain in domains]
    if constructs:
        filtered = [p for p in filtered if p.construct in constructs]
    if max_prompts and len(filtered) > max_prompts:
        # stratified sample: equal share per domain, preserving group order
        from collections import defaultdict
        by_domain: dict[str, list[EvalPrompt]] = defaultdict(list)
        for p in filtered:
            by_domain[p.domain].append(p)
        per = max(1, max_prompts // len(by_domain))
        sampled: list[EvalPrompt] = []
        for dom in sorted(by_domain):
            sampled.extend(by_domain[dom][:per])
        # backfill if rounding left room
        if len(sampled) < max_prompts:
            used = set(p.id for p in sampled)
            for p in filtered:
                if p.id not in used:
                    sampled.append(p)
                    if len(sampled) >= max_prompts:
                        break
        filtered = sampled[:max_prompts]
    return filtered


def _group_prompts(prompts: list[EvalPrompt]) -> dict[str, list[EvalPrompt]]:
    """Group prompts by session_group. Ungrouped prompts get solo groups."""
    groups: dict[str, list[EvalPrompt]] = defaultdict(list)
    solo_counter = 0
    for p in prompts:
        if p.session_group:
            groups[p.session_group].append(p)
        else:
            solo_counter += 1
            groups[f"_solo_{solo_counter}"].append(p)
    return dict(groups)


async def _register_test_user(client: httpx.AsyncClient) -> str | None:
    """Register (or login) an ephemeral test user, return JWT."""
    email = f"eval_{int(time.time())}@test.local"
    payload = {"email": email, "password": "eval_harness_2024", "name": "Eval Bot"}

    try:
        resp = await client.post("/auth/register", json=payload)
        if resp.status_code == 200:
            return resp.json().get("access_token")
        # already exists? try login
        resp = await client.post(
            "/auth/login", json={"email": email, "password": "eval_harness_2024"}
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception as exc:
        logger.warning("Auth setup failed: %s — running unauthenticated", exc)

    return None
