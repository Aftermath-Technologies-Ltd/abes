# Author: Bradley R. Kinnard
"""
Reporter — renders evaluation results as Markdown, JSON, and terminal summaries.

Generates human-readable reports from SuiteStats + per-prompt results.
Optionally writes to disk for CI artifact collection.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .stats import SuiteStats, DomainStats


def terminal_summary(stats: SuiteStats) -> str:
    """Compact terminal-friendly summary with pass/fail indicator."""
    icon = "PASS" if stats.pass_rate >= 0.70 else "FAIL"
    ci = f"[{stats.ci_lower:.3f}, {stats.ci_upper:.3f}]"
    eco = stats.ecology_violations_total

    lines = [
        f"\n{'=' * 60}",
        f"  ABES Cognitive Eval  —  {icon}",
        f"{'=' * 60}",
        f"  Total prompts : {stats.total}",
        f"  Passed        : {stats.passed}/{stats.total} ({stats.pass_rate:.1%})",
        f"  95% CI        : {ci}",
        f"  Mean cosine   : {stats.mean_similarity:.3f} +/- {stats.std_similarity:.3f}",
        f"  Ecology viols : {eco}",
        f"{'-' * 60}",
    ]

    # per-domain breakdown
    for dom in sorted(stats.domains.values(), key=lambda d: d.domain):
        bar = _bar(dom.pass_rate, 20)
        viols = f"  [{dom.ecology_violations} viols]" if dom.ecology_violations else ""
        lines.append(
            f"  {dom.domain:<26} {dom.passed:>3}/{dom.total:<3} "
            f"{dom.pass_rate:>5.1%} {bar} "
            f"cos={dom.mean_similarity:.2f}{viols}"
        )

    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)


def markdown_report(
    stats: SuiteStats,
    results: list[dict],
    run_meta: dict[str, Any] | None = None,
) -> str:
    """Full Markdown report suitable for CI artifacts or docs."""
    meta = run_meta or {}
    ts = meta.get("timestamp", datetime.now(timezone.utc).isoformat())
    model = meta.get("model", "unknown")

    md = [
        "# ABES Cognitive Evaluation Report",
        "",
        f"**Date**: {ts}  ",
        f"**Model**: {model}  ",
        f"**Total prompts**: {stats.total}  ",
        f"**Pass rate**: {stats.pass_rate:.1%} "
        f"(95% CI: [{stats.ci_lower:.3f}, {stats.ci_upper:.3f}])  ",
        f"**Mean cosine similarity**: {stats.mean_similarity:.3f} "
        f"+/- {stats.std_similarity:.3f}  ",
        f"**Ecology violations**: {stats.ecology_violations_total}  ",
        "",
        "## Domain Breakdown",
        "",
        "| Domain | Passed | Rate | Mean cos | Violations |",
        "|--------|--------|------|----------|------------|",
    ]

    for dom in sorted(stats.domains.values(), key=lambda d: d.domain):
        md.append(
            f"| {dom.domain} | {dom.passed}/{dom.total} | "
            f"{dom.pass_rate:.1%} | {dom.mean_similarity:.3f} | "
            f"{dom.ecology_violations} |"
        )

    # failures table — only the misses
    failures = [r for r in results if not r["passed"]]
    if failures:
        md.extend([
            "",
            "## Failed Prompts",
            "",
            "| ID | Domain | Construct | Similarity | Details |",
            "|----|--------|-----------|------------|---------|",
        ])
        for f in failures[:50]:     # cap at 50 to keep report readable
            details = "; ".join(f.get("details", []))[:80]
            md.append(
                f"| {f['id']} | {f['domain']} | {f['construct']} | "
                f"{f['similarity']:.3f} | {details} |"
            )
        if len(failures) > 50:
            md.append(f"\n*...and {len(failures) - 50} more failures omitted.*")

    # ecology violations detail
    viols = [
        v for r in results
        for v in r.get("ecology_violations", [])
    ]
    if viols:
        md.extend([
            "",
            "## Ecology Violations",
            "",
            "| Invariant | Belief ID | Severity | Message |",
            "|-----------|-----------|----------|---------|",
        ])
        for v in viols[:30]:
            msg = v.get("message", "")[:60]
            md.append(
                f"| {v['invariant']} | {v.get('belief_id', '?')[:8]}... | "
                f"{v.get('severity', 'error')} | {msg} |"
            )

    return "\n".join(md)


def json_report(
    stats: SuiteStats,
    results: list[dict],
    run_meta: dict[str, Any] | None = None,
) -> dict:
    """Structured dict for JSON serialization."""
    return {
        "meta": run_meta or {},
        "summary": {
            "total": stats.total,
            "passed": stats.passed,
            "pass_rate": round(stats.pass_rate, 4),
            "ci_lower": round(stats.ci_lower, 4),
            "ci_upper": round(stats.ci_upper, 4),
            "mean_similarity": round(stats.mean_similarity, 4),
            "std_similarity": round(stats.std_similarity, 4),
            "ecology_violations": stats.ecology_violations_total,
        },
        "domains": {
            name: {
                "total": d.total,
                "passed": d.passed,
                "pass_rate": round(d.pass_rate, 4),
                "mean_similarity": round(d.mean_similarity, 4),
                "ecology_violations": d.ecology_violations,
            }
            for name, d in stats.domains.items()
        },
        "results": results,
    }


def write_reports(
    stats: SuiteStats,
    results: list[dict],
    output_dir: str | Path = "results/cognitive_eval",
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write all report formats to disk. Returns paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    paths: dict[str, Path] = {}

    # JSON (full)
    json_path = out / f"eval_{ts}.json"
    data = json_report(stats, results, run_meta)
    json_path.write_text(json.dumps(data, indent=2, default=str))
    paths["json"] = json_path

    # Markdown
    md_path = out / f"eval_{ts}.md"
    md_path.write_text(markdown_report(stats, results, run_meta))
    paths["markdown"] = md_path

    # Also write a latest symlink / copy
    latest_json = out / "latest.json"
    latest_json.write_text(json.dumps(data, indent=2, default=str))
    paths["latest_json"] = latest_json

    return paths


# ---- helpers ----

def _bar(ratio: float, width: int = 20) -> str:
    """ASCII progress bar."""
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"
