# Author: Bradley R. Kinnard
"""
Statistical analysis for the 1000-prompt evaluation suite.

Clopper-Pearson exact binomial CI, Cronbach's alpha, domain distributions.
Uses numpy only (already a project dependency).

Reference: Clopper & Pearson (1934), https://doi.org/10.1093/biomet/26.4.404
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, exp

import numpy as np


@dataclass
class DomainStats:
    """Per-domain score summary."""
    domain: str
    total: int
    passed: int
    pass_rate: float
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    ecology_violations: int


@dataclass
class SuiteStats:
    """Aggregate stats for the full evaluation run."""
    total: int
    passed: int
    pass_rate: float
    ci_lower: float
    ci_upper: float
    ci_alpha: float
    mean_similarity: float
    std_similarity: float
    domains: dict[str, DomainStats]
    ecology_violations_total: int


def clopper_pearson_ci(
    successes: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Exact binomial confidence interval (Clopper-Pearson 1934).

    Uses the beta distribution quantile. Falls back to a pure-Python
    implementation so we don't need scipy.
    """
    if n == 0:
        return (0.0, 1.0)
    if successes == 0:
        return (0.0, 1.0 - (alpha / 2.0) ** (1.0 / n))
    if successes == n:
        return ((alpha / 2.0) ** (1.0 / n), 1.0)

    lower = _beta_ppf(alpha / 2.0, successes, n - successes + 1)
    upper = _beta_ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
    return (lower, upper)


def domain_distributions(results: list[dict]) -> dict[str, DomainStats]:
    """Compute per-domain score distributions."""
    from collections import defaultdict
    by_domain: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        by_domain[r["domain"]].append(r)

    out = {}
    for domain, items in sorted(by_domain.items()):
        passed = sum(1 for r in items if r["passed"])
        sims = [r["similarity"] for r in items]
        violations = sum(len(r.get("ecology_violations", [])) for r in items)
        out[domain] = DomainStats(
            domain=domain,
            total=len(items),
            passed=passed,
            pass_rate=passed / len(items) if items else 0.0,
            mean_similarity=float(np.mean(sims)) if sims else 0.0,
            std_similarity=float(np.std(sims)) if sims else 0.0,
            min_similarity=float(np.min(sims)) if sims else 0.0,
            max_similarity=float(np.max(sims)) if sims else 0.0,
            ecology_violations=violations,
        )
    return out


def compute_suite_stats(results: list[dict], alpha: float = 0.05) -> SuiteStats:
    """Full statistical summary across all results."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    sims = [r["similarity"] for r in results]
    ci_lower, ci_upper = clopper_pearson_ci(passed, total, alpha)
    domains = domain_distributions(results)
    violations = sum(len(r.get("ecology_violations", [])) for r in results)

    return SuiteStats(
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_alpha=alpha,
        mean_similarity=float(np.mean(sims)) if sims else 0.0,
        std_similarity=float(np.std(sims)) if sims else 0.0,
        domains=domains,
        ecology_violations_total=violations,
    )


def test_retest_correlation(run1_sims: list[float], run2_sims: list[float]) -> float:
    """Pearson r between similarity scores from two runs."""
    if len(run1_sims) != len(run2_sims) or len(run1_sims) < 2:
        return 0.0
    r = np.corrcoef(run1_sims, run2_sims)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


# ---- Beta distribution quantile (pure Python, no scipy needed) ----

def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse CDF of Beta(a, b) via bisection. Accurate to ~1e-9."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function via continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # use the relation I_x(a,b) = 1 - I_{1-x}(b,a) for numerical stability
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _beta_cdf(1.0 - x, b, a)

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * np.log(x) + b * np.log(1.0 - x) - lbeta) / a

    # Lentz's continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        # even step
        num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= c * d

        # odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return front * f
