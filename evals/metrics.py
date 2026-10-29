"""Ranking metrics implemented from scratch and cross-checked against scikit-learn in tests.

nDCG uses linear gain (gain = relevance label) and a log2(rank + 1) discount, the convention
used by trec_eval and by the BEIR benchmark. A document counts as relevant when its label > 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def ndcg_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    gains = [max(qrel.get(d, 0), 0) for d in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in qrel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


@dataclass
class RunScores:
    qids: list[str]
    per_query: dict[str, np.ndarray]  # metric name -> value per qid (aligned with qids)

    def mean(self) -> dict[str, float]:
        return {m: float(v.mean()) if len(v) else 0.0 for m, v in self.per_query.items()}


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    qids: Sequence[str] | None = None,
    ks: tuple[int, ...] = (10, 100),
) -> RunScores:
    """Score a run. Queries with no relevant judgement are skipped. Missing run entries score 0."""
    pool = qids if qids is not None else sorted(qrels)
    keep = [q for q in pool if q in qrels and any(g > 0 for g in qrels[q].values())]
    out: dict[str, list[float]] = {f"ndcg@{ks[0]}": [], f"mrr@{ks[0]}": []}
    for k in ks:
        out[f"recall@{k}"] = []
    for q in keep:
        ranked = run.get(q, [])
        out[f"ndcg@{ks[0]}"].append(ndcg_at_k(ranked, qrels[q], ks[0]))
        out[f"mrr@{ks[0]}"].append(mrr_at_k(ranked, qrels[q], ks[0]))
        for k in ks:
            out[f"recall@{k}"].append(recall_at_k(ranked, qrels[q], k))
    return RunScores(keep, {m: np.asarray(v, dtype=np.float64) for m, v in out.items()})


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap over queries for mean(a) - mean(b).

    Returns the observed difference, a 95% percentile interval, and a two-sided p-value
    estimated as twice the smaller tail mass of the bootstrap distribution.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be equal-length non-empty vectors")
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {
        "diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": min(p, 1.0),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
