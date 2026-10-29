from __future__ import annotations

from collections.abc import Sequence


def rrf(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def minmax_weighted(
    score_lists: Sequence[dict[str, float]], weights: Sequence[float]
) -> list[tuple[str, float]]:
    """Alternative fusion: min-max normalise each score list then take a weighted sum."""
    fused: dict[str, float] = {}
    for w, scores in zip(weights, score_lists, strict=True):
        if not scores:
            continue
        lo, hi = min(scores.values()), max(scores.values())
        span = hi - lo
        for doc_id, s in scores.items():
            norm = (s - lo) / span if span > 0 else 1.0
            fused[doc_id] = fused.get(doc_id, 0.0) + w * norm
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
