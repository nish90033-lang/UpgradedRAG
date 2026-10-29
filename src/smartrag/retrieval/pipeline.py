from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .bm25 import BM25Index, top_k
from .embeddings import Embedder
from .fusion import rrf
from .rerank import Reranker
from .tokenize import tokenize

# dense_fn(query_vector, k, min_trust) -> [(chunk_id, similarity)], best first
DenseFn = Callable[[np.ndarray, int, float], list[tuple[str, float]]]


@dataclass(frozen=True)
class PoolChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    idx: int
    text: str
    trust: float = 100.0


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # "bm25" | "dense" | "hybrid"
    bm25_k: int = 100
    dense_k: int = 100
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_dense: float = 1.0
    rerank_top_n: int = 30
    final_k: int = 8
    max_per_doc: int | None = None
    trust_threshold: float = 50.0
    abstain_threshold: float | None = None  # applies to the top rerank score only


@dataclass
class Hit:
    chunk: PoolChunk
    score: float
    fused: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank: float | None = None


@dataclass
class RetrievalResult:
    answerable: bool
    hits: list[Hit] = field(default_factory=list)
    blocked_reason: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    pool_size: int = 0
    blocked_chunks: int = 0


class ArrayDenseIndex:
    """Exact cosine search over an in-memory matrix of L2-normalised vectors."""

    def __init__(self, ids: list[str], matrix: np.ndarray, trust: Sequence[float] | None = None):
        self.ids = ids
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.trust = np.asarray(trust if trust is not None else [100.0] * len(ids))

    def __call__(self, qvec: np.ndarray, k: int, min_trust: float = 0.0) -> list[tuple[str, float]]:
        if not len(self.ids):
            return []
        sims = self.matrix @ np.asarray(qvec, dtype=np.float32)
        sims = np.where(self.trust >= min_trust, sims, -np.inf)
        return [(self.ids[i], s) for i, s in top_k(sims, k) if np.isfinite(s)]


class Retriever:
    """Trust filter, then BM25 and dense search, RRF fusion, and optional cross-encoder rerank."""

    def __init__(
        self, embedder: Embedder | None, reranker: Reranker | None = None, cache_size: int = 8
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self._bm25_cache: OrderedDict[str, BM25Index] = OrderedDict()
        self._cache_size = cache_size

    def _bm25_for(self, pool: Sequence[PoolChunk]) -> BM25Index:
        key = hashlib.sha1("\x00".join(c.chunk_id for c in pool).encode()).hexdigest()
        index = self._bm25_cache.get(key)
        if index is None:
            index = BM25Index().fit([tokenize(c.text) for c in pool])
            self._bm25_cache[key] = index
            while len(self._bm25_cache) > self._cache_size:
                self._bm25_cache.popitem(last=False)
        else:
            self._bm25_cache.move_to_end(key)
        return index

    def retrieve(
        self,
        query: str,
        pool: Sequence[PoolChunk],
        dense_fn: DenseFn | None,
        cfg: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = cfg or RetrievalConfig()
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        # Layer 3: blocked chunks never reach any retrieval stage.
        allowed = [c for c in pool if c.trust >= cfg.trust_threshold]
        blocked = len(pool) - len(allowed)
        t["filter_ms"] = (time.perf_counter() - t0) * 1000
        result = RetrievalResult(answerable=False, pool_size=len(pool), blocked_chunks=blocked)
        if not pool:
            result.blocked_reason = "no_documents"
            return result
        if not allowed:
            result.blocked_reason = "all_blocked"
            result.timings_ms = t
            return result

        by_id = {c.chunk_id: c for c in allowed}
        allowed_ids = set(by_id)
        bm25_rank: dict[str, int] = {}
        dense_rank: dict[str, int] = {}

        if cfg.mode in {"bm25", "hybrid"}:
            s = time.perf_counter()
            index = self._bm25_for(allowed)
            for rank, (i, _) in enumerate(index.search(tokenize(query), cfg.bm25_k), start=1):
                bm25_rank[allowed[i].chunk_id] = rank
            t["bm25_ms"] = (time.perf_counter() - s) * 1000

        if cfg.mode in {"dense", "hybrid"}:
            if self.embedder is None or dense_fn is None:
                raise ValueError("dense retrieval needs an embedder and a dense_fn")
            s = time.perf_counter()
            qvec = self.embedder.embed_queries([query])[0]
            t["embed_ms"] = (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            hits = dense_fn(qvec, cfg.dense_k, cfg.trust_threshold)
            rank = 0
            for chunk_id, _ in hits:
                if chunk_id in allowed_ids:
                    rank += 1
                    dense_rank[chunk_id] = rank
            t["dense_ms"] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        bm25_order = sorted(bm25_rank, key=bm25_rank.get)  # type: ignore[arg-type]
        dense_order = sorted(dense_rank, key=dense_rank.get)  # type: ignore[arg-type]
        if cfg.mode == "bm25":
            fused = [(cid, 1.0 / r) for r, cid in enumerate(bm25_order, start=1)]
        elif cfg.mode == "dense":
            fused = [(cid, 1.0 / r) for r, cid in enumerate(dense_order, start=1)]
        else:
            fused = rrf([bm25_order, dense_order], k=cfg.rrf_k, weights=[cfg.w_bm25, cfg.w_dense])
        t["fuse_ms"] = (time.perf_counter() - s) * 1000

        hits_out = [
            Hit(
                chunk=by_id[cid],
                score=score,
                fused=score,
                bm25_rank=bm25_rank.get(cid),
                dense_rank=dense_rank.get(cid),
            )
            for cid, score in fused
        ]

        if self.reranker is not None and hits_out:
            s = time.perf_counter()
            head = hits_out[: cfg.rerank_top_n]
            scores = self.reranker.score(query, [h.chunk.text for h in head])
            for h, sc in zip(head, scores, strict=True):
                h.rerank = sc
            head.sort(key=lambda h: -(h.rerank or 0.0))
            for h in head:
                h.score = h.rerank  # type: ignore[assignment]
            hits_out = head + hits_out[cfg.rerank_top_n :]
            t["rerank_ms"] = (time.perf_counter() - s) * 1000
            top = head[0].rerank
            thr = cfg.abstain_threshold
            if thr is not None and top is not None and top < thr:
                result.blocked_reason = "low_relevance"
                result.timings_ms = {**t, "total_ms": (time.perf_counter() - t0) * 1000}
                return result

        if not hits_out:
            result.blocked_reason = "no_relevant_content"
            result.timings_ms = {**t, "total_ms": (time.perf_counter() - t0) * 1000}
            return result

        if cfg.max_per_doc:
            counts: dict[str, int] = {}
            capped: list[Hit] = []
            for h in hits_out:
                if counts.get(h.chunk.doc_id, 0) < cfg.max_per_doc:
                    counts[h.chunk.doc_id] = counts.get(h.chunk.doc_id, 0) + 1
                    capped.append(h)
            hits_out = capped

        result.hits = hits_out[: cfg.final_k]
        result.answerable = True
        t["total_ms"] = (time.perf_counter() - t0) * 1000
        result.timings_ms = t
        return result


def with_overrides(cfg: RetrievalConfig, **kw: object) -> RetrievalConfig:
    return replace(cfg, **kw)  # type: ignore[arg-type]
