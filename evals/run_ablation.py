"""Retrieval ablation on a BEIR dataset.

    python -m evals.run_ablation --data data/scifact --embedder st:BAAI/bge-small-en-v1.5 \
        --reranker st:cross-encoder/ms-marco-MiniLM-L-6-v2 --out results

Protocol
  1. Queries are split into dev and test with a fixed seed.
  2. Fusion hyperparameters (RRF k and the BM25 weight) are tuned on dev only.
  3. Every system is scored on test only. Nothing is tuned on test.
  4. Differences against the v1 baseline and against BM25 get a paired bootstrap interval,
     and so do hybrid against dense and (when enabled) rerank against hybrid.
  5. Latency is measured per query, single process, after a short warm-up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartrag.ingest import chunk_words
from smartrag.ingest.sanitize import detect
from smartrag.retrieval import (
    ArrayDenseIndex,
    HashingEmbedder,
    OverlapReranker,
    PoolChunk,
    RetrievalConfig,
    Retriever,
    rrf,
)
from smartrag.retrieval.lexical_v1 import LexicalV1

from .beir_data import BeirDataset, load_beir, split_queries
from .metrics import evaluate_run, paired_bootstrap

DEPTH = 300  # chunk candidates kept per query; enough for 100 distinct documents
KS_GRID = [1, 3, 10, 30, 60, 100]
W_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def make_embedder(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(int(arg or 256)), True
    if kind == "st":
        from smartrag.retrieval import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(arg or "BAAI/bge-small-en-v1.5"), False
    raise SystemExit(f"unknown embedder spec: {spec}")


def make_reranker(spec: str):
    if spec == "none":
        return None, False
    kind, _, arg = spec.partition(":")
    if kind == "overlap":
        return OverlapReranker(), True
    if kind == "st":
        from smartrag.retrieval import CrossEncoderReranker

        return CrossEncoderReranker(arg or "cross-encoder/ms-marco-MiniLM-L-6-v2"), False
    raise SystemExit(f"unknown reranker spec: {spec}")


def build_chunks(ds: BeirDataset, size: int, overlap: int) -> list[PoolChunk]:
    chunks: list[PoolChunk] = []
    for doc_id in sorted(ds.corpus):
        for c in chunk_words(ds.doc_text(doc_id), size, overlap):
            chunks.append(PoolChunk(f"{doc_id}#{c.idx}", doc_id, doc_id, c.idx, c.text, 100.0))
    return chunks


def embed_cached(embedder, chunks: list[PoolChunk], cache_dir: Path) -> np.ndarray:
    key = hashlib.sha1(
        (embedder.name + "\x00" + "\x00".join(c.text for c in chunks)).encode()
    ).hexdigest()[:16]
    path = cache_dir / f"emb-{embedder.name.replace('/', '_')}-{key}.npy"
    if path.exists():
        return np.load(path)
    matrix = embedder.embed_documents([c.text for c in chunks])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def docs_from_chunk_ranking(chunk_ids: list[str], depth: int = 100) -> list[str]:
    seen: dict[str, None] = {}
    for cid in chunk_ids:
        seen.setdefault(cid.rsplit("#", 1)[0], None)
        if len(seen) >= depth:
            break
    return list(seen)


@dataclass
class SystemRun:
    name: str
    ranking: dict[str, list[str]]
    latency_ms: dict[str, list[float]]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def run_system(name, rank_fn, queries: dict[str, str], qids: list[str], warmup: int = 3):
    for q in qids[:warmup]:
        rank_fn(queries[q])
    ranking, totals = {}, []
    stage: dict[str, list[float]] = {}
    for q in qids:
        t0 = time.perf_counter()
        docs, timings = rank_fn(queries[q])
        totals.append((time.perf_counter() - t0) * 1000)
        ranking[q] = docs
        for k, v in timings.items():
            stage.setdefault(k, []).append(v)
    stage["wall_ms"] = totals
    return SystemRun(name, ranking, stage)


def tune_fusion(bm25_lists, dense_lists, dev_qids, qrels, ks_grid, w_grid):
    """Grid search on dev only. Returns the best (rrf_k, w_bm25) and the whole grid."""
    grid = []
    for k in ks_grid:
        for w in w_grid:
            run = {}
            for q in dev_qids:
                fused = rrf([bm25_lists[q], dense_lists[q]], k=k, weights=[w, 1.0])
                run[q] = docs_from_chunk_ranking([cid for cid, _ in fused])
            score = evaluate_run(run, qrels, dev_qids).mean()["ndcg@10"]
            grid.append({"rrf_k": k, "w_bm25": w, "dev_ndcg@10": round(score, 5)})
    best = max(grid, key=lambda g: (g["dev_ndcg@10"], -abs(g["w_bm25"] - 1.0)))
    return best, grid


def grid_edges(best: dict, ks_grid: list, w_grid: list) -> list[str]:
    """Names of tuned parameters that landed on the edge of their search grid.

    An optimum on the edge means the true optimum may lie outside the grid."""
    edges = []
    if best["rrf_k"] in (min(ks_grid), max(ks_grid)):
        edges.append("rrf_k")
    if best["w_bm25"] in (min(w_grid), max(w_grid)):
        edges.append("w_bm25")
    return edges


def pairwise(
    names: list[str], scored: list, pairs: list[tuple[int, int]], n_resamples: int, seed: int
) -> list[dict]:
    """Paired bootstrap of nDCG@10 for chosen system pairs: (a, b) tests mean(a) - mean(b)."""
    out = []
    for a, b in pairs:
        if a < len(scored) and b < len(scored):
            res = paired_bootstrap(
                scored[a].per_query["ndcg@10"], scored[b].per_query["ndcg@10"], n_resamples, seed
            )
            out.append({"a": names[a], "b": names[b], **res})
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": git_commit(),
    }
    for mod in ("scipy", "sklearn", "sentence_transformers", "torch"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def fmt_row(name, mean, boot_v1, boot_bm25, lat):
    def delta(b):
        if b is None:
            return "n/a"
        return f"{b['diff']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]"

    return (
        f"| {name} | {mean['ndcg@10']:.3f} | {mean['recall@10']:.3f} | {mean['recall@100']:.3f}"
        f" | {mean['mrr@10']:.3f} | {delta(boot_v1)} | {delta(boot_bm25)}"
        f" | {percentile(lat, 50):.1f} | {percentile(lat, 95):.1f} |"
    )


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="BEIR dataset directory")
    ap.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    ap.add_argument("--reranker", default="st:cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--chunk-words", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--rerank-top-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default=".cache/embeddings")
    args = ap.parse_args(argv)

    embedder, emb_is_double = make_embedder(args.embedder)
    reranker, rr_is_double = make_reranker(args.reranker)
    smoke = emb_is_double or rr_is_double

    ds = load_beir(args.data)
    qids_all = sorted(ds.queries)
    if args.limit_queries:
        qids_all = qids_all[: args.limit_queries]
    dev, test = split_queries(qids_all, args.seed, args.dev_fraction)
    print(
        f"{ds.name}: {len(ds.corpus)} docs, {len(qids_all)} queries "
        f"({len(dev)} dev, {len(test)} test)"
    )

    chunks = build_chunks(ds, args.chunk_words, args.overlap)
    t0 = time.perf_counter()
    matrix = embed_cached(embedder, chunks, Path(args.cache))
    index_seconds = time.perf_counter() - t0
    dense = ArrayDenseIndex([c.chunk_id for c in chunks], matrix)
    texts = [c.text for c in chunks]
    queries = ds.queries

    def retriever_rank(retriever, cfg):
        def fn(query):
            res = retriever.retrieve(query, chunks, dense, cfg)
            return docs_from_chunk_ranking([h.chunk.chunk_id for h in res.hits]), res.timings_ms

        return fn

    base_cfg = RetrievalConfig(
        bm25_k=DEPTH,
        dense_k=DEPTH,
        final_k=DEPTH,
        trust_threshold=0.0,
        rerank_top_n=args.rerank_top_n,
    )
    plain = Retriever(embedder, None)

    # Stage lists for tuning (dev only).
    bm25_lists, dense_lists = {}, {}
    for q in dev:
        bm = plain.retrieve(
            queries[q],
            chunks,
            dense,
            RetrievalConfig(mode="bm25", bm25_k=DEPTH, final_k=DEPTH, trust_threshold=0.0),
        )
        de = plain.retrieve(
            queries[q],
            chunks,
            dense,
            RetrievalConfig(mode="dense", dense_k=DEPTH, final_k=DEPTH, trust_threshold=0.0),
        )
        bm25_lists[q] = [h.chunk.chunk_id for h in bm.hits]
        dense_lists[q] = [h.chunk.chunk_id for h in de.hits]
    best, grid = tune_fusion(
        bm25_lists,
        dense_lists,
        dev,
        ds.qrels,
        ks_grid=[10, 30, 60, 100],
        w_grid=[0.5, 0.75, 1.0, 1.5, 2.0],
    )
    edges = grid_edges(best, KS_GRID, W_GRID)
    print(
        f"tuned on dev: rrf_k={best['rrf_k']} w_bm25={best['w_bm25']} "
        f"(dev nDCG@10={best['dev_ndcg@10']})"
    )
    if edges:
        print(f"WARNING: {', '.join(edges)} sit on the edge of the search grid")
    hybrid_cfg = RetrievalConfig(
        **{**base_cfg.__dict__, "mode": "hybrid", "rrf_k": best["rrf_k"], "w_bm25": best["w_bm25"]}
    )

    v1 = LexicalV1(texts)

    def v1_rank(query):
        t = time.perf_counter()
        scores = v1.scores(query)
        order = np.argsort(-scores, kind="stable")[:DEPTH]
        docs = docs_from_chunk_ranking([chunks[i].chunk_id for i in order])
        return docs, {"score_ms": (time.perf_counter() - t) * 1000}

    systems = [
        ("v1 lexical (hashed TF + BM25Okapi, 70/30)", v1_rank),
        (
            "BM25 (v2 tokeniser)",
            retriever_rank(plain, RetrievalConfig(**{**base_cfg.__dict__, "mode": "bm25"})),
        ),
        (
            f"Dense ({embedder.name})",
            retriever_rank(plain, RetrievalConfig(**{**base_cfg.__dict__, "mode": "dense"})),
        ),
        ("BM25 + dense (RRF)", retriever_rank(plain, hybrid_cfg)),
    ]
    if reranker is not None:
        systems.append(
            (
                f"BM25 + dense (RRF) + rerank top {args.rerank_top_n}",
                retriever_rank(Retriever(embedder, reranker), hybrid_cfg),
            )
        )

    runs = [run_system(name, fn, queries, test) for name, fn in systems]
    scored = [evaluate_run(r.ranking, ds.qrels, test) for r in runs]
    means = [s.mean() for s in scored]

    rows, results = [], []
    for i, (r, s, m) in enumerate(zip(runs, scored, means, strict=True)):
        b_v1 = (
            None
            if i == 0
            else paired_bootstrap(
                s.per_query["ndcg@10"], scored[0].per_query["ndcg@10"], args.bootstrap, args.seed
            )
        )
        b_bm = (
            None
            if i <= 1
            else paired_bootstrap(
                s.per_query["ndcg@10"], scored[1].per_query["ndcg@10"], args.bootstrap, args.seed
            )
        )
        rows.append(fmt_row(r.name, m, b_v1, b_bm, r.latency_ms["wall_ms"]))
        results.append(
            {
                "system": r.name,
                "metrics": m,
                "n_queries": len(s.qids),
                "ndcg@10_vs_v1": b_v1,
                "ndcg@10_vs_bm25": b_bm,
                "latency_ms": {
                    k: {"p50": percentile(v, 50), "p95": percentile(v, 95)}
                    for k, v in r.latency_ms.items()
                },
            }
        )

    names = [r.name for r in runs]
    # system order: 0 v1, 1 BM25, 2 dense, 3 hybrid, 4 hybrid + rerank (when enabled)
    pw = pairwise(names, scored, [(3, 2), (4, 3)], args.bootstrap, args.seed)

    # Corpus-level false positive rate of the injection sanitiser on real documents.
    flagged = [d for d in ds.corpus if detect(ds.doc_text(d))]
    fp = {
        "documents": len(ds.corpus),
        "flagged": len(flagged),
        "rate": len(flagged) / max(len(ds.corpus), 1),
        "examples": flagged[:10],
    }

    header = (
        "| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 vs v1 (95% CI) "
        "| nDCG@10 vs BM25 (95% CI) | p50 ms | p95 ms |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    banner = (
        (
            "> SMOKE TEST: a test-double embedder or reranker was used. These numbers verify "
            "the pipeline only and say nothing about retrieval quality.\n\n"
        )
        if smoke
        else ""
    )
    edge_note = (
        f"> Warning: {', '.join(edges)} landed on the edge of the search grid, so the best value "
        "may lie outside it.\n\n"
        if edges
        else ""
    )
    pair_lines = [
        "Pairwise comparisons, nDCG@10 (paired bootstrap, 95% interval, not corrected for "
        "multiple comparisons):\n",
        "| A | B | A minus B (95% CI) | p |",
        "|---|---|---|---|",
    ] + [
        f"| {c['a']} | {c['b']} | {c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
        f"| {c['p_value']:.3f} |"
        for c in pw
    ]
    md = (
        f"# {ds.name} ablation{' (SMOKE TEST)' if smoke else ''}\n\n{banner}"
        f"Test queries: {len(test)}. Tuned on {len(dev)} dev queries: rrf_k={best['rrf_k']}, "
        f"w_bm25={best['w_bm25']}. Seed {args.seed}.\n\n{edge_note}{header}\n"
        + "\n".join(rows)
        + "\n\n"
        + "\n".join(pair_lines)
        + "\n\n"
        f"Embedding {len(chunks)} chunks took {index_seconds:.1f}s (cached on later runs).\n"
        f"Sanitiser flagged {fp['flagged']} of {fp['documents']} corpus documents "
        f"({fp['rate']:.2%}).\n"
    )

    out_dir = Path(args.out) / (ds.name + ("-smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": ds.name,
        "smoke_test": smoke,
        "args": vars(args),
        "environment": environment(),
        "n_docs": len(ds.corpus),
        "n_chunks": len(chunks),
        "dev_qids": dev,
        "test_qids": test,
        "tuned": best,
        "tuned_at_grid_edge": edges,
        "search_grid": {"rrf_k": KS_GRID, "w_bm25": W_GRID},
        "tuning_grid_dev": grid,
        "pairwise_ndcg@10": pw,
        "systems": results,
        "sanitiser_false_positives": fp,
        "embedding_seconds": index_seconds,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "results.md").write_text(md)
    print(md)
    return payload


if __name__ == "__main__":
    main()
