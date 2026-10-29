"""A tiny synthetic BEIR-format dataset for plumbing tests.

The numbers it produces mean nothing. It exists so the evaluation code can be exercised
offline and in CI. Real results come from real BEIR datasets (see scripts/download_beir.py).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def make_synthetic_beir(
    out_dir: str | Path, n_topics: int = 8, docs_per_topic: int = 10, seed: int = 0
) -> Path:
    rng = random.Random(seed)
    root = Path(out_dir)
    (root / "qrels").mkdir(parents=True, exist_ok=True)
    shared = [f"common{i}" for i in range(20)]
    docs, queries, qrels = [], [], []
    for t in range(n_topics):
        pool = [f"topic{t}word{i}" for i in range(14)]
        for d in range(docs_per_topic):
            words = rng.sample(pool, 9) + rng.sample(shared, 4)
            rng.shuffle(words)
            doc_id = f"d{t}_{d}"
            docs.append({"_id": doc_id, "title": f"Topic {t}", "text": " ".join(words)})
            if d < 3:
                qid = f"q{t}_{d}"
                queries.append({"_id": qid, "text": " ".join(rng.sample(words[:9], 3))})
                qrels.append((qid, doc_id, 1))
    with open(root / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(root / "queries.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    with open(root / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.writelines(f"{q}\t{d}\t{s}\n" for q, d, s in qrels)
    return root
