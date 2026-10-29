from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]]  # id -> {"title": ..., "text": ...}
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def doc_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        return f"{d.get('title', '')} {d.get('text', '')}".strip()


def load_beir(path: str | Path, split: str = "test") -> BeirDataset:
    """Load a dataset in the standard BEIR layout:
    corpus.jsonl, queries.jsonl, qrels/<split>.tsv (header: query-id, corpus-id, score)."""
    root = Path(path)
    corpus: dict[str, dict[str, str]] = {}
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}
    queries: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(score)
    queries = {q: t for q, t in queries.items() if q in qrels}
    return BeirDataset(name=root.name, corpus=corpus, queries=queries, qrels=qrels)


def split_queries(
    qids: list[str], seed: int = 0, dev_fraction: float = 0.5
) -> tuple[list[str], list[str]]:
    """Deterministic dev/test split. Tune on dev only; report on test only."""
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    cut = int(round(len(ordered) * dev_fraction))
    return sorted(ordered[:cut]), sorted(ordered[cut:])
