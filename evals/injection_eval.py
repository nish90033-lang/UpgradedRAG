"""Measure what the injection sanitiser catches and what it wrongly flags.

    python -m evals.injection_eval
    python -m evals.injection_eval --extra my_cases.jsonl --corpus data/scifact

Case files are JSON lines. Malicious cases need id, category, text, author. Benign cases need
id, text, author. The default cases were written by the assistant that built the sanitiser,
so they are not an independent test. Add cases written by other people through --extra and
set their "author" field, and the report breaks results down by author.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from smartrag.ingest.sanitize import detect, trust_score

from .metrics import wilson_interval

DATA = Path(__file__).parent / "data"
FILLER = (
    "The committee reviewed the annual budget and discussed staffing, facilities, and "
    "the schedule for the coming term. "
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_in_document(text: str, n_words: int, seed: int) -> str:
    """Place an attack at a random position inside a long benign document."""
    rng = random.Random(seed)
    words = (FILLER * (n_words // 20 + 1)).split()[:n_words]
    pos = rng.randrange(len(words) + 1)
    return " ".join(words[:pos]) + "\n" + text + "\n" + " ".join(words[pos:])


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci_low": lo, "ci_high": hi}


def run(malicious: list[dict], benign: list[dict], doc_words: int = 20000) -> dict:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_author: dict[str, list[bool]] = defaultdict(list)
    missed, position_mismatch, capped = [], 0, 0
    for i, case in enumerate(malicious):
        hit = bool(detect(case["text"]))
        by_cat[case["category"]].append(hit)
        by_author[case.get("author", "unknown")].append(hit)
        if not hit:
            missed.append(
                {"id": case["id"], "category": case["category"], "text": case["text"][:90]}
            )
        # The hard cap makes detection independent of document length and position.
        embedded = detect(embed_in_document(case["text"], doc_words, seed=i))
        if bool(embedded) != hit:
            position_mismatch += 1
        if hit and trust_score(embedded) <= 40.0:
            capped += 1

    flagged_benign = [
        {"id": b["id"], "text": b["text"][:90], "patterns": detect(b["text"])}
        for b in benign
        if detect(b["text"])
    ]
    all_hits = [h for hits in by_cat.values() for h in hits]
    return {
        "detection_overall": rate(sum(all_hits), len(all_hits)),
        "detection_by_category": {c: rate(sum(h), len(h)) for c, h in sorted(by_cat.items())},
        "detection_by_author": {a: rate(sum(h), len(h)) for a, h in sorted(by_author.items())},
        "false_positive_rate": rate(len(flagged_benign), len(benign)),
        "missed": missed,
        "flagged_benign": flagged_benign,
        "position_and_length_mismatches": position_mismatch,
        "detected_cases_capped_at_40_inside_long_document": capped,
        "document_words": doc_words,
    }


def to_markdown(r: dict) -> str:
    def cell(x):
        return (
            f"{x['k']}/{x['n']} = {x['rate']:.0%} (95% CI {x['ci_low']:.0%} to {x['ci_high']:.0%})"
        )

    lines = [
        "# Injection sanitiser evaluation",
        "",
        f"Overall detection: {cell(r['detection_overall'])}",
        f"False positives on benign text: {cell(r['false_positive_rate'])}",
        "",
        "| Category | Detected |",
        "|---|---|",
    ]
    lines += [f"| {c} | {cell(v)} |" for c, v in r["detection_by_category"].items()]
    lines += ["", "| Author | Detected |", "|---|---|"]
    lines += [f"| {a} | {cell(v)} |" for a, v in r["detection_by_author"].items()]
    lines += [
        "",
        f"Detection changed when an attack was embedded in a {r['document_words']}-word "
        f"document: {r['position_and_length_mismatches']} cases. "
        f"Trust capped at 40 for {r['detected_cases_capped_at_40_inside_long_document']} "
        "detected cases inside the long document.",
        "",
        "## Missed attacks",
        "",
    ]
    lines += [f"- {m['id']} ({m['category']}): {m['text']}" for m in r["missed"]]
    lines += ["", "## Benign text that was flagged", ""]
    lines += [f"- {b['id']}: {b['text']} {b['patterns']}" for b in r["flagged_benign"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--malicious", default=str(DATA / "injection_cases.jsonl"))
    ap.add_argument("--benign", default=str(DATA / "benign_cases.jsonl"))
    ap.add_argument("--extra", help="additional malicious cases, for example written by others")
    ap.add_argument("--extra-benign", help="additional benign cases")
    ap.add_argument("--corpus", help="BEIR dataset directory for a false positive scan")
    ap.add_argument("--doc-words", type=int, default=20000)
    ap.add_argument("--out", default="results/injection")
    args = ap.parse_args(argv)

    malicious = load_jsonl(Path(args.malicious))
    benign = load_jsonl(Path(args.benign))
    if args.extra:
        malicious += load_jsonl(Path(args.extra))
    if args.extra_benign:
        benign += load_jsonl(Path(args.extra_benign))
    result = run(malicious, benign, args.doc_words)

    if args.corpus:
        from .beir_data import load_beir

        ds = load_beir(args.corpus)
        flagged = [d for d in ds.corpus if detect(ds.doc_text(d))]
        result["corpus_scan"] = {
            "dataset": ds.name,
            **rate(len(flagged), len(ds.corpus)),
            "examples": flagged[:10],
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "injection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = to_markdown(result)
    if "corpus_scan" in result:
        c = result["corpus_scan"]
        md += (
            f"\n## Corpus scan ({c['dataset']})\n\nFlagged {c['k']} of {c['n']} real documents "
            f"({c['rate']:.2%}).\n"
        )
    (out / "injection.md").write_text(md, encoding="utf-8")
    print(md)
    return result


if __name__ == "__main__":
    main()
