"""End-to-end evaluation on your own documents and a hand-written question set.

    python -m evals.rag_eval --docs data/docs --goldset data/goldset.jsonl \
        --embedder st:BAAI/bge-small-en-v1.5 --reranker st:cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --llm groq --calibrate

Goldset format, one JSON object per line:
  {"id": "q01", "question": "...", "answerable": true,
   "evidence": ["exact phrase that must appear in a retrieved chunk", "..."],
   "reference_answer": "optional"}
  {"id": "q51", "question": "Out of scope question", "answerable": false}
  {"id": "q60", "question": "And what about its cost?", "answerable": true, "evidence": ["..."],
   "history": [{"question": "What is the Pro plan?", "answer": "The Pro plan is ..."}]}

Rows with "history" are follow-up questions. They are searched after query rewriting, and the
report compares retrieval hit rate with and without the rewrite.

Retrieval hit: every evidence phrase appears somewhere in the retrieved chunks. Write evidence
phrases yourself, from the documents, so the retriever is graded against ground truth and not
against its own output.

Metrics: retrieval hit rate and MRR, abstention quality, citation validity, and faithfulness
judged by a language model. A model judging its own output is lenient, so use --judge-model to
pick a different model, and check the exported hand-check sample by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import httpx

from smartrag.config import Settings
from smartrag.dotenv import load_dotenv
from smartrag.ingest import extract_pdf_path
from smartrag.llm import LLM, GeminiLLM, GroqLLM
from smartrag.service import SmartRAGService
from smartrag.storage import MemoryStore

from .metrics import wilson_interval
from .run_ablation import make_embedder, make_reranker

JUDGE_SYSTEM = (
    "You verify whether an answer is fully supported by the numbered sources. "
    'Reply with JSON only: {"supported": true or false, "unsupported_claims": ["..."]}. '
    "An answer is supported only if every factual claim in it follows from the sources."
)


def norm(s: str) -> str:
    normalised = unicodedata.normalize("NFKC", s).translate(
        str.maketrans({"‘": "'", "’": "'", "‛": "'", "“": '"', "”": '"'})
    )
    return re.sub(r"\s+", " ", normalised.lower()).strip()


def load_goldset(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "question", "answerable"):
                if key not in row:
                    raise ValueError(f"goldset line {n}: missing '{key}'")
            hist = row.get("history")
            if hist is not None and (
                not isinstance(hist, list)
                or not all(isinstance(t, dict) and "question" in t for t in hist)
            ):
                raise ValueError(f"goldset line {n}: history must be a list of turns")
            if row["answerable"] and not row.get("evidence"):
                raise ValueError(f"goldset line {n}: answerable questions need 'evidence'")
            rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("goldset ids must be unique")
    return rows


def load_docs(path: str | Path) -> dict[str, str]:
    docs = {}
    for p in sorted(Path(path).iterdir()):
        if p.suffix.lower() == ".pdf":
            docs[p.name] = extract_pdf_path(str(p))
        elif p.suffix.lower() in {".txt", ".md"}:
            docs[p.name] = p.read_text(encoding="utf-8")
    return docs


def evidence_hit(chunk_texts: list[str], evidence: list[str]) -> tuple[bool, int | None]:
    """Return whether all evidence was found and its first complete-evidence rank."""
    normalised = [norm(t) for t in chunk_texts]
    ev = [norm(e) for e in evidence]
    found: set[str] = set()
    for rank, text in enumerate(normalised, start=1):
        found.update(e for e in ev if e in text)
        if len(found) == len(set(ev)):
            return True, rank
    return False, None


def parse_json_object(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge(llm: LLM, question: str, answer: str, passages: list[str]) -> dict:
    sources = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    raw = llm.complete(
        JUDGE_SYSTEM, f"Sources:\n{sources}\n\nQuestion: {question}\n\nAnswer: {answer}", 300
    )
    parsed = parse_json_object(raw)
    if not parsed or "supported" not in parsed:
        return {"supported": None, "unsupported_claims": [], "raw": raw}
    return {
        "supported": bool(parsed["supported"]),
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "raw": raw,
    }


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci_low": lo, "ci_high": hi}


def choose_threshold(answerable_scores: list[float], unanswerable_scores: list[float]) -> float:
    """Threshold on the top rerank score that maximises balanced accuracy.

    Scores >= threshold are answered; scores below are refused."""
    candidates = sorted(set(answerable_scores + unanswerable_scores))
    best, best_ba = candidates[0] if candidates else 0.0, -1.0
    for t in candidates:
        tpr = sum(s >= t for s in answerable_scores) / max(len(answerable_scores), 1)
        tnr = sum(s < t for s in unanswerable_scores) / max(len(unanswerable_scores), 1)
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best, best_ba = t, ba
    return best


def balanced_accuracy(thr: float, ans: list[float], unans: list[float]) -> dict:
    tpr = sum(s >= thr for s in ans) / max(len(ans), 1)
    tnr = sum(s < thr for s in unans) / max(len(unans), 1)
    return {
        "threshold": thr,
        "answerable_kept": tpr,
        "unanswerable_refused": tnr,
        "balanced_accuracy": (tpr + tnr) / 2,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
    }


def build_service(
    docs: dict[str, str],
    embedder,
    reranker,
    llm,
    settings: Settings,
    abstain_threshold: float | None,
):
    svc = SmartRAGService(
        MemoryStore(embedder.dim),
        embedder,
        llm,
        reranker,
        settings,
        abstain_threshold=abstain_threshold,
    )
    user, _ = svc.signup("eval@local.test", "evaluation-password")
    for name, text in docs.items():
        svc.ingest_text(user.id, name, text)
    return svc, user.id


def evaluate(
    svc: SmartRAGService, user_id: str, gold: list[dict], judge_llm: LLM | None, seed: int = 0
) -> dict:
    hits, mrr, ans_total = 0, 0.0, 0
    fu_n = fu_raw = fu_rewritten = fu_failed = 0
    per_question, abstain_tp, abstain_fp, abstain_fn, abstain_tn = [], 0, 0, 0, 0
    cited_ok, answered, faithful, judged = 0, 0, 0, 0
    active_judge = judge_llm
    judge_rate_limited = False
    judgments = []
    for g in gold:
        history = g.get("history")
        search_q, rewritten_q, _ = svc.rewrite_question(g["question"], history)
        retrieval = svc.retrieve(user_id, search_q)
        texts = [h.chunk.text for h in retrieval.hits]
        if history and g["answerable"] and svc.rewriter is not None:
            raw_texts = [h.chunk.text for h in svc.retrieve(user_id, g["question"]).hits]
            fu_n += 1
            fu_raw += evidence_hit(raw_texts, g["evidence"])[0]
            fu_rewritten += evidence_hit(texts, g["evidence"])[0]
            fu_failed += rewritten_q is None and search_q == g["question"]
        rec = {
            "id": g["id"],
            "answerable": g["answerable"],
            "retrieval_answerable": retrieval.answerable,
            "retrieved_top": [
                {
                    "rank": rank,
                    "document": hit.chunk.doc_name,
                    "chunk_index": hit.chunk.idx,
                    "score": round(float(hit.score), 6),
                    "trust": hit.chunk.trust,
                    "excerpt": hit.chunk.text[:400],
                }
                for rank, hit in enumerate(retrieval.hits[:5], start=1)
            ],
        }
        if g["answerable"]:
            ans_total += 1
            ok, first = evidence_hit(texts, g["evidence"])
            hits += ok
            mrr += 1.0 / first if ok and first else 0.0
            rec.update(evidence_hit=ok, first_evidence_rank=first)
        if svc.llm is not None:
            out = svc.query(user_id, g["question"], record_history=False, history=history)
            refused = not out.answerable
            rec.update(
                answer=out.answer,
                refused=refused,
                citations=out.citations,
                invalid_citations=out.invalid_citations,
                llm_called=out.llm_called,
            )
            if g["answerable"]:
                abstain_fn += refused
                abstain_tn += not refused
            else:
                abstain_tp += refused
                abstain_fp += not refused
            if not refused:
                answered += 1
                cited_ok += bool(out.citations) and not out.invalid_citations
                if active_judge is not None and g["answerable"]:
                    try:
                        verdict = judge(
                            active_judge,
                            g["question"],
                            out.answer,
                            [s["excerpt"] for s in out.sources],
                        )
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code != 429:
                            raise
                        active_judge = None
                        judge_rate_limited = True
                        rec["judge"] = {"supported": None, "error": "rate_limited"}
                        print(
                            "Warning: judge request was rate-limited (HTTP 429); "
                            "skipping remaining faithfulness judgments.",
                            file=sys.stderr,
                        )
                    else:
                        rec["judge"] = verdict
                        judgments.append(
                            {
                                "id": g["id"],
                                "question": g["question"],
                                "answer": out.answer,
                                "sources": out.sources,
                                **verdict,
                            }
                        )
                        if verdict["supported"] is not None:
                            judged += 1
                            faithful += verdict["supported"]
        per_question.append(rec)

    summary: dict = {
        "n_questions": len(gold),
        "n_answerable": ans_total,
        "retrieval_hit_rate": rate(hits, ans_total),
        "retrieval_mrr": mrr / ans_total if ans_total else 0.0,
    }
    if fu_n:
        summary["rewrite_comparison"] = {
            "follow_up_questions": fu_n,
            "hit_without_rewrite": rate(fu_raw, fu_n),
            "hit_with_rewrite": rate(fu_rewritten, fu_n),
            "unchanged_or_failed": fu_failed,
        }
    if svc.llm is not None:
        prec = abstain_tp / max(abstain_tp + abstain_fp, 1)
        rec_ = abstain_tp / max(abstain_tp + abstain_fn, 1)
        summary["abstention"] = {
            "refused_unanswerable": rate(abstain_tp, abstain_tp + abstain_fp),
            "refused_answerable_wrongly": rate(abstain_fn, abstain_fn + abstain_tn),
            "precision": prec,
            "recall": rec_,
            "f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0,
        }
        summary["citation_validity"] = rate(cited_ok, answered)
        if judge_llm is not None:
            summary["faithfulness_judged"] = rate(faithful, judged)
            summary["faithfulness_judge_status"] = (
                "rate_limited" if judge_rate_limited else "complete"
            )
    return {
        "summary": summary,
        "per_question": per_question,
        "judgments": judgments,
        "handcheck_sample": random.Random(seed).sample(judgments, min(10, len(judgments))),
    }


def calibrate(svc: SmartRAGService, user_id: str, gold: list[dict], seed: int = 0) -> dict:
    """Pick the abstention threshold on a dev half and report balanced accuracy on a test half."""
    if svc.reranker is None:
        raise ValueError("calibration needs a reranker")
    scored = []
    for g in gold:
        r = svc.retrieve(
            user_id,
            g["question"],
            cfg=svc.retrieval_config().__class__(
                **{**svc.retrieval_config().__dict__, "abstain_threshold": None}
            ),
        )
        top = max((h.rerank for h in r.hits if h.rerank is not None), default=float("-inf"))
        scored.append((g["id"], g["answerable"], top))
    ids = sorted(i for i, _, _ in scored)
    random.Random(seed).shuffle(ids)
    dev_ids = set(ids[: len(ids) // 2])
    dev = [(a, s) for i, a, s in scored if i in dev_ids]
    test = [(a, s) for i, a, s in scored if i not in dev_ids]

    def split(rows):
        return [s for a, s in rows if a], [s for a, s in rows if not a]

    dev_a, dev_u = split(dev)
    thr = choose_threshold(dev_a, dev_u)
    test_a, test_u = split(test)
    return {
        "chosen_on": "dev",
        "threshold": thr,
        "dev": balanced_accuracy(thr, dev_a, dev_u),
        "test": balanced_accuracy(thr, test_a, test_u),
    }


def ragas_rows(gold: list[dict], result: dict, svc: SmartRAGService, user_id: str) -> list[dict]:
    """Rows in the layout RAGAS expects, so you can run RAGAS on the same answers."""
    by_id = {g["id"]: g for g in gold}
    rows = []
    for rec in result["per_question"]:
        if "answer" not in rec or rec.get("refused"):
            continue
        g = by_id[rec["id"]]
        contexts = [h.chunk.text for h in svc.retrieve(user_id, g["question"]).hits]
        rows.append(
            {
                "user_input": g["question"],
                "retrieved_contexts": contexts,
                "response": rec["answer"],
                "reference": g.get("reference_answer", ""),
            }
        )
    return rows


def to_markdown(result: dict, cal: dict | None) -> str:
    s = result["summary"]

    def cell(x):
        return (
            f"{x['k']}/{x['n']} = {x['rate']:.0%} (95% CI {x['ci_low']:.0%} to {x['ci_high']:.0%})"
        )

    lines = [
        "# RAG evaluation",
        "",
        f"Questions: {s['n_questions']} ({s['n_answerable']} answerable)",
        "",
        f"- Retrieval hit rate: {cell(s['retrieval_hit_rate'])}",
        f"- Retrieval MRR: {s['retrieval_mrr']:.3f}",
    ]
    if "rewrite_comparison" in s:
        r = s["rewrite_comparison"]
        lines.append(
            f"- Follow-up questions ({r['follow_up_questions']}): retrieval hit without rewrite "
            f"{cell(r['hit_without_rewrite'])}, with rewrite {cell(r['hit_with_rewrite'])}"
        )
    if "abstention" in s:
        a = s["abstention"]
        lines += [
            f"- Out-of-scope questions refused: {cell(a['refused_unanswerable'])}",
            f"- In-scope questions wrongly refused: {cell(a['refused_answerable_wrongly'])}",
            f"- Abstention precision {a['precision']:.2f}, recall {a['recall']:.2f}, "
            f"F1 {a['f1']:.2f}",
            f"- Answers with valid citations: {cell(s['citation_validity'])}",
        ]
    if "faithfulness_judged" in s:
        lines.append(f"- Faithfulness (model judge): {cell(s['faithfulness_judged'])}")
    if s.get("faithfulness_judge_status") == "rate_limited":
        lines.append(
            "- Faithfulness judge was rate-limited; remaining judgments were skipped. "
            "Other evaluation metrics completed."
        )
    if cal:
        t = cal["test"]
        lines += [
            "",
            "## Abstention threshold calibration",
            f"Chosen on dev: {cal['threshold']:.3f}. On held-out test: balanced accuracy "
            f"{t['balanced_accuracy']:.2f} (answerable kept {t['answerable_kept']:.2f}, "
            f"unanswerable refused {t['unanswerable_refused']:.2f}, "
            f"n={t['n_answerable']}+{t['n_unanswerable']}).",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--goldset", required=True)
    ap.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    ap.add_argument("--reranker", default="st:cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--llm", choices=["none", "groq", "gemini"], default="none")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="override the model request timeout in seconds (Gemini defaults to 180)",
    )
    ap.add_argument("--judge-model", default=None, help="defaults to the answering model")
    ap.add_argument(
        "--skip-judge",
        action="store_true",
        help="skip faithfulness-judge requests while keeping answer generation",
    )
    ap.add_argument("--abstain-threshold", type=float, default=None)
    ap.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="override the number of fused candidates reranked (default: 30)",
    )
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rag")
    args = ap.parse_args(argv)

    docs_path = Path(args.docs)
    if not docs_path.is_dir():
        ap.error(f"--docs must be an existing directory: {docs_path}")
    goldset_path = Path(args.goldset)
    if not goldset_path.is_file():
        ap.error(f"--goldset must be an existing file: {goldset_path}")
    docs, gold = load_docs(docs_path), load_goldset(goldset_path)
    if not docs:
        ap.error(f"No .pdf, .txt, or .md documents found in {docs_path}")
    if not gold:
        ap.error(f"No questions found in goldset: {goldset_path}")
    if args.llm_timeout is not None and args.llm_timeout <= 0:
        ap.error("--llm-timeout must be greater than zero")
    if args.rerank_top_n is not None and args.rerank_top_n <= 0:
        ap.error("--rerank-top-n must be greater than zero")

    load_dotenv()
    llm_model = args.llm_model or (
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if args.llm == "gemini"
        else Settings.from_env().llm_model
    )
    embedder, _ = make_embedder(args.embedder)
    reranker, _ = make_reranker(args.reranker)
    llm_factory = {"groq": GroqLLM, "gemini": GeminiLLM}.get(args.llm)
    llm_options = {"model": llm_model}
    if args.llm_timeout is not None:
        llm_options["timeout"] = args.llm_timeout
    llm = llm_factory(**llm_options) if llm_factory else None
    if args.skip_judge:
        judge_llm = None
    elif args.judge_model:
        judge_options = {**llm_options, "model": args.judge_model}
        judge_llm = llm_factory(**judge_options) if llm_factory else None
    else:
        judge_llm = llm
    settings = Settings(rerank_top_n=args.rerank_top_n or Settings().rerank_top_n)
    svc, uid = build_service(docs, embedder, reranker, llm, settings, args.abstain_threshold)

    result = evaluate(svc, uid, gold, judge_llm, args.seed)
    cal = calibrate(svc, uid, gold, args.seed) if args.calibrate and reranker else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.json").write_text(json.dumps({**result, "calibration": cal}, indent=2))
    md = to_markdown(result, cal)
    (out / "rag.md").write_text(md)
    if result["judgments"]:
        (out / "judgments.jsonl").write_text("\n".join(json.dumps(j) for j in result["judgments"]))
        with open(out / "handcheck.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "answer", "judge_supported", "human_supported"])
            for j in result["handcheck_sample"]:
                w.writerow([j["id"], j["question"], j["answer"], j["supported"], ""])
    if llm is not None:
        (out / "ragas.jsonl").write_text(
            "\n".join(json.dumps(r) for r in ragas_rows(gold, result, svc, uid))
        )
    print(md)
    return result


if __name__ == "__main__":
    main()
