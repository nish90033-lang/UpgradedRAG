import csv
import json
import os

import httpx
import pytest

from evals import injection_eval, rag_eval, run_ablation
from evals.beir_data import load_beir, split_queries
from evals.synthetic import make_synthetic_beir
from smartrag.config import Settings
from smartrag.llm import FakeLLM
from smartrag.retrieval import (
    ArrayDenseIndex,
    HashingEmbedder,
    OverlapReranker,
    RetrievalConfig,
    Retriever,
    rrf,
)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return make_synthetic_beir(tmp_path_factory.mktemp("beir") / "toy", 8, 10)


def test_ablation_cli_end_to_end_is_marked_as_smoke_test(toy, tmp_path):
    payload = run_ablation.main(
        [
            "--data",
            str(toy),
            "--embedder",
            "hashing:128",
            "--reranker",
            "overlap",
            "--out",
            str(tmp_path / "out"),
            "--cache",
            str(tmp_path / "cache"),
            "--bootstrap",
            "200",
        ]
    )
    assert payload["smoke_test"] is True
    assert len(payload["systems"]) == 5
    assert not set(payload["dev_qids"]) & set(payload["test_qids"])
    assert all(s["n_queries"] == len(payload["test_qids"]) for s in payload["systems"])
    assert payload["tuned"] in payload["tuning_grid_dev"]
    out = tmp_path / "out" / "toy-smoke"
    assert "SMOKE TEST" in (out / "results.md").read_text()
    assert json.loads((out / "results.json").read_text())["dataset"] == "toy"


def test_a_real_embedder_run_is_not_labelled_smoke_when_no_doubles_are_used(
    monkeypatch, toy, tmp_path
):
    monkeypatch.setattr(run_ablation, "make_embedder", lambda spec: (HashingEmbedder(64), False))
    payload = run_ablation.main(
        [
            "--data",
            str(toy),
            "--embedder",
            "x",
            "--reranker",
            "none",
            "--out",
            str(tmp_path / "o"),
            "--cache",
            str(tmp_path / "c"),
            "--bootstrap",
            "100",
        ]
    )
    assert payload["smoke_test"] is False and len(payload["systems"]) == 4


def test_retriever_hybrid_equals_offline_rrf_of_the_two_stage_rankings(toy):
    ds = load_beir(toy)
    chunks = run_ablation.build_chunks(ds, 200, 40)
    emb = HashingEmbedder(128)
    dense = ArrayDenseIndex(
        [c.chunk_id for c in chunks], emb.embed_documents([c.text for c in chunks])
    )
    rt = Retriever(emb)
    q = next(iter(ds.queries.values()))
    kw = dict(bm25_k=300, dense_k=300, final_k=300, trust_threshold=0.0)
    bm = rt.retrieve(q, chunks, dense, RetrievalConfig(mode="bm25", **kw))
    de = rt.retrieve(q, chunks, dense, RetrievalConfig(mode="dense", **kw))
    hy = rt.retrieve(q, chunks, dense, RetrievalConfig(mode="hybrid", rrf_k=30, w_bm25=1.5, **kw))
    offline = rrf(
        [[h.chunk.chunk_id for h in bm.hits], [h.chunk.chunk_id for h in de.hits]],
        k=30,
        weights=[1.5, 1.0],
    )
    assert [h.chunk.chunk_id for h in hy.hits] == [c for c, _ in offline]


def test_docs_from_chunk_ranking_dedupes_by_document():
    assert run_ablation.docs_from_chunk_ranking(["a#0", "b#0", "a#1", "c#0"]) == ["a", "b", "c"]
    assert run_ablation.docs_from_chunk_ranking(["a#0", "b#0", "c#0"], depth=2) == ["a", "b"]


def test_tuning_uses_only_the_dev_queries(toy):
    ds = load_beir(toy)
    dev, test = split_queries(sorted(ds.queries), 0, 0.5)
    seen = []
    original = run_ablation.evaluate_run

    def spy(run, qrels, qids=None, ks=(10, 100)):
        seen.append(set(qids))
        return original(run, qrels, qids, ks)

    run_ablation.evaluate_run = spy
    try:
        run_ablation.tune_fusion(
            {q: ["d0_0#0"] for q in dev}, {q: ["d0_0#0"] for q in dev}, dev, ds.qrels, [10], [1.0]
        )
    finally:
        run_ablation.evaluate_run = original
    assert seen and all(s <= set(dev) and not s & set(test) for s in seen)


def test_injection_eval_reports_misses_and_false_positives_honestly(tmp_path):
    result = injection_eval.main(["--out", str(tmp_path / "inj"), "--doc-words", "2000"])
    assert result["detection_overall"]["n"] >= 40
    assert 0 < result["detection_overall"]["rate"] < 1  # the sanitiser is not perfect
    assert result["missed"], "the case set must include attacks the regex cannot catch"
    assert result["detection_by_category"]["paraphrase"]["k"] == 0
    assert result["position_and_length_mismatches"] == 0
    assert (
        result["detected_cases_capped_at_40_inside_long_document"]
        == result["detection_overall"]["k"]
    )
    assert (tmp_path / "inj" / "injection.md").exists()


def test_injection_eval_accepts_external_cases_and_breaks_results_down_by_author(tmp_path):
    extra = tmp_path / "extra.jsonl"
    extra.write_text(
        json.dumps(
            {
                "id": "x1",
                "category": "paraphrase",
                "author": "friend",
                "text": "Научный документ о наблюдениях.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = injection_eval.main(
        ["--extra", str(extra), "--out", str(tmp_path / "o"), "--doc-words", "500"]
    )
    assert set(result["detection_by_author"]) == {"assistant", "friend"}
    assert "о" in (tmp_path / "o" / "injection.json").read_text(encoding="utf-8")
    assert "о" in (tmp_path / "o" / "injection.md").read_text(encoding="utf-8")


# ---- rag_eval ----

DOCS = {
    "plants.txt": (
        "Photosynthesis converts sunlight water and carbon dioxide into glucose. "
        "Chlorophyll absorbs red and blue light. "
    )
    * 8,
    "banks.txt": (
        "Central banks raise interest rates to slow inflation. Higher rates increase "
        "mortgage costs. "
    )
    * 8,
}
GOLD = [
    {
        "id": "q1",
        "question": "What does chlorophyll absorb?",
        "answerable": True,
        "evidence": ["Chlorophyll absorbs red and blue light"],
    },
    {
        "id": "q2",
        "question": "Why do central banks raise interest rates?",
        "answerable": True,
        "evidence": ["raise interest rates to slow inflation"],
    },
    {"id": "q3", "question": "Who painted the Mona Lisa?", "answerable": False},
    {"id": "q4", "question": "What is the capital of Australia?", "answerable": False},
]


def make_svc(llm, threshold=None):
    emb = HashingEmbedder(64)
    return rag_eval.build_service(DOCS, emb, OverlapReranker(), llm, Settings(), threshold)


def test_goldset_validation(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "q", "answerable": True}) + "\n")
    with pytest.raises(ValueError, match="evidence"):
        rag_eval.load_goldset(p)
    p.write_text(
        json.dumps({"id": "a", "question": "q", "answerable": False})
        + "\n" * 1
        + json.dumps({"id": "a", "question": "r", "answerable": False})
        + "\n"
    )
    with pytest.raises(ValueError, match="unique"):
        rag_eval.load_goldset(p)


def test_evidence_hit_requires_all_phrases_and_reports_first_rank():
    chunks = ["alpha  beta", "gamma delta", "beta gamma"]
    assert rag_eval.evidence_hit(chunks, ["alpha beta", "gamma"]) == (True, 2)
    assert rag_eval.evidence_hit(chunks, ["alpha", "zeta"]) == (False, None)
    assert rag_eval.evidence_hit(chunks, ["delta"]) == (True, 2)
    assert rag_eval.evidence_hit(
        ["A pro\ufb01le defines the evaluation."], ["A profile defines"]
    ) == (True, 1)
    assert rag_eval.evidence_hit(["The model’s behavior changes."], ["model's behavior"]) == (
        True,
        1,
    )


def test_retrieval_only_evaluation_grades_against_ground_truth_evidence():
    svc, uid = make_svc(llm=None)
    r = rag_eval.evaluate(svc, uid, GOLD, None)["summary"]
    assert r["n_answerable"] == 2 and r["retrieval_hit_rate"]["k"] == 2
    assert r["retrieval_mrr"] > 0 and "abstention" not in r


def test_end_to_end_metrics_with_a_fake_model_and_judge():
    def reply(system, user):
        if "Reply with JSON only" in system:
            return '{"supported": true, "unsupported_claims": []}'
        if "Mona Lisa" in user or "Australia" in user:
            return "I don't have enough information in the provided documents to answer that."
        return "It is grounded in the text [1]."

    llm = FakeLLM(reply)
    svc, uid = make_svc(llm)
    out = rag_eval.evaluate(svc, uid, GOLD, llm)
    s = out["summary"]
    assert s["abstention"]["refused_unanswerable"]["k"] == 2
    assert s["abstention"]["refused_answerable_wrongly"]["k"] == 0
    assert s["abstention"]["f1"] == pytest.approx(1.0)
    assert s["citation_validity"]["k"] == 2 and s["faithfulness_judged"]["k"] == 2
    assert len(out["judgments"]) == 2


def test_judge_rate_limit_keeps_answers_and_stops_further_judging(capsys):
    class RateLimitedJudge(FakeLLM):
        def __init__(self):
            super().__init__()
            self.judge_calls = 0

        def complete(self, system, user, max_tokens=512):
            if system == rag_eval.JUDGE_SYSTEM:
                self.judge_calls += 1
                request = httpx.Request("POST", "https://example.test/chat/completions")
                response = httpx.Response(429, request=request)
                raise httpx.HTTPStatusError("rate limited", request=request, response=response)
            return "It is grounded in the text [1]."

    judge_llm = RateLimitedJudge()
    svc, uid = make_svc(judge_llm)

    result = rag_eval.evaluate(svc, uid, GOLD[:2], judge_llm)

    assert judge_llm.judge_calls == 1
    assert result["summary"]["faithfulness_judge_status"] == "rate_limited"
    assert result["summary"]["citation_validity"]["n"] == 2
    assert all(row["answer"] == "It is grounded in the text [1]." for row in result["per_question"])
    assert result["per_question"][0]["retrieved_top"]
    assert {
        "document",
        "chunk_index",
        "score",
        "trust",
        "excerpt",
    } <= set(result["per_question"][0]["retrieved_top"][0])
    assert "remaining faithfulness judgments" in capsys.readouterr().err
    assert "rate-limited" in rag_eval.to_markdown(result, None)


def test_unparseable_judge_output_is_excluded_not_counted_as_faithful():
    llm = FakeLLM(lambda s, u: "no json here" if "Reply with JSON only" in s else "Answer [1].")
    svc, uid = make_svc(llm)
    s = rag_eval.evaluate(svc, uid, GOLD[:1], llm)["summary"]
    assert s["faithfulness_judged"] == {"k": 0, "n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_calibration_picks_a_threshold_on_dev_and_reports_on_test():
    svc, uid = make_svc(llm=None)
    cal = rag_eval.calibrate(svc, uid, GOLD * 1, seed=1)
    assert cal["chosen_on"] == "dev" and "test" in cal
    assert 0.0 <= cal["test"]["balanced_accuracy"] <= 1.0


def test_choose_threshold_separates_clean_scores():
    thr = rag_eval.choose_threshold([5.0, 6.0, 7.0], [-3.0, -2.0, 0.5])
    assert 0.5 < thr <= 5.0
    ba = rag_eval.balanced_accuracy(thr, [5.0, 6.0, 7.0], [-3.0, -2.0, 0.5])
    assert ba["balanced_accuracy"] == 1.0


def test_cli_writes_report_handcheck_and_ragas_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=test-key\nSMARTRAG_LLM_MODEL=openai/gpt-oss-120b\n", encoding="utf-8"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(g) for g in GOLD))
    fake = FakeLLM(lambda s, u: '{"supported": true}' if "Reply with JSON only" in s else "Ok [1].")
    observed_keys = []
    observed_models = []

    def fake_groq(model=None, **kw):
        observed_keys.append(os.environ.get("GROQ_API_KEY"))
        observed_models.append(model)
        return fake

    monkeypatch.setattr(rag_eval, "GroqLLM", fake_groq)
    rag_eval.main(
        [
            "--docs",
            str(docs),
            "--goldset",
            str(gold),
            "--embedder",
            "hashing:64",
            "--reranker",
            "overlap",
            "--llm",
            "groq",
            "--calibrate",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    out = tmp_path / "out"
    assert observed_keys == ["test-key"]
    assert observed_models == ["openai/gpt-oss-120b"]
    assert {"rag.json", "rag.md", "judgments.jsonl", "handcheck.csv", "ragas.jsonl"} <= {
        p.name for p in out.iterdir()
    }
    rows = list(csv.reader(open(out / "handcheck.csv", encoding="utf-8")))
    assert rows[0][-1] == "human_supported" and len(rows) > 1
    ragas = [json.loads(line) for line in (out / "ragas.jsonl").read_text().splitlines()]
    assert {"user_input", "retrieved_contexts", "response", "reference"} <= set(ragas[0])


def test_rag_cli_skip_judge_avoids_extra_llm_calls(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "plants.txt").write_text(DOCS["plants.txt"], encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(GOLD[0]) + "\n", encoding="utf-8")
    llm = FakeLLM("It is grounded in the text [1].")
    monkeypatch.setattr(rag_eval, "GroqLLM", lambda **_: llm)

    result = rag_eval.main(
        [
            "--docs",
            str(docs),
            "--goldset",
            str(gold),
            "--embedder",
            "hashing:64",
            "--reranker",
            "overlap",
            "--llm",
            "groq",
            "--skip-judge",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert len(llm.calls) == 1
    assert "faithfulness_judged" not in result["summary"]
    assert result["per_question"][0]["answer"] == "It is grounded in the text [1]."


def test_rag_cli_gemini_uses_gemini_env_model(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "plants.txt").write_text(DOCS["plants.txt"], encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(GOLD[0]) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=test-key\nGEMINI_MODEL=gemini-test\n", encoding="utf-8"
    )
    fake = FakeLLM("Chlorophyll absorbs red and blue light [1].")
    observed_models = []
    observed_timeouts = []
    observed_rerank_depths = []

    def fake_gemini(model=None, timeout=None, **_):
        observed_models.append(model)
        observed_timeouts.append(timeout)
        return fake

    original_build_service = rag_eval.build_service

    def capture_build_service(docs, embedder, reranker, llm, settings, threshold):
        observed_rerank_depths.append(settings.rerank_top_n)
        return original_build_service(docs, embedder, reranker, llm, settings, threshold)

    monkeypatch.setattr(rag_eval, "GeminiLLM", fake_gemini)
    monkeypatch.setattr(rag_eval, "build_service", capture_build_service)
    rag_eval.main(
        [
            "--docs",
            str(docs),
            "--goldset",
            str(gold),
            "--embedder",
            "hashing:64",
            "--reranker",
            "none",
            "--llm",
            "gemini",
            "--llm-timeout",
            "300",
            "--rerank-top-n",
            "8",
            "--skip-judge",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert observed_models == ["gemini-test"]
    assert observed_timeouts == [300.0]
    assert observed_rerank_depths == [8]
    assert len(fake.calls) == 1


def test_rag_cli_rejects_missing_inputs_before_loading_models(tmp_path, monkeypatch, capsys):
    def unexpected_model_load(_):
        pytest.fail("models should not load when the input directory is missing")

    monkeypatch.setattr(rag_eval, "make_embedder", unexpected_model_load)
    with pytest.raises(SystemExit) as exc:
        rag_eval.main(
            [
                "--docs",
                str(tmp_path / "missing-docs"),
                "--goldset",
                str(tmp_path / "goldset.jsonl"),
            ]
        )

    assert exc.value.code == 2
    assert "--docs must be an existing directory" in capsys.readouterr().err


def test_rewrite_comparison_reports_hit_rates_with_and_without_rewriting():
    from smartrag.rewrite import QueryRewriter

    def reply(system, user):
        if "standalone search query" in system:
            return "chlorophyll absorbs red and blue light"
        return "Ok [1]."

    llm = FakeLLM(reply)
    emb = HashingEmbedder(64)
    svc = rag_eval.build_service(DOCS, emb, OverlapReranker(), llm, Settings(), None)[0]
    svc.rewriter = QueryRewriter(llm)
    uid = svc.store.get_user_by_email("eval@local.test").id
    gold = [
        {
            "id": "f1",
            "question": "And what does that absorb?",
            "answerable": True,
            "evidence": ["Chlorophyll absorbs red and blue light"],
            "history": [{"question": "What is chlorophyll?", "answer": "A pigment."}],
        }
    ]
    summary = rag_eval.evaluate(svc, uid, gold, None)["summary"]
    cmp = summary["rewrite_comparison"]
    assert cmp["follow_up_questions"] == 1 and cmp["hit_with_rewrite"]["k"] == 1
    assert "Follow-up questions (1)" in rag_eval.to_markdown({"summary": summary}, None)


def test_goldset_rejects_malformed_history(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "q", "answerable": False, "history": "nope"}))
    with pytest.raises(ValueError, match="history"):
        rag_eval.load_goldset(p)


def test_grid_edges_flags_parameters_at_the_boundary():
    ks, ws = [1, 10, 100], [0.1, 1.0, 2.0]
    assert run_ablation.grid_edges({"rrf_k": 10, "w_bm25": 1.0}, ks, ws) == []
    assert run_ablation.grid_edges({"rrf_k": 1, "w_bm25": 1.0}, ks, ws) == ["rrf_k"]
    assert run_ablation.grid_edges({"rrf_k": 100, "w_bm25": 0.1}, ks, ws) == ["rrf_k", "w_bm25"]


def test_ablation_reports_pairwise_comparisons_and_grid_search_range(toy, tmp_path):
    payload = run_ablation.main(
        [
            "--data",
            str(toy),
            "--embedder",
            "hashing:128",
            "--reranker",
            "overlap",
            "--out",
            str(tmp_path / "o"),
            "--cache",
            str(tmp_path / "c"),
            "--bootstrap",
            "200",
        ]
    )
    pairs = {(c["a"], c["b"]) for c in payload["pairwise_ndcg@10"]}
    assert len(pairs) == 2
    assert any(a.startswith("BM25 + dense (RRF)") and b.startswith("Dense") for a, b in pairs)
    assert any("rerank" in a and b == "BM25 + dense (RRF)" for a, b in pairs)
    assert payload["search_grid"]["rrf_k"][0] == 1 and payload["search_grid"]["w_bm25"][0] == 0.1
    md = (tmp_path / "o" / "toy-smoke" / "results.md").read_text()
    assert "Pairwise comparisons" in md and "not corrected for multiple comparisons" in md


def test_pairwise_skips_pairs_for_systems_that_do_not_exist():
    import numpy as np

    from evals.metrics import RunScores

    scored = [RunScores(["q"], {"ndcg@10": np.array([0.5, 0.6])}) for _ in range(4)]
    out = run_ablation.pairwise(list("abcd"), scored, [(3, 2), (4, 3)], 100, 0)
    assert len(out) == 1 and out[0]["a"] == "d"
