import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from evals.beir_data import load_beir, split_queries
from evals.metrics import (
    evaluate_run,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
    wilson_interval,
)
from evals.synthetic import make_synthetic_beir


def test_ndcg_hand_computed_binary():
    # ranked d1,d2,d3 with d1 and d3 relevant: DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3)
    got = ndcg_at_k(["d1", "d2", "d3"], {"d1": 1, "d3": 1}, 3)
    assert got == pytest.approx(1.5 / (1 + 1 / math.log2(3)))


def test_ndcg_hand_computed_graded():
    got = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2)
    assert got == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_ndcg_perfect_and_empty_cases():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a": 1}, 10) == 0.0
    assert ndcg_at_k([], {"a": 1}, 10) == 0.0
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_recall_and_mrr():
    qrel = {"a": 1, "b": 1, "c": 0, "d": 1}
    ranked = ["x", "a", "c", "b", "y", "z"]
    assert recall_at_k(ranked, qrel, 4) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, qrel, 1) == 0.0
    assert mrr_at_k(ranked, qrel, 10) == pytest.approx(0.5)
    assert mrr_at_k(["x", "y"], qrel, 10) == 0.0


def test_ndcg_matches_scikit_learn_on_random_rankings():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = 30
        rel = rng.choice([0, 0, 0, 1, 2, 3], size=n)
        if rel.sum() == 0:
            continue
        order = rng.permutation(n)  # ranked list of all documents
        ranked = [f"d{i}" for i in order]
        qrel = {f"d{i}": int(rel[i]) for i in range(n)}
        y_score = np.empty(n)
        y_score[order] = np.arange(n, 0, -1)  # strictly decreasing along the ranking, no ties
        for k in (5, 10):
            expected = ndcg_score(rel.reshape(1, -1), y_score.reshape(1, -1), k=k)
            assert ndcg_at_k(ranked, qrel, k) == pytest.approx(expected, abs=1e-9)


def test_evaluate_run_skips_queries_without_relevant_docs_and_scores_missing_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
    s = evaluate_run({"q1": ["a"]}, qrels)
    assert s.qids == ["q1", "q2"]
    assert s.per_query["ndcg@10"].tolist() == [1.0, 0.0]


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.2, 0.8, 200)
    clear = paired_bootstrap(b + 0.1, b, n_resamples=2000)
    assert clear["ci_low"] > 0 and clear["p_value"] < 0.01 and clear["diff"] == pytest.approx(0.1)
    null = paired_bootstrap(b, b[::-1].copy(), n_resamples=2000)
    assert null["ci_low"] < 0 < null["ci_high"]
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1.0]), np.array([1.0, 2.0]))


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0


def test_split_is_deterministic_disjoint_and_seed_dependent():
    ids = [f"q{i}" for i in range(40)]
    a = split_queries(ids, seed=1)
    assert a == split_queries(ids, seed=1)
    assert not set(a[0]) & set(a[1]) and sorted(a[0] + a[1]) == sorted(ids)
    assert a != split_queries(ids, seed=2)


def test_beir_loader_roundtrip(tmp_path):
    root = make_synthetic_beir(tmp_path / "toy", n_topics=3, docs_per_topic=5)
    ds = load_beir(root)
    assert len(ds.corpus) == 15 and len(ds.queries) == 9 == len(ds.qrels)
    assert all(v == {k: 1} or isinstance(v, dict) for k, v in ds.qrels.items())
