import random

import numpy as np
from rank_bm25 import BM25Okapi

from smartrag.retrieval.bm25 import BM25Index, top_k
from smartrag.retrieval.tokenize import tokenize


def _corpus(seed=0, n_docs=60, vocab=80):
    rng = random.Random(seed)
    words = [f"t{i}" for i in range(vocab)]
    return [[rng.choice(words) for _ in range(rng.randint(5, 60))] for _ in range(n_docs)]


def test_okapi_mode_matches_rank_bm25_exactly():
    docs = _corpus()
    ref = BM25Okapi(docs, k1=1.5, b=0.75, epsilon=0.25)
    mine = BM25Index(k1=1.5, b=0.75, idf="okapi", epsilon=0.25).fit(docs)
    rng = random.Random(1)
    for _ in range(25):
        q = [f"t{rng.randrange(90)}" for _ in range(rng.randint(1, 6))]  # includes unknown terms
        assert np.allclose(mine.scores(q), ref.get_scores(q), atol=1e-9)


def test_lucene_idf_is_never_negative_and_ranks_matching_doc_first():
    docs = [
        tokenize(t)
        for t in [
            "the cat sat on the mat",
            "dogs chase cats in the park",
            "stock markets fell sharply",
            "the cat and the dog",
            "quantum computing uses qubits",
        ]
    ]
    index = BM25Index().fit(docs)
    assert (index._weights.data >= 0).all()
    top = index.search(tokenize("qubits"), 3)
    assert top[0][0] == 4 and len(top) == 1  # only one doc contains the term


def test_unknown_terms_score_zero_and_search_returns_nothing():
    index = BM25Index().fit([["a", "b"], ["c"]])
    assert index.scores(["zzz"]).tolist() == [0.0, 0.0]
    assert index.search(["zzz"], 5) == []


def test_repeated_query_terms_count_repeatedly():
    index = BM25Index().fit([["a", "b"], ["b", "c"], ["d"]])
    assert np.isclose(index.scores(["a", "a"])[0], 2 * index.scores(["a"])[0])


def test_top_k_orders_descending_with_stable_ties():
    s = np.array([0.1, 0.9, 0.9, 0.5, 0.0])
    assert [i for i, _ in top_k(s, 3)] == [1, 2, 3]
    assert [i for i, _ in top_k(s, 10, positive_only=True)] == [1, 2, 3, 0]
    assert top_k(s, 0) == [] and top_k(np.array([]), 3) == []


def test_length_normalisation_prefers_the_shorter_document():
    short = ["apple", "pie"]
    long = ["apple"] + [f"filler{i}" for i in range(80)]
    index = BM25Index().fit([long, short])
    s = index.scores(["apple"])
    assert s[1] > s[0]
