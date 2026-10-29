import random

import numpy as np

from smartrag.retrieval.lexical_v1 import LexicalV1

from .reference_v1 import hybrid_search

VOCAB = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron".split()


def test_lexical_v1_equals_the_original_v1_function():
    rng = random.Random(3)
    texts = [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(8, 40))) for _ in range(30)]
    baseline = LexicalV1(texts)
    for _ in range(10):
        query = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 4)))
        assert np.allclose(baseline.scores(query), hybrid_search(query, texts), atol=1e-12)


def test_v1_tfidf_component_has_no_idf_weighting():
    """HashingVectorizer returns normalised term frequency. A rare term and a common term
    contribute equally, which is why the v1 'TF-IDF' is hashed term frequency."""
    texts = ["common rare", "common x1", "common x2", "common x3"]
    base = LexicalV1(texts, weights=(1.0, 0.0))
    assert np.isclose(base.scores("rare")[0], base.scores("common")[0])
