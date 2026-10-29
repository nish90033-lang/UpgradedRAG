"""The v1 lexical scorer, kept as the evaluation baseline.

It reproduces the v1 scoring faithfully: HashingVectorizer features (hashed term frequency
with L2 norm; HashingVectorizer applies no idf weighting) scored by cosine similarity, fused
70/30 with BM25Okapi over lowercase whitespace tokens, where BM25 is floored at zero and
divided by its per-query maximum.
"""

from __future__ import annotations

import numpy as np


class LexicalV1:
    def __init__(self, texts: list[str], weights: tuple[float, float] = (0.7, 0.3)) -> None:
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import HashingVectorizer

        self.weights = weights
        self.vectorizer = HashingVectorizer(
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.matrix = self.vectorizer.transform(texts)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def scores(self, query: str) -> np.ndarray:
        qv = self.vectorizer.transform([query])
        tfidf = (self.matrix @ qv.T).toarray().ravel()
        raw = np.maximum(np.asarray(self.bm25.get_scores(query.lower().split())), 0)
        top = raw.max() if len(raw) else 0.0
        bm25 = raw / top if top > 0 else raw
        return self.weights[0] * tfidf + self.weights[1] * bm25
