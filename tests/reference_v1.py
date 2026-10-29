"""Verbatim copy of the v1 scoring function, used to prove LexicalV1 is faithful.

Source: backend/retrieval.py of the v1 repository (hybrid_search and its vectorizer).
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    analyzer="word",
    ngram_range=(1, 2),
    stop_words="english",
)


def hybrid_search(query: str, chunks: list[str], weights: tuple = (0.7, 0.3)) -> list[float]:
    if not chunks:
        return []
    tfidf_w, bm25_w = weights
    all_texts = chunks + [query]
    matrix = vectorizer.transform(all_texts)
    chunk_vectors = matrix[:-1]
    query_vector = matrix[-1]
    tfidf_scores = cosine_similarity(query_vector, chunk_vectors).flatten()
    tokenized_chunks = [c.lower().split() for c in chunks]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_raw = np.array(bm25.get_scores(tokenized_query))
    bm25_raw = np.maximum(bm25_raw, 0)
    bm25_max = bm25_raw.max()
    bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
    combined = tfidf_w * tfidf_scores + bm25_w * bm25_scores
    return combined.tolist()
