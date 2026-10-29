from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import sparse


class BM25Index:
    """Sparse-matrix BM25. The index is built once and queried in vectorised form.

    idf="lucene" gives log(1 + (N - df + 0.5) / (df + 0.5)), which is never negative.
    idf="okapi" reproduces rank_bm25.BM25Okapi, including its epsilon floor for negative
    idf values. The okapi mode exists so the implementation can be checked against that library.
    """

    def __init__(
        self, k1: float = 1.2, b: float = 0.75, idf: str = "lucene", epsilon: float = 0.25
    ) -> None:
        if idf not in {"lucene", "okapi"}:
            raise ValueError("idf must be 'lucene' or 'okapi'")
        self.k1, self.b, self.idf_mode, self.epsilon = k1, b, idf, epsilon
        self.vocab: dict[str, int] = {}
        self.n_docs = 0
        self._weights: sparse.csc_matrix | None = None

    def fit(self, docs: list[list[str]]) -> BM25Index:
        self.n_docs = len(docs)
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        doc_len = np.zeros(self.n_docs, dtype=np.float64)
        for i, tokens in enumerate(docs):
            doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(j)
                data.append(float(tf))
        self.vocab = vocab
        n_terms = len(vocab)
        tf = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_docs, n_terms))
        df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = self._idf(df)
        avgdl = float(doc_len.mean()) if self.n_docs else 0.0
        tf = tf.tocoo()
        denom_norm = 1.0 - self.b + self.b * (doc_len[tf.row] / avgdl if avgdl else 0.0)
        sat = tf.data * (self.k1 + 1.0) / (tf.data + self.k1 * denom_norm)
        weights = sparse.coo_matrix((sat * idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        self._weights = weights.tocsc()
        return self

    def _idf(self, df: np.ndarray) -> np.ndarray:
        n = self.n_docs
        if self.idf_mode == "lucene":
            return np.log1p((n - df + 0.5) / (df + 0.5))
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        if len(idf):
            floor = self.epsilon * float(idf.sum() / len(idf))
            idf = np.where(idf < 0, floor, idf)
        return idf

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score for every document. Repeated query terms count repeatedly."""
        assert self._weights is not None, "call fit() first"
        cols = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float64)
        return np.asarray(self._weights[:, cols].sum(axis=1)).ravel()

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        s = self.scores(query_tokens)
        return top_k(s, k, positive_only=True)


def top_k(scores: np.ndarray, k: int, positive_only: bool = False) -> list[tuple[int, float]]:
    """Indices of the k largest scores, best first. Ties break by lower index."""
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
    else:
        part = np.arange(n)
    order = sorted(part.tolist(), key=lambda i: (-scores[i], i))
    out = [(i, float(scores[i])) for i in order]
    if positive_only:
        out = [(i, s) for i, s in out if s > 0 and not math.isnan(s)]
    return out
