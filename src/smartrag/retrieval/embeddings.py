from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from .tokenize import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    This is a test double. It has no semantic knowledge, so it must never be used to
    report retrieval quality. It lets tests exercise the full dense path offline.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.calls = 0
        self.texts_embedded = 0

    def _embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return l2_normalize(out)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers (install the `ml` extra)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        doc_prefix: str = "",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.query_prefix, self.doc_prefix, self.batch_size = query_prefix, doc_prefix, batch_size

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.doc_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


class OpenAICompatEmbedder:
    """Remote embeddings through any OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, timeout: float = 60.0):
        import httpx

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.name, self.dim, self._model = model, dim, model

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for start in range(0, len(texts), 96):
            batch = texts[start : start + 96]
            r = self._client.post("/embeddings", json={"model": self._model, "input": batch})
            r.raise_for_status()
            rows = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in rows)
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)
