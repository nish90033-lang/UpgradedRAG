from __future__ import annotations

from typing import Protocol

from .tokenize import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Cross-encoder reranker via sentence-transformers (install the `ml` extra)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.name = model_name
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


class OverlapReranker:
    """Test double: fraction of query terms present in the passage."""

    name = "overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(passages)
        return [len(q & set(tokenize(p))) / len(q) for p in passages]


_OFF = {"", "none", "off", "false", "0", "disabled"}


def reranker_from_setting(name: str | None) -> Reranker | None:
    """Build the configured reranker, or None when the setting turns it off.

    Loading the model is lazy, so a disabled reranker never imports sentence-transformers."""
    if name is None or name.strip().lower() in _OFF:
        return None
    return CrossEncoderReranker(name.strip())
