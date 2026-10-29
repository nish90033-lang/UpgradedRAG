from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    idx: int
    text: str
    content_hash: str


def content_hash(text: str) -> str:
    """SHA-256 of whitespace-normalised text. Stable across reflowed copies of a document."""
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def chunk_words(
    text: str, size: int = 200, overlap: int = 40, max_chunks: int | None = None
) -> list[Chunk]:
    """Split text into overlapping word windows.

    The final window ends exactly at the last word, so no trailing chunk is a strict
    subset of its predecessor. Each chunk carries a content hash used for incremental sync.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    words = text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        body = " ".join(window)
        chunks.append(Chunk(idx=len(chunks), text=body, content_hash=content_hash(body)))
        if start + size >= len(words):
            break
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks
