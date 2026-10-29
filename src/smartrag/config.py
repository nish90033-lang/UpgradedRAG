from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Defaults are safe for local development."""

    database_url: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "none"  # reranking is opt-in; see README
    trust_threshold: float = 50.0
    chunk_words: int = 200
    chunk_overlap: int = 40
    final_k: int = 8
    max_per_doc: int = 3
    rerank_top_n: int = 30
    max_upload_bytes: int = 25 * 1024 * 1024
    token_ttl_seconds: int = 7 * 24 * 3600

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            database_url=env("DATABASE_URL"),
            llm_model=env("SMARTRAG_LLM_MODEL", cls.llm_model),
            embedding_model=env("SMARTRAG_EMBEDDING_MODEL", cls.embedding_model),
            rerank_model=env("SMARTRAG_RERANK_MODEL", cls.rerank_model),
        )
