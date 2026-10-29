from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from smartrag.retrieval.pipeline import PoolChunk


class DuplicateError(ValueError):
    """Raised when a unique constraint (email, document name, or content hash) is violated."""


@dataclass(frozen=True)
class User:
    id: str
    email: str


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    owner_id: str
    name: str
    content_hash: str
    trust_score: float
    patterns: list[str]
    chunk_count: int
    source_type: str
    created_at: datetime
    shared: bool = False


@dataclass(frozen=True)
class ChunkRow:
    idx: int
    text: str
    content_hash: str
    embedding: np.ndarray


@dataclass(frozen=True)
class HistoryItem:
    id: int
    question: str
    answer: str
    answerable: bool
    created_at: datetime
    sources: list[dict] = field(default_factory=list)


class Store(Protocol):
    dim: int

    def init(self) -> None: ...
    def create_user(self, email: str, password_hash: str) -> User: ...
    def get_user_with_hash(self, email: str) -> tuple[User, str] | None: ...
    def get_user_by_email(self, email: str) -> User | None: ...
    def find_document_by_name(self, owner_id: str, name: str) -> DocumentRecord | None: ...
    def find_document_by_hash(self, owner_id: str, content_hash: str) -> DocumentRecord | None: ...
    def create_document(
        self,
        owner_id: str,
        name: str,
        content_hash: str,
        trust_score: float,
        patterns: list[str],
        source_type: str,
    ) -> DocumentRecord: ...
    def update_document(
        self, doc_id: str, content_hash: str, trust_score: float, patterns: list[str]
    ) -> None: ...
    def delete_document(self, owner_id: str, doc_id: str) -> bool: ...
    def list_documents(self, user_id: str) -> list[DocumentRecord]: ...
    def share_document(self, owner_id: str, doc_id: str, with_user_id: str) -> bool: ...
    def chunk_embeddings(self, doc_id: str) -> dict[str, np.ndarray]: ...
    def replace_chunks(self, doc_id: str, trust_score: float, rows: list[ChunkRow]) -> None: ...
    def pool(
        self, user_id: str, doc_ids: list[str] | None = None, min_trust: float = 0.0
    ) -> list[PoolChunk]: ...
    def dense_search(
        self,
        user_id: str,
        qvec: np.ndarray,
        k: int,
        doc_ids: list[str] | None = None,
        min_trust: float = 0.0,
    ) -> list[tuple[str, float]]: ...
    def add_history(
        self, user_id: str, question: str, answer: str, answerable: bool, sources: list[dict]
    ) -> None: ...
    def list_history(self, user_id: str, limit: int = 50) -> list[HistoryItem]: ...
