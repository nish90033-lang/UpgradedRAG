from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from smartrag.retrieval.pipeline import ArrayDenseIndex, PoolChunk

from .base import ChunkRow, DocumentRecord, DuplicateError, HistoryItem, User


class _Doc:
    def __init__(self, rec: DocumentRecord) -> None:
        self.rec = rec
        self.chunks: list[tuple[str, ChunkRow]] = []  # (chunk_id, row)


class MemoryStore:
    """In-process store with the same semantics as PostgresStore. Used for tests and eval."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._lock = threading.RLock()
        self._users: dict[str, tuple[User, str]] = {}
        self._emails: dict[str, str] = {}
        self._docs: dict[str, _Doc] = {}
        self._shares: set[tuple[str, str]] = set()
        self._history: list[tuple[str, HistoryItem]] = []
        self._history_seq = 0

    def init(self) -> None:
        return None

    # users
    def create_user(self, email: str, password_hash: str) -> User:
        email = email.strip().lower()
        with self._lock:
            if email in self._emails:
                raise DuplicateError("email already registered")
            user = User(id=str(uuid.uuid4()), email=email)
            self._users[user.id] = (user, password_hash)
            self._emails[email] = user.id
            return user

    def get_user_with_hash(self, email: str):
        with self._lock:
            uid = self._emails.get(email.strip().lower())
            return self._users[uid] if uid else None

    def get_user_by_email(self, email: str) -> User | None:
        found = self.get_user_with_hash(email)
        return found[0] if found else None

    # documents
    def find_document_by_name(self, owner_id: str, name: str):
        with self._lock:
            for d in self._docs.values():
                if d.rec.owner_id == owner_id and d.rec.name == name:
                    return d.rec
        return None

    def find_document_by_hash(self, owner_id: str, content_hash: str):
        with self._lock:
            for d in self._docs.values():
                if d.rec.owner_id == owner_id and d.rec.content_hash == content_hash:
                    return d.rec
        return None

    def create_document(self, owner_id, name, content_hash, trust_score, patterns, source_type):
        with self._lock:
            if self.find_document_by_name(owner_id, name):
                raise DuplicateError("document name already exists")
            if self.find_document_by_hash(owner_id, content_hash):
                raise DuplicateError("document content already exists")
            rec = DocumentRecord(
                id=str(uuid.uuid4()),
                owner_id=owner_id,
                name=name,
                content_hash=content_hash,
                trust_score=trust_score,
                patterns=list(patterns),
                chunk_count=0,
                source_type=source_type,
                created_at=datetime.now(UTC),
            )
            self._docs[rec.id] = _Doc(rec)
            return rec

    def update_document(self, doc_id, content_hash, trust_score, patterns) -> None:
        with self._lock:
            d = self._docs[doc_id]
            d.rec = replace(
                d.rec, content_hash=content_hash, trust_score=trust_score, patterns=list(patterns)
            )

    def delete_document(self, owner_id: str, doc_id: str) -> bool:
        with self._lock:
            d = self._docs.get(doc_id)
            if not d or d.rec.owner_id != owner_id:
                return False
            del self._docs[doc_id]
            self._shares = {s for s in self._shares if s[0] != doc_id}
            return True

    def _accessible(self, user_id: str, doc_ids: list[str] | None) -> list[_Doc]:
        wanted = set(doc_ids) if doc_ids is not None else None
        out = []
        for d in self._docs.values():
            if wanted is not None and d.rec.id not in wanted:
                continue
            if d.rec.owner_id == user_id or (d.rec.id, user_id) in self._shares:
                out.append(d)
        return out

    def list_documents(self, user_id: str) -> list[DocumentRecord]:
        with self._lock:
            return [
                replace(d.rec, shared=d.rec.owner_id != user_id)
                for d in self._accessible(user_id, None)
            ]

    def share_document(self, owner_id: str, doc_id: str, with_user_id: str) -> bool:
        with self._lock:
            d = self._docs.get(doc_id)
            if not d or d.rec.owner_id != owner_id or with_user_id == owner_id:
                return False
            if with_user_id not in self._users:
                return False
            self._shares.add((doc_id, with_user_id))
            return True

    # chunks
    def chunk_embeddings(self, doc_id: str) -> dict[str, np.ndarray]:
        with self._lock:
            d = self._docs.get(doc_id)
            return {row.content_hash: row.embedding for _, row in d.chunks} if d else {}

    def replace_chunks(self, doc_id: str, trust_score: float, rows: list[ChunkRow]) -> None:
        with self._lock:
            d = self._docs[doc_id]
            for r in rows:
                if r.embedding.shape != (self.dim,):
                    raise ValueError(f"embedding must have dimension {self.dim}")
            d.chunks = [(str(uuid.uuid4()), r) for r in rows]
            d.rec = replace(d.rec, chunk_count=len(rows), trust_score=trust_score)

    def pool(self, user_id, doc_ids=None, min_trust=0.0) -> list[PoolChunk]:
        with self._lock:
            out: list[PoolChunk] = []
            for d in self._accessible(user_id, doc_ids):
                if d.rec.trust_score < min_trust:
                    continue
                for cid, row in d.chunks:
                    out.append(
                        PoolChunk(cid, d.rec.id, d.rec.name, row.idx, row.text, d.rec.trust_score)
                    )
            return out

    def dense_search(self, user_id, qvec, k, doc_ids=None, min_trust=0.0):
        with self._lock:
            ids: list[str] = []
            vecs: list[np.ndarray] = []
            for d in self._accessible(user_id, doc_ids):
                if d.rec.trust_score < min_trust:
                    continue
                for cid, row in d.chunks:
                    ids.append(cid)
                    vecs.append(row.embedding)
            if not ids:
                return []
            return ArrayDenseIndex(ids, np.vstack(vecs))(qvec, k)

    # history
    def add_history(self, user_id, question, answer, answerable, sources) -> None:
        with self._lock:
            self._history_seq += 1
            item = HistoryItem(
                self._history_seq, question, answer, answerable, datetime.now(UTC), sources
            )
            self._history.append((user_id, item))

    def list_history(self, user_id: str, limit: int = 50) -> list[HistoryItem]:
        with self._lock:
            items = [i for uid, i in self._history if uid == user_id]
            return list(reversed(items))[:limit]
