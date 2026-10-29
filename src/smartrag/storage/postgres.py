from __future__ import annotations

import json

import numpy as np
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from smartrag.retrieval.pipeline import PoolChunk

from .base import ChunkRow, DocumentRecord, DuplicateError, HistoryItem, User


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in v) + "]"


def _parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)


_ACCESS = """(d.owner_id = %(uid)s::uuid OR EXISTS (
    SELECT 1 FROM document_shares s WHERE s.document_id = d.id AND s.user_id = %(uid)s::uuid))"""


class PostgresStore:
    """PostgreSQL store with pgvector. Access control is enforced in SQL.

    exact=True disables index scans for dense queries. The per-user access filter would
    otherwise be applied after an approximate HNSW scan, which can return fewer than k rows.
    At per-user corpus sizes an exact scan is fast. Set exact=False only for very large
    shared corpora where the HNSW recall trade-off is acceptable.
    """

    def __init__(self, dsn: str, dim: int, exact: bool = True, max_size: int = 10) -> None:
        self.dsn, self.dim, self.exact = dsn, dim, exact
        self.pool_ = ConnectionPool(
            dsn, min_size=1, max_size=max_size, open=True, kwargs={"row_factory": dict_row}
        )

    def close(self) -> None:
        self.pool_.close()

    def init(self) -> None:
        with self.pool_.connection() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    email text NOT NULL UNIQUE,
                    password_hash text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now())""")
            c.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name text NOT NULL,
                    content_hash text NOT NULL,
                    trust_score real NOT NULL,
                    patterns text[] NOT NULL DEFAULT '{}',
                    chunk_count int NOT NULL DEFAULT 0,
                    source_type text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (owner_id, name),
                    UNIQUE (owner_id, content_hash))""")
            c.execute("""
                CREATE TABLE IF NOT EXISTS document_shares (
                    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    PRIMARY KEY (document_id, user_id))""")
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    idx int NOT NULL,
                    content text NOT NULL,
                    content_hash text NOT NULL,
                    trust_score real NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    UNIQUE (document_id, idx))""")
            c.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)""")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (document_id)")
            c.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id bigserial PRIMARY KEY,
                    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    question text NOT NULL,
                    answer text NOT NULL,
                    answerable boolean NOT NULL,
                    sources jsonb NOT NULL DEFAULT '[]',
                    created_at timestamptz NOT NULL DEFAULT now())""")

    def drop_all(self) -> None:
        """Drop every table. For tests only."""
        with self.pool_.connection() as c:
            c.execute(
                "DROP TABLE IF EXISTS chat_history, chunks, document_shares, documents, users "
                "CASCADE"
            )

    def reset(self) -> None:
        """Delete all rows. For tests only."""
        with self.pool_.connection() as c:
            c.execute("TRUNCATE chat_history, chunks, document_shares, documents, users CASCADE")

    # users
    def create_user(self, email: str, password_hash: str) -> User:
        email = email.strip().lower()
        try:
            with self.pool_.connection() as c:
                row = c.execute(
                    "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id::text",
                    (email, password_hash),
                ).fetchone()
        except UniqueViolation as e:
            raise DuplicateError("email already registered") from e
        return User(id=row["id"], email=email)

    def get_user_with_hash(self, email: str):
        with self.pool_.connection() as c:
            row = c.execute(
                "SELECT id::text, email, password_hash FROM users WHERE email = %s",
                (email.strip().lower(),),
            ).fetchone()
        return (User(row["id"], row["email"]), row["password_hash"]) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        found = self.get_user_with_hash(email)
        return found[0] if found else None

    # documents
    @staticmethod
    def _doc(row: dict, shared: bool = False) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            owner_id=row["owner_id"],
            name=row["name"],
            content_hash=row["content_hash"],
            trust_score=float(row["trust_score"]),
            patterns=list(row["patterns"]),
            chunk_count=row["chunk_count"],
            source_type=row["source_type"],
            created_at=row["created_at"],
            shared=shared,
        )

    _DOC_COLS = (
        "id::text, owner_id::text, name, content_hash, trust_score, patterns, "
        "chunk_count, source_type, created_at"
    )

    def find_document_by_name(self, owner_id: str, name: str):
        with self.pool_.connection() as c:
            row = c.execute(
                f"SELECT {self._DOC_COLS} FROM documents WHERE owner_id=%s::uuid AND name=%s",
                (owner_id, name),
            ).fetchone()
        return self._doc(row) if row else None

    def find_document_by_hash(self, owner_id: str, content_hash: str):
        with self.pool_.connection() as c:
            row = c.execute(
                f"SELECT {self._DOC_COLS} FROM documents "
                "WHERE owner_id=%s::uuid AND content_hash=%s",
                (owner_id, content_hash),
            ).fetchone()
        return self._doc(row) if row else None

    def create_document(self, owner_id, name, content_hash, trust_score, patterns, source_type):
        try:
            with self.pool_.connection() as c:
                row = c.execute(
                    f"""INSERT INTO documents
                        (owner_id, name, content_hash, trust_score, patterns, source_type)
                        VALUES (%s::uuid, %s, %s, %s, %s, %s) RETURNING {self._DOC_COLS}""",
                    (owner_id, name, content_hash, trust_score, patterns, source_type),
                ).fetchone()
        except UniqueViolation as e:
            raise DuplicateError("document name or content already exists") from e
        return self._doc(row)

    def update_document(self, doc_id, content_hash, trust_score, patterns) -> None:
        with self.pool_.connection() as c:
            c.execute(
                "UPDATE documents SET content_hash=%s, trust_score=%s, patterns=%s "
                "WHERE id=%s::uuid",
                (content_hash, trust_score, patterns, doc_id),
            )

    def delete_document(self, owner_id: str, doc_id: str) -> bool:
        with self.pool_.connection() as c:
            cur = c.execute(
                "DELETE FROM documents WHERE id=%s::uuid AND owner_id=%s::uuid", (doc_id, owner_id)
            )
            return cur.rowcount > 0

    def list_documents(self, user_id: str) -> list[DocumentRecord]:
        cols = (
            "d.id::text AS id, d.owner_id::text AS owner_id, d.name, d.content_hash, "
            "d.trust_score, d.patterns, d.chunk_count, d.source_type, d.created_at"
        )
        with self.pool_.connection() as c:
            rows = c.execute(
                f"SELECT {cols} FROM documents d WHERE {_ACCESS} ORDER BY d.created_at, d.name",
                {"uid": user_id},
            ).fetchall()
        return [self._doc(r, shared=r["owner_id"] != user_id) for r in rows]

    def share_document(self, owner_id: str, doc_id: str, with_user_id: str) -> bool:
        if owner_id == with_user_id:
            return False
        with self.pool_.connection() as c:
            ok = c.execute(
                "SELECT 1 FROM documents WHERE id=%s::uuid AND owner_id=%s::uuid",
                (doc_id, owner_id),
            ).fetchone()
            target = c.execute("SELECT 1 FROM users WHERE id=%s::uuid", (with_user_id,)).fetchone()
            if not ok or not target:
                return False
            c.execute(
                "INSERT INTO document_shares (document_id, user_id) VALUES (%s::uuid, %s::uuid) "
                "ON CONFLICT DO NOTHING",
                (doc_id, with_user_id),
            )
        return True

    # chunks
    def chunk_embeddings(self, doc_id: str) -> dict[str, np.ndarray]:
        with self.pool_.connection() as c:
            rows = c.execute(
                "SELECT content_hash, embedding::text AS e FROM chunks WHERE document_id=%s::uuid",
                (doc_id,),
            ).fetchall()
        return {r["content_hash"]: _parse_vec(r["e"]) for r in rows}

    def replace_chunks(self, doc_id: str, trust_score: float, rows: list[ChunkRow]) -> None:
        for r in rows:
            if r.embedding.shape != (self.dim,):
                raise ValueError(f"embedding must have dimension {self.dim}")
        with self.pool_.connection() as c, c.transaction():
            c.execute("DELETE FROM chunks WHERE document_id=%s::uuid", (doc_id,))
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, idx, content, content_hash, trust_score, "
                    "embedding) VALUES (%s::uuid, %s, %s, %s, %s, %s::vector)",
                    [
                        (doc_id, r.idx, r.text, r.content_hash, trust_score, _vec(r.embedding))
                        for r in rows
                    ],
                )
            c.execute(
                "UPDATE documents SET chunk_count=%s, trust_score=%s WHERE id=%s::uuid",
                (len(rows), trust_score, doc_id),
            )

    def pool(self, user_id, doc_ids=None, min_trust=0.0) -> list[PoolChunk]:
        sql = f"""SELECT c.id::text AS id, d.id::text AS doc_id, d.name, c.idx, c.content,
                         c.trust_score
                  FROM chunks c JOIN documents d ON d.id = c.document_id
                  WHERE {_ACCESS} AND c.trust_score >= %(mt)s"""
        params: dict = {"uid": user_id, "mt": min_trust}
        if doc_ids is not None:
            sql += " AND d.id = ANY(%(docs)s::uuid[])"
            params["docs"] = doc_ids
        sql += " ORDER BY d.created_at, d.name, c.idx"
        with self.pool_.connection() as c:
            rows = c.execute(sql, params).fetchall()
        return [
            PoolChunk(
                r["id"], r["doc_id"], r["name"], r["idx"], r["content"], float(r["trust_score"])
            )
            for r in rows
        ]

    def dense_search(self, user_id, qvec, k, doc_ids=None, min_trust=0.0):
        sql = f"""SELECT c.id::text AS id, 1 - (c.embedding <=> %(q)s::vector) AS sim
                  FROM chunks c JOIN documents d ON d.id = c.document_id
                  WHERE {_ACCESS} AND c.trust_score >= %(mt)s"""
        params: dict = {"uid": user_id, "mt": min_trust, "q": _vec(qvec), "k": k}
        if doc_ids is not None:
            sql += " AND d.id = ANY(%(docs)s::uuid[])"
            params["docs"] = doc_ids
        sql += " ORDER BY c.embedding <=> %(q)s::vector LIMIT %(k)s"
        with self.pool_.connection() as c, c.transaction():
            if self.exact:
                c.execute("SET LOCAL enable_indexscan = off")
            rows = c.execute(sql, params).fetchall()
        return [(r["id"], float(r["sim"])) for r in rows]

    # history
    def add_history(self, user_id, question, answer, answerable, sources) -> None:
        with self.pool_.connection() as c:
            c.execute(
                "INSERT INTO chat_history (user_id, question, answer, answerable, sources) "
                "VALUES (%s::uuid, %s, %s, %s, %s::jsonb)",
                (user_id, question, answer, answerable, json.dumps(sources)),
            )

    def list_history(self, user_id: str, limit: int = 50) -> list[HistoryItem]:
        with self.pool_.connection() as c:
            rows = c.execute(
                "SELECT id, question, answer, answerable, sources, created_at FROM chat_history "
                "WHERE user_id=%s::uuid ORDER BY id DESC LIMIT %s",
                (user_id, limit),
            ).fetchall()
        return [
            HistoryItem(
                r["id"], r["question"], r["answer"], r["answerable"], r["created_at"], r["sources"]
            )
            for r in rows
        ]
