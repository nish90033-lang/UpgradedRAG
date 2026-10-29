from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .auth import AuthError, TokenManager, hash_password, verify_password
from .config import Settings
from .ingest import chunk_words, content_hash, sanitize, trust_score
from .llm import LLM, build_prompt, is_abstention, parse_citations
from .retrieval import Embedder, Reranker, Retriever
from .retrieval.pipeline import RetrievalConfig, RetrievalResult
from .rewrite import QueryRewriter
from .storage import ChunkRow, DocumentRecord, DuplicateError, Store, User


class ServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class IngestResult:
    status: str  # created | updated | unchanged
    document: DocumentRecord
    chunks_total: int
    chunks_embedded: int
    chunks_reused: int
    patterns: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: str
    answerable: bool
    sources: list[dict]
    blocked_reason: str | None
    citations: list[int]
    invalid_citations: list[int]
    llm_called: bool
    timings_ms: dict[str, float]
    rewritten_query: str | None = None


class SmartRAGService:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
        tokens: TokenManager | None = None,
        abstain_threshold: float | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.store, self.embedder, self.llm, self.reranker = store, embedder, llm, reranker
        self.settings = settings or Settings()
        self.tokens = tokens or TokenManager(ttl_seconds=self.settings.token_ttl_seconds)
        self.retriever = Retriever(embedder, reranker)
        self.abstain_threshold = abstain_threshold
        self.rewriter = rewriter
        if embedder.dim != store.dim:
            raise ValueError(f"embedder dim {embedder.dim} != store dim {store.dim}")

    # auth
    def signup(self, email: str, password: str) -> tuple[User, str]:
        email = email.strip().lower()
        if "@" not in email or len(email) < 3:
            raise ServiceError("invalid_email", "Enter a valid email address.")
        if len(password) < 8:
            raise ServiceError("weak_password", "Password must be at least 8 characters.")
        try:
            user = self.store.create_user(email, hash_password(password))
        except DuplicateError as e:
            raise ServiceError("email_taken", "That email is already registered.") from e
        return user, self.tokens.issue(user.id, user.email)

    def login(self, email: str, password: str) -> tuple[User, str]:
        found = self.store.get_user_with_hash(email)
        # Same error for unknown email and wrong password.
        if not found or not verify_password(password, found[1]):
            raise ServiceError("bad_credentials", "Incorrect email or password.")
        return found[0], self.tokens.issue(found[0].id, found[0].email)

    def authenticate(self, token: str) -> str:
        try:
            return self.tokens.verify(token).user_id
        except AuthError as e:
            raise ServiceError("unauthorized", "Invalid or expired token.") from e

    # ingestion
    def ingest_text(
        self, user_id: str, name: str, text: str, source_type: str = "text"
    ) -> IngestResult:
        name = name.strip()
        if not name:
            raise ServiceError("invalid_name", "A document name is required.")
        if not text or not text.strip():
            raise ServiceError("empty_document", "The document contains no text.")

        raw_hash = content_hash(text)
        same = self.store.find_document_by_hash(user_id, raw_hash)
        if same is not None:
            if same.name == name:
                return IngestResult(
                    "unchanged", same, same.chunk_count, 0, same.chunk_count, list(same.patterns)
                )
            raise ServiceError("duplicate", f"This content is already stored as '{same.name}'.")

        cleaned = sanitize(text)
        trust = trust_score(cleaned.matched)
        chunks = chunk_words(
            cleaned.clean_text, self.settings.chunk_words, self.settings.chunk_overlap
        )
        if not chunks:
            raise ServiceError("empty_document", "Nothing remained after sanitisation.")

        existing = self.store.find_document_by_name(user_id, name)
        known = self.store.chunk_embeddings(existing.id) if existing else {}
        missing = list(dict.fromkeys(c.content_hash for c in chunks if c.content_hash not in known))
        fresh: dict[str, np.ndarray] = {}
        if missing:
            by_hash = {c.content_hash: c.text for c in chunks}
            vecs = np.asarray(self.embedder.embed_documents([by_hash[h] for h in missing]))
            if vecs.shape != (len(missing), self.embedder.dim):
                raise ServiceError("embedding_error", "Embedder returned an unexpected shape.")
            fresh = dict(zip(missing, vecs, strict=True))
        rows = [
            ChunkRow(
                c.idx,
                c.text,
                c.content_hash,
                known[c.content_hash] if c.content_hash in known else fresh[c.content_hash],
            )
            for c in chunks
        ]
        reused = sum(1 for c in chunks if c.content_hash in known)

        try:
            if existing:
                self.store.update_document(existing.id, raw_hash, trust, cleaned.matched)
                doc_id, status = existing.id, "updated"
            else:
                doc = self.store.create_document(
                    user_id, name, raw_hash, trust, cleaned.matched, source_type
                )
                doc_id, status = doc.id, "created"
        except DuplicateError as e:
            raise ServiceError("duplicate", "A document with this name or content exists.") from e
        self.store.replace_chunks(doc_id, trust, rows)
        record = next(d for d in self.store.list_documents(user_id) if d.id == doc_id)
        return IngestResult(
            status, record, len(chunks), len(chunks) - reused, reused, cleaned.matched
        )

    def delete_document(self, user_id: str, doc_id: str) -> None:
        if not self.store.delete_document(user_id, doc_id):
            raise ServiceError("not_found", "Document not found.")

    def share_document(self, user_id: str, doc_id: str, with_email: str) -> None:
        target = self.store.get_user_by_email(with_email)
        if target is None or not self.store.share_document(user_id, doc_id, target.id):
            raise ServiceError("not_found", "Document or user not found.")

    def list_documents(self, user_id: str) -> list[DocumentRecord]:
        return self.store.list_documents(user_id)

    # query
    def retrieval_config(self) -> RetrievalConfig:
        s = self.settings
        return RetrievalConfig(
            mode="hybrid",
            final_k=s.final_k,
            max_per_doc=s.max_per_doc,
            rerank_top_n=s.rerank_top_n,
            trust_threshold=s.trust_threshold,
            abstain_threshold=self.abstain_threshold,
        )

    def retrieve(
        self,
        user_id: str,
        question: str,
        doc_ids: list[str] | None = None,
        cfg: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = cfg or self.retrieval_config()
        pool = self.store.pool(user_id, doc_ids)
        return self.retriever.retrieve(
            question,
            pool,
            lambda q, k, mt: self.store.dense_search(user_id, q, k, doc_ids, mt),
            cfg,
        )

    def rewrite_question(
        self, question: str, history: list[dict] | None
    ) -> tuple[str, str | None, float | None]:
        """(query to search, rewritten text if it changed, milliseconds spent or None)."""
        if self.rewriter is None or not history:
            return question, None, None
        t = time.perf_counter()
        res = self.rewriter.rewrite(question, history)
        elapsed = (time.perf_counter() - t) * 1000
        return res.query, (res.query if res.rewritten else None), elapsed

    def query(
        self,
        user_id: str,
        question: str,
        doc_ids: list[str] | None = None,
        record_history: bool = True,
        history: list[dict] | None = None,
    ) -> QueryResult:
        question = question.strip()
        if not question:
            raise ServiceError("empty_question", "Ask a question.")
        t0 = time.perf_counter()
        search_q, rewritten_query, rewrite_ms = self.rewrite_question(question, history)
        result = self.retrieve(user_id, search_q, doc_ids)
        timings = dict(result.timings_ms)
        if rewrite_ms is not None:
            timings["rewrite_ms"] = rewrite_ms

        sources = [
            {
                "n": i,
                "doc_id": h.chunk.doc_id,
                "doc_name": h.chunk.doc_name,
                "chunk_id": h.chunk.chunk_id,
                "chunk_index": h.chunk.idx,
                "excerpt": h.chunk.text[:400],
                "score": round(float(h.score), 6),
                "trust": h.chunk.trust,
            }
            for i, h in enumerate(result.hits, start=1)
        ]

        if not result.answerable:
            message = {
                "no_documents": "No documents are available to search.",
                "all_blocked": "The selected documents were blocked by the safety filter.",
            }.get(result.blocked_reason or "", "No sufficiently relevant content was found.")
            return self._finish(
                user_id,
                question,
                message,
                False,
                [],
                result.blocked_reason,
                [],
                [],
                False,
                timings,
                t0,
                record_history,
                rewritten_query,
            )

        if self.llm is None:
            raise ServiceError("llm_unavailable", "No language model is configured.")
        system, user = build_prompt(search_q, [h.chunk.text for h in result.hits])
        s = time.perf_counter()
        answer = self.llm.complete(system, user)
        timings["llm_ms"] = (time.perf_counter() - s) * 1000
        if is_abstention(answer):
            return self._finish(
                user_id,
                question,
                answer,
                False,
                [],
                "model_abstained",
                [],
                [],
                True,
                timings,
                t0,
                record_history,
                rewritten_query,
            )
        valid, invalid = parse_citations(answer, len(sources))
        return self._finish(
            user_id,
            question,
            answer,
            True,
            sources,
            None,
            valid,
            invalid,
            True,
            timings,
            t0,
            record_history,
            rewritten_query,
        )

    def _finish(
        self,
        user_id,
        question,
        answer,
        answerable,
        sources,
        blocked,
        valid,
        invalid,
        llm_called,
        timings,
        t0,
        record_history,
        rewritten_query=None,
    ) -> QueryResult:
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        if record_history:
            self.store.add_history(user_id, question, answer, answerable, sources)
        return QueryResult(
            answer,
            answerable,
            sources,
            blocked,
            valid,
            invalid,
            llm_called,
            timings,
            rewritten_query,
        )
