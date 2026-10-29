from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings
from .dotenv import load_dotenv
from .ingest import extract_pdf_bytes, normalize_text
from .service import ServiceError, SmartRAGService

_STATUS = {
    "invalid_email": 422,
    "weak_password": 422,
    "email_taken": 409,
    "bad_credentials": 401,
    "unauthorized": 401,
    "invalid_name": 422,
    "empty_document": 422,
    "duplicate": 409,
    "not_found": 404,
    "empty_question": 422,
    "llm_unavailable": 503,
    "embedding_error": 502,
    "too_large": 413,
    "bad_pdf": 422,
}


class Credentials(BaseModel):
    email: str
    password: str


class TextUpload(BaseModel):
    name: str
    text: str


class HistoryTurn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(default="", max_length=4000)


class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    # Recent turns of the current conversation, used only to rewrite follow-up questions.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=6)


class ShareRequest(BaseModel):
    email: str = Field(min_length=3)


def build_service(settings: Settings | None = None) -> SmartRAGService:
    """Construct the production service from environment settings."""
    from .auth import TokenManager
    from .llm import GroqLLM
    from .retrieval import SentenceTransformerEmbedder, reranker_from_setting
    from .rewrite import QueryRewriter
    from .storage.postgres import PostgresStore

    load_dotenv()
    settings = settings or Settings.from_env()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is required. Set it in the environment or in a .env file "
            "in the project folder."
        )
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = PostgresStore(settings.database_url, dim=embedder.dim)
    store.init()
    llm = GroqLLM(model=settings.llm_model)
    return SmartRAGService(
        store=store,
        embedder=embedder,
        llm=llm,
        rewriter=QueryRewriter(llm),
        reranker=reranker_from_setting(settings.rerank_model),
        settings=settings,
        tokens=TokenManager(
            os.environ.get("SMARTRAG_JWT_PRIVATE_KEY_PEM") or None, settings.token_ttl_seconds
        ),
    )


def create_app(service: SmartRAGService | None = None) -> FastAPI:
    load_dotenv()
    holder: dict[str, SmartRAGService] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        holder["svc"] = service or build_service()
        yield

    app = FastAPI(title="SmartRAG v2", version="2.0.0", lifespan=lifespan)
    origins = [o for o in os.environ.get("SMARTRAG_CORS_ORIGINS", "").split(",") if o]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )
    if service is not None:
        holder["svc"] = service

    def svc() -> SmartRAGService:
        return holder["svc"]

    def current_user(
        authorization: str | None = Header(default=None), s: SmartRAGService = Depends(svc)
    ) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Missing bearer token.")
        try:
            return s.authenticate(authorization[7:].strip())
        except ServiceError as e:
            raise HTTPException(401, e.message) from e

    @app.exception_handler(ServiceError)
    async def service_error(_, exc: ServiceError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"error": exc.code, "detail": exc.message}, status_code=_STATUS.get(exc.code, 400)
        )

    def doc_json(d) -> dict:
        return {
            "id": d.id,
            "name": d.name,
            "trust_score": d.trust_score,
            "patterns": d.patterns,
            "chunk_count": d.chunk_count,
            "source_type": d.source_type,
            "shared": d.shared,
            "created_at": d.created_at.isoformat(),
        }

    def ingest_json(r) -> dict:
        return {
            "status": r.status,
            "document": doc_json(r.document),
            "chunks_total": r.chunks_total,
            "chunks_embedded": r.chunks_embedded,
            "chunks_reused": r.chunks_reused,
            "patterns": r.patterns,
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/auth/signup", status_code=201)
    def signup(body: Credentials, s: SmartRAGService = Depends(svc)):
        user, token = s.signup(body.email, body.password)
        return {"token": token, "user": {"id": user.id, "email": user.email}}

    @app.post("/auth/login")
    def login(body: Credentials, s: SmartRAGService = Depends(svc)):
        user, token = s.login(body.email, body.password)
        return {"token": token, "user": {"id": user.id, "email": user.email}}

    @app.get("/auth/public-key")
    def public_key(s: SmartRAGService = Depends(svc)):
        return {"algorithm": "ES256", "pem": s.tokens.public_pem}

    @app.post("/documents/pdf", status_code=201)
    async def upload_pdf(
        file: UploadFile = File(...),
        name: str | None = Form(default=None),
        uid: str = Depends(current_user),
        s: SmartRAGService = Depends(svc),
    ):
        data = await file.read(s.settings.max_upload_bytes + 1)
        if len(data) > s.settings.max_upload_bytes:
            raise ServiceError("too_large", "File exceeds the upload limit.")
        try:
            text = extract_pdf_bytes(data)
        except Exception as e:  # pymupdf raises several error types for malformed files
            raise ServiceError("bad_pdf", "Could not read this PDF.") from e
        result = s.ingest_text(uid, name or file.filename or "document.pdf", text, "pdf")
        return ingest_json(result)

    @app.post("/documents/text", status_code=201)
    def upload_text(
        body: TextUpload, uid: str = Depends(current_user), s: SmartRAGService = Depends(svc)
    ):
        return ingest_json(s.ingest_text(uid, body.name, normalize_text(body.text), "text"))

    @app.get("/documents")
    def documents(uid: str = Depends(current_user), s: SmartRAGService = Depends(svc)):
        return {"documents": [doc_json(d) for d in s.list_documents(uid)]}

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(
        doc_id: str, uid: str = Depends(current_user), s: SmartRAGService = Depends(svc)
    ):
        s.delete_document(uid, doc_id)

    @app.post("/documents/{doc_id}/share", status_code=204)
    def share_document(
        doc_id: str,
        body: ShareRequest,
        uid: str = Depends(current_user),
        s: SmartRAGService = Depends(svc),
    ):
        s.share_document(uid, doc_id, body.email)

    @app.post("/query")
    def query(
        body: QueryRequest, uid: str = Depends(current_user), s: SmartRAGService = Depends(svc)
    ):
        r = s.query(
            uid, body.question, body.doc_ids, history=[t.model_dump() for t in body.history]
        )
        return {
            "answer": r.answer,
            "answerable": r.answerable,
            "sources": r.sources,
            "blocked_reason": r.blocked_reason,
            "citations": r.citations,
            "invalid_citations": r.invalid_citations,
            "llm_called": r.llm_called,
            "rewritten_query": r.rewritten_query,
            "timings_ms": {k: round(v, 2) for k, v in r.timings_ms.items()},
        }

    @app.get("/history")
    def history(
        limit: int = 50, uid: str = Depends(current_user), s: SmartRAGService = Depends(svc)
    ):
        items = s.store.list_history(uid, min(max(limit, 1), 200))
        return {
            "history": [
                {
                    "id": i.id,
                    "question": i.question,
                    "answer": i.answer,
                    "answerable": i.answerable,
                    "created_at": i.created_at.isoformat(),
                    "sources": i.sources,
                }
                for i in items
            ]
        }

    return app
