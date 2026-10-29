from __future__ import annotations

import os

import pytest

from smartrag.config import Settings
from smartrag.llm import FakeLLM
from smartrag.retrieval import HashingEmbedder, OverlapReranker
from smartrag.service import SmartRAGService
from smartrag.storage import MemoryStore

PG_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://smartrag:smartrag@localhost:5432/smartrag_test"
)


def _pg_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(PG_DSN, connect_timeout=2) as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


PG_OK = _pg_available()


def make_store(kind: str, dim: int):
    if kind == "memory":
        return MemoryStore(dim)
    if not PG_OK:
        pytest.skip("PostgreSQL with pgvector is not reachable (set TEST_DATABASE_URL)")
    from smartrag.storage.postgres import PostgresStore

    store = PostgresStore(PG_DSN, dim=dim)
    store.drop_all()
    store.init()
    return store


@pytest.fixture(params=["memory", "postgres"])
def store_kind(request):
    return request.param


@pytest.fixture
def store8(store_kind):
    s = make_store(store_kind, 8)
    yield s
    if hasattr(s, "close"):
        s.close()


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=64)


@pytest.fixture
def llm():
    return FakeLLM("Grounded answer [1]")


@pytest.fixture
def service(store_kind, embedder, llm):
    store = make_store(store_kind, embedder.dim)
    svc = SmartRAGService(
        store=store,
        embedder=embedder,
        llm=llm,
        reranker=OverlapReranker(),
        settings=Settings(chunk_words=200, chunk_overlap=40, final_k=6, max_per_doc=3),
    )
    yield svc
    if hasattr(store, "close"):
        store.close()


@pytest.fixture
def memory_service(embedder, llm):
    store = MemoryStore(embedder.dim)
    return SmartRAGService(
        store=store,
        embedder=embedder,
        llm=llm,
        reranker=OverlapReranker(),
        settings=Settings(chunk_words=200, chunk_overlap=40, final_k=6, max_per_doc=3),
    )


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))
