import pytest
from fastapi.testclient import TestClient

from smartrag.api import create_app
from smartrag.config import Settings
from smartrag.llm import FakeLLM
from smartrag.retrieval import HashingEmbedder, OverlapReranker
from smartrag.rewrite import QueryRewriter
from smartrag.service import SmartRAGService
from smartrag.storage import MemoryStore

BANKS = (
    "Central banks set the policy rate. The cost of a mortgage rises when the policy rate "
    "rises, so the monthly cost of borrowing increases for households. "
) * 6
PLANTS = (
    "Photosynthesis converts sunlight water and carbon dioxide into glucose and oxygen. "
    "Chlorophyll absorbs red and blue light. "
) * 6
HISTORY = [
    {"question": "How do interest rates work?", "answer": "Central banks set the policy rate."}
]


def make(rewrite_reply="mortgage cost policy rate"):
    def reply(system, user):
        if "standalone search query" in system:
            return rewrite_reply
        return "The cost rises with the rate [1]."

    llm = FakeLLM(reply)
    emb = HashingEmbedder(64)
    svc = SmartRAGService(
        MemoryStore(64), emb, llm, OverlapReranker(), Settings(), rewriter=QueryRewriter(llm)
    )
    user, _ = svc.signup("a@x.y", "password123")
    svc.ingest_text(user.id, "banks", BANKS)
    svc.ingest_text(user.id, "plants", PLANTS)
    return svc, user, llm


def test_follow_up_is_searched_and_answered_with_the_rewritten_question():
    svc, user, llm = make()
    out = svc.query(user.id, "And what about the cost?", history=HISTORY)
    assert out.rewritten_query == "mortgage cost policy rate"
    assert out.sources[0]["doc_name"] == "banks"
    answer_prompt = llm.calls[-1][1]
    assert "Question: mortgage cost policy rate" in answer_prompt
    assert "rewrite_ms" in out.timings_ms


def test_history_stores_the_users_own_wording():
    svc, user, _ = make()
    svc.query(user.id, "And what about the cost?", history=HISTORY)
    assert svc.store.list_history(user.id)[0].question == "And what about the cost?"


def test_first_question_skips_the_rewriter():
    svc, user, llm = make()
    out = svc.query(user.id, "How does the policy rate affect a mortgage?")
    assert out.rewritten_query is None and "rewrite_ms" not in out.timings_ms
    assert not any("standalone search query" in system for system, _ in llm.calls)


def test_rewrite_failure_falls_back_and_still_answers():
    svc, user, _ = make(rewrite_reply="")
    out = svc.query(user.id, "monthly cost of borrowing mortgage", history=HISTORY)
    assert out.rewritten_query is None and out.answerable


def test_service_without_a_rewriter_ignores_history():
    svc, user, llm = make()
    svc.rewriter = None
    out = svc.query(user.id, "mortgage cost policy rate", history=HISTORY)
    assert out.rewritten_query is None and out.answerable


def test_api_accepts_history_and_returns_the_rewritten_query():
    svc, user, _ = make()
    with TestClient(create_app(svc)) as c:
        token = c.post("/auth/login", json={"email": "a@x.y", "password": "password123"}).json()[
            "token"
        ]
        h = {"Authorization": f"Bearer {token}"}
        r = c.post("/query", headers=h, json={"question": "And the cost?", "history": HISTORY})
        assert r.status_code == 200 and r.json()["rewritten_query"] == "mortgage cost policy rate"
        none = c.post(
            "/query", headers=h, json={"question": "How does the policy rate affect a mortgage?"}
        )
        assert none.json()["rewritten_query"] is None


@pytest.mark.parametrize(
    "history",
    [
        [{"question": "q", "answer": "a"}] * 7,
        [{"question": "q" * 2001, "answer": "a"}],
        [{"answer": "missing question"}],
    ],
)
def test_api_rejects_oversized_or_malformed_history(history):
    svc, _, _ = make()
    with TestClient(create_app(svc)) as c:
        token = c.post("/auth/login", json={"email": "a@x.y", "password": "password123"}).json()[
            "token"
        ]
        r = c.post(
            "/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "q", "history": history},
        )
        assert r.status_code == 422
