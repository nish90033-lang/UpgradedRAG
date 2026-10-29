"""Demo backend for frontend development and end-to-end tests.

    python -m smartrag.demo

Runs the real API, retrieval pipeline, trust filter, and access control on an in-memory store.
Two parts are stand-ins so it needs no models, database, or API key:
  * the embedder is a hashing test double (no semantics), and
  * the "LLM" is extractive: it returns the best-matching source sentence with a citation.
Answers from this mode say nothing about SmartRAG's quality. Do not use it for evaluation.
"""

from __future__ import annotations

import os
import re

from .api import create_app
from .config import Settings
from .llm import ABSTAIN_TEXT
from .retrieval import HashingEmbedder, OverlapReranker
from .retrieval.tokenize import tokenize
from .rewrite import QueryRewriter
from .service import SmartRAGService
from .storage import MemoryStore

_SOURCE = re.compile(r'<source id="(\d+)">\n(.*?)\n</source>', re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveLLM:
    """Returns the source sentence that shares the most terms with the question."""

    name = "extractive-demo"

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        if "standalone search query" in system:  # demo rewriter: prepend the previous question
            prior = re.findall(r"User: (.*)", user)
            latest = user.rsplit("Latest message:", 1)[-1].split("\n")[0].strip()
            return f"{prior[-1]} {latest}" if prior else latest
        question = user.rsplit("Question:", 1)[-1].strip()
        q_terms = set(tokenize(question))
        best: tuple[int, str, str] = (0, "", "")
        for sid, body in _SOURCE.findall(user):
            for sentence in _SENTENCE.split(body.replace("&lt;", "<")):
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap > best[0]:
                    best = (overlap, sid, sentence.strip())
        if best[0] == 0:
            return ABSTAIN_TEXT
        return f"{best[2]} [{best[1]}]"


def build_demo_service() -> SmartRAGService:
    embedder = HashingEmbedder(dim=256)
    llm = ExtractiveLLM()
    return SmartRAGService(
        store=MemoryStore(embedder.dim),
        embedder=embedder,
        llm=llm,
        rewriter=QueryRewriter(llm),
        reranker=OverlapReranker(),
        settings=Settings(),
    )


def create_demo_app():
    os.environ.setdefault(
        "SMARTRAG_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )
    return create_app(build_demo_service())


app = create_demo_app()

if __name__ == "__main__":
    import uvicorn

    print("DEMO MODE: hashing embedder and extractive answers. Not for evaluation.")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
