"""Glue-code tests for the model adapters, using stand-ins for the network and the models.

These prove prefixes, batching, ordering, normalisation, and request shape. They do not prove
that a real model loads. That needs the `ml` extra and network access.
"""

import json
import sys
import types

import httpx
import numpy as np
import pytest

from smartrag.llm import GeminiLLM, GroqLLM
from smartrag.retrieval import OpenAICompatEmbedder, SentenceTransformerEmbedder
from smartrag.retrieval.rerank import CrossEncoderReranker


class FakeST:
    instances: list = []

    def __init__(self, name, device=None):
        self.name, self.encoded = name, []
        FakeST.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        self.encoded.append((list(texts), batch_size, normalize_embeddings))
        return np.tile(np.array([3.0, 4.0, 0.0, 0.0]), (len(texts), 1)) / 5.0


class FakeCE:
    def __init__(self, name, max_length=512, device=None):
        self.name, self.max_length, self.pairs = name, max_length, []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs.append(list(pairs))
        return np.array([float(len(p[1])) for p in pairs])


@pytest.fixture
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer, mod.CrossEncoder = FakeST, FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)


def test_sentence_transformer_adapter_prefixes_queries_but_not_documents(fake_st):
    emb = SentenceTransformerEmbedder("some/model", query_prefix="Q: ", doc_prefix="")
    assert emb.dim == 4 and emb.name == "some/model"
    emb.embed_queries(["what is x"])
    emb.embed_documents(["x is y"])
    (q_texts, _, q_norm), (d_texts, _, d_norm) = emb.model.encoded
    assert q_texts == ["Q: what is x"] and d_texts == ["x is y"]
    assert q_norm and d_norm


def test_cross_encoder_adapter_scores_pairs_and_handles_empty_input(fake_st):
    ce = CrossEncoderReranker("some/ce")
    assert ce.score("q", []) == []
    assert ce.score("q", ["aa", "bbbb"]) == [2.0, 4.0]
    assert ce.model.pairs[0] == [("q", "aa"), ("q", "bbbb")]


def test_openai_compat_embedder_batches_reorders_and_normalises():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(len(body["input"]))
        assert request.headers["authorization"] == "Bearer key"
        data = [
            {"index": i, "embedding": [float(i + 1), 0.0, 0.0]} for i in range(len(body["input"]))
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})  # out of order on purpose

    emb = OpenAICompatEmbedder("https://example.test/v1", "key", "m", dim=3)
    emb._client = httpx.Client(
        base_url="https://example.test/v1",
        headers={"Authorization": "Bearer key"},
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed_documents([f"t{i}" for i in range(100)])
    assert seen == [96, 4] and out.shape == (100, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[:, 0], 1.0)  # all vectors point along the first axis after normalising


def test_groq_client_sends_the_expected_request_and_parses_the_reply():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "  hi [1]  "}}]})

    llm = GroqLLM(api_key="k", model="some-model", temperature=0.0)
    llm._client = httpx.Client(
        base_url="https://api.groq.com/openai/v1", transport=httpx.MockTransport(handler)
    )
    assert llm.complete("sys", "usr", max_tokens=77) == "hi [1]"
    body = captured["body"]
    assert captured["url"].endswith("/chat/completions")
    assert body["model"] == "some-model" and body["max_tokens"] == 77 and body["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_groq_client_requires_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GroqLLM()


def test_gemini_client_sends_generate_content_request_and_parses_reply():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "  hi "}, {"text": "[1]  "}]}}]},
        )

    llm = GeminiLLM(api_key="test-key", model="gemini-test", temperature=0.0)
    assert llm.timeout == 180.0
    assert llm._client.timeout.read == 180.0
    llm._client = httpx.Client(
        base_url="https://generativelanguage.googleapis.com",
        headers={"x-goog-api-key": "test-key"},
        transport=httpx.MockTransport(handler),
    )

    assert llm.complete("system", "user", max_tokens=77) == "hi [1]"
    assert captured["url"].endswith("/v1beta/models/gemini-test:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["body"] == {
        "system_instruction": {"parts": [{"text": "system"}]},
        "contents": [{"role": "user", "parts": [{"text": "user"}]}],
        "generationConfig": {"maxOutputTokens": 77, "temperature": 0.0},
    }


def test_gemini_client_requires_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLM()
