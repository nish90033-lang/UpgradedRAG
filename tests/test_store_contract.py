"""One contract, run against both the memory store and PostgreSQL with pgvector."""

import numpy as np
import pytest

from smartrag.retrieval import ArrayDenseIndex
from smartrag.storage import ChunkRow, DuplicateError


def unit(seed: int, dim: int = 8) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def add_doc(store, owner, name, texts, trust=100.0, seed0=0, content_hash=None):
    doc = store.create_document(owner, name, content_hash or f"h-{name}", trust, [], "text")
    rows = [ChunkRow(i, t, f"ch-{name}-{i}", unit(seed0 + i)) for i, t in enumerate(texts)]
    store.replace_chunks(doc.id, trust, rows)
    return doc, rows


def test_users_unique_email_and_lookup(store8):
    u = store8.create_user("A@Example.com", "hash")
    assert u.email == "a@example.com"
    with pytest.raises(DuplicateError):
        store8.create_user("a@example.com", "hash2")
    found, pw = store8.get_user_with_hash("A@example.COM")
    assert found.id == u.id and pw == "hash"
    assert store8.get_user_with_hash("nobody@x.y") is None


def test_document_uniqueness_by_name_and_by_content(store8):
    u = store8.create_user("a@x.y", "h")
    store8.create_document(u.id, "doc", "hash1", 100, [], "text")
    with pytest.raises(DuplicateError):
        store8.create_document(u.id, "doc", "hash2", 100, [], "text")
    with pytest.raises(DuplicateError):
        store8.create_document(u.id, "other", "hash1", 100, [], "text")
    other = store8.create_user("b@x.y", "h")
    store8.create_document(other.id, "doc", "hash1", 100, [], "text")  # allowed for another user


def test_pool_returns_chunks_in_order_with_metadata(store8):
    u = store8.create_user("a@x.y", "h")
    doc, _ = add_doc(store8, u.id, "notes", ["one", "two", "three"])
    pool = store8.pool(u.id)
    assert [c.text for c in pool] == ["one", "two", "three"]
    assert {c.doc_name for c in pool} == {"notes"} and {c.doc_id for c in pool} == {doc.id}
    assert all(c.trust == 100.0 for c in pool)
    assert store8.list_documents(u.id)[0].chunk_count == 3


def test_users_cannot_see_each_others_chunks_until_shared(store8):
    a = store8.create_user("a@x.y", "h")
    b = store8.create_user("b@x.y", "h")
    doc, _ = add_doc(store8, a.id, "private", ["secret plan"])
    assert store8.pool(b.id) == [] and store8.dense_search(b.id, unit(0), 5) == []
    assert store8.list_documents(b.id) == []
    assert store8.share_document(a.id, doc.id, b.id)
    assert [c.text for c in store8.pool(b.id)] == ["secret plan"]
    assert store8.dense_search(b.id, unit(0), 5)
    listed = store8.list_documents(b.id)
    assert listed[0].shared is True and store8.list_documents(a.id)[0].shared is False


def test_sharing_rules(store8):
    a = store8.create_user("a@x.y", "h")
    b = store8.create_user("b@x.y", "h")
    doc, _ = add_doc(store8, a.id, "d", ["t"])
    assert not store8.share_document(b.id, doc.id, a.id)  # only the owner may share
    assert not store8.share_document(a.id, doc.id, a.id)  # not with yourself
    assert not store8.share_document(a.id, doc.id, "00000000-0000-0000-0000-000000000000")


def test_delete_is_owner_only_and_removes_chunks_and_shares(store8):
    a = store8.create_user("a@x.y", "h")
    b = store8.create_user("b@x.y", "h")
    doc, _ = add_doc(store8, a.id, "d", ["t"])
    store8.share_document(a.id, doc.id, b.id)
    assert not store8.delete_document(b.id, doc.id)
    assert store8.delete_document(a.id, doc.id)
    assert store8.pool(a.id) == [] and store8.pool(b.id) == []
    assert store8.chunk_embeddings(doc.id) == {}


def test_doc_ids_and_min_trust_filters(store8):
    u = store8.create_user("a@x.y", "h")
    clean, _ = add_doc(store8, u.id, "clean", ["c1", "c2"], trust=100.0, seed0=0)
    bad, _ = add_doc(store8, u.id, "bad", ["b1"], trust=40.0, seed0=10)
    assert {c.text for c in store8.pool(u.id)} == {"c1", "c2", "b1"}
    assert {c.text for c in store8.pool(u.id, [clean.id])} == {"c1", "c2"}
    assert {c.text for c in store8.pool(u.id, min_trust=50)} == {"c1", "c2"}
    assert store8.pool(u.id, [bad.id], min_trust=50) == []
    pool_ids = {c.chunk_id: c.text for c in store8.pool(u.id)}
    hits = store8.dense_search(u.id, unit(10), 10, min_trust=50)
    assert {pool_ids[i] for i, _ in hits} == {"c1", "c2"}


def test_dense_search_matches_brute_force_cosine(store8):
    u = store8.create_user("a@x.y", "h")
    _, rows = add_doc(store8, u.id, "d", [f"t{i}" for i in range(20)], seed0=100)
    q = unit(103) * 0.9 + unit(7) * 0.1
    q /= np.linalg.norm(q)
    pool = store8.pool(u.id)
    expected = ArrayDenseIndex([c.chunk_id for c in pool], np.vstack([r.embedding for r in rows]))(
        q, 6
    )
    got = store8.dense_search(u.id, q, 6)
    assert [i for i, _ in got] == [i for i, _ in expected]
    assert np.allclose([s for _, s in got], [s for _, s in expected], atol=1e-4)


def test_replace_chunks_replaces_and_embedding_roundtrip(store8):
    u = store8.create_user("a@x.y", "h")
    doc, rows = add_doc(store8, u.id, "d", ["x", "y"])
    cached = store8.chunk_embeddings(doc.id)
    assert set(cached) == {"ch-d-0", "ch-d-1"}
    assert np.allclose(cached["ch-d-0"], rows[0].embedding, atol=1e-6)
    store8.replace_chunks(doc.id, 100.0, [ChunkRow(0, "z", "new", unit(5))])
    assert [c.text for c in store8.pool(u.id)] == ["z"]


def test_wrong_embedding_dimension_is_rejected(store8):
    u = store8.create_user("a@x.y", "h")
    doc = store8.create_document(u.id, "d", "h", 100, [], "text")
    with pytest.raises(ValueError):
        store8.replace_chunks(doc.id, 100, [ChunkRow(0, "t", "h", np.ones(3, dtype=np.float32))])


def test_update_document_changes_hash_trust_and_patterns(store8):
    u = store8.create_user("a@x.y", "h")
    doc = store8.create_document(u.id, "d", "h1", 100, [], "text")
    store8.update_document(doc.id, "h2", 40.0, ["role_switch"])
    got = store8.find_document_by_name(u.id, "d")
    assert got.content_hash == "h2" and got.trust_score == 40.0 and got.patterns == ["role_switch"]
    assert store8.find_document_by_hash(u.id, "h2").id == doc.id


def test_history_roundtrip_newest_first(store8):
    u = store8.create_user("a@x.y", "h")
    store8.add_history(u.id, "q1", "a1", True, [{"n": 1, "doc_name": "d"}])
    store8.add_history(u.id, "q2", "a2", False, [])
    items = store8.list_history(u.id)
    assert [i.question for i in items] == ["q2", "q1"]
    assert items[1].sources == [{"n": 1, "doc_name": "d"}] and items[0].answerable is False
    assert store8.list_history(u.id, limit=1)[0].question == "q2"
