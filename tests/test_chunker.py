import pytest

from smartrag.ingest import chunk_words, content_hash


def test_window_boundaries_and_final_chunk_reaches_last_word():
    text = " ".join(f"w{i}" for i in range(450))
    chunks = chunk_words(text, size=200, overlap=40)
    assert [c.idx for c in chunks] == [0, 1, 2]
    assert chunks[0].text.split()[0] == "w0" and chunks[0].text.split()[-1] == "w199"
    assert chunks[1].text.split()[0] == "w160"
    assert chunks[2].text.split()[-1] == "w449"


def test_overlap_is_shared_between_neighbours():
    text = " ".join(f"w{i}" for i in range(300))
    a, b = chunk_words(text, size=200, overlap=40)[:2]
    assert a.text.split()[-40:] == b.text.split()[:40]


def test_short_text_single_chunk():
    chunks = chunk_words("just a few words", size=200, overlap=40)
    assert len(chunks) == 1 and chunks[0].text == "just a few words"


def test_empty_text_yields_no_chunks():
    assert chunk_words("   \n ") == []


def test_no_trailing_chunk_is_a_subset_of_its_predecessor():
    text = " ".join(f"w{i}" for i in range(360))  # exactly two windows reach the end
    chunks = chunk_words(text, size=200, overlap=40)
    assert len(chunks) == 2
    assert chunks[-1].text.split()[-1] == "w359"


def test_max_chunks_cap():
    text = " ".join(f"w{i}" for i in range(5000))
    assert len(chunk_words(text, size=100, overlap=10, max_chunks=7)) == 7


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_words("a b c", size=size, overlap=overlap)


def test_hash_ignores_whitespace_but_not_content():
    assert content_hash("a  b\n c") == content_hash("a b c")
    assert content_hash("a b c") != content_hash("a b d")
