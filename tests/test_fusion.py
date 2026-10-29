import pytest

from smartrag.retrieval.fusion import minmax_weighted, rrf


def test_rrf_scores_match_the_formula():
    out = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["c"] == pytest.approx(1 / 63)
    assert out["d"] == pytest.approx(1 / 63)


def test_rrf_tie_break_is_first_seen_order():
    ranked = [d for d, _ in rrf([["a", "b"], ["b", "a"]])]
    assert ranked == ["a", "b"]


def test_item_ranked_first_by_both_lists_wins():
    ranked = [d for d, _ in rrf([["x", "a", "b"], ["x", "c", "d"]])]
    assert ranked[0] == "x"


def test_weights_shift_the_outcome():
    r1, r2 = ["a", "b"], ["b", "a"]
    assert rrf([r1, r2], weights=[1, 3])[0][0] == "b"
    assert rrf([r1, r2], weights=[3, 1])[0][0] == "a"


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_minmax_weighted_normalises_each_list():
    out = dict(minmax_weighted([{"a": 10.0, "b": 0.0}, {"a": 0.2, "b": 0.4}], [1.0, 1.0]))
    assert out["a"] == pytest.approx(1.0 + 0.0)
    assert out["b"] == pytest.approx(0.0 + 1.0)
