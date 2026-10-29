import sys
import types

import pytest

from smartrag.config import Settings
from smartrag.retrieval import reranker_from_setting


@pytest.mark.parametrize("value", [None, "", "none", "None", " OFF ", "false", "0", "disabled"])
def test_off_values_disable_the_reranker_without_importing_any_model_library(value, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # an import would now fail
    assert reranker_from_setting(value) is None


def test_a_model_name_builds_a_cross_encoder(monkeypatch):
    seen = {}

    class FakeCE:
        def __init__(self, name, max_length=512, device=None):
            seen["name"] = name

    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    rr = reranker_from_setting("  some/cross-encoder ")
    assert rr is not None and seen["name"] == "some/cross-encoder"


def test_reranking_is_off_by_default_and_configurable_from_the_environment(monkeypatch):
    monkeypatch.delenv("SMARTRAG_RERANK_MODEL", raising=False)
    assert Settings.from_env().rerank_model == "none"
    monkeypatch.setenv("SMARTRAG_RERANK_MODEL", "x/y")
    assert Settings.from_env().rerank_model == "x/y"
