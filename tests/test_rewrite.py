import pytest

from smartrag.llm import FakeLLM
from smartrag.rewrite import QueryRewriter, build_rewrite_prompt, clean_rewrite

HISTORY = [
    {
        "question": "Tell me about chlorophyll",
        "answer": "Chlorophyll is a pigment in plants. " * 30,
    },
    {"question": "Where is it found?", "answer": "In chloroplasts."},
]


def test_clean_rewrite_strips_labels_quotes_and_extra_lines():
    assert clean_rewrite("Query: what does chlorophyll absorb") == "what does chlorophyll absorb"
    assert clean_rewrite('"chlorophyll absorption"') == "chlorophyll absorption"
    assert clean_rewrite("standalone search query: x y\nHere is why I chose that") == "x y"
    assert clean_rewrite("   ") is None
    assert clean_rewrite("") is None
    assert clean_rewrite("x" * 301) is None
    assert clean_rewrite("x" * 300) == "x" * 300


def test_prompt_keeps_only_recent_turns_and_truncates_long_answers():
    turns = [{"question": f"q{i}", "answer": "a" * 1000} for i in range(6)]
    system, user = build_rewrite_prompt("latest?", turns, max_turns=3, max_answer_chars=50)
    assert "q5" in user and "q3" in user and "q2" not in user
    assert "a" * 51 not in user and "Latest message: latest?" in user
    assert "Do not answer" in system


def test_no_history_means_no_model_call():
    llm = FakeLLM("should not be used")
    result = QueryRewriter(llm).rewrite("What is chlorophyll?", [])
    assert result.query == "What is chlorophyll?" and not result.rewritten and not result.failed
    assert llm.calls == []
    assert QueryRewriter(llm).rewrite("x", None).query == "x" and llm.calls == []


def test_follow_up_is_rewritten_using_history():
    llm = FakeLLM("Query: where is chlorophyll found in a plant cell")
    result = QueryRewriter(llm).rewrite("Where is it?", HISTORY)
    assert result.rewritten and not result.failed
    assert result.query == "where is chlorophyll found in a plant cell"
    assert "Tell me about chlorophyll" in llm.calls[0][1]


def test_unchanged_output_is_not_reported_as_rewritten():
    result = QueryRewriter(FakeLLM("what is chlorophyll")).rewrite("What is chlorophyll", HISTORY)
    assert not result.rewritten and not result.failed


@pytest.mark.parametrize("bad", ["", "   \n", "x" * 400])
def test_unusable_output_falls_back_to_the_original(bad):
    result = QueryRewriter(FakeLLM(bad)).rewrite("Where is it?", HISTORY)
    assert result.query == "Where is it?" and result.failed and not result.rewritten


def test_model_errors_fall_back_to_the_original():
    def boom(system, user):
        raise TimeoutError("slow")

    result = QueryRewriter(FakeLLM(boom)).rewrite("Where is it?", HISTORY)
    assert result.query == "Where is it?" and result.failed
