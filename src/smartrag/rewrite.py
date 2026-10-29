"""Query rewriting for follow-up questions.

A follow-up such as "what about its cost?" cannot be searched on its own. The rewriter uses the
recent conversation to turn it into a standalone search query. Design rules:

* It runs only when there is conversation history. A first question is searched as written,
  so a single-turn query never pays for an extra model call.
* The output is validated (one line, bounded length). If the model fails, times out, or returns
  something unusable, the original question is searched instead and the result says so.
* The rewritten text is used for retrieval and for the answer prompt. The user's own wording is
  what gets stored in chat history.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .llm import LLM

REWRITE_SYSTEM = (
    "You rewrite the user's latest message as one standalone search query for a document search "
    "engine. Resolve pronouns and references such as 'it', 'that', or 'the second one' using the "
    "conversation. Keep names, numbers, and technical terms exactly. Do not answer the "
    "question. Reply with the query only, on one line, without quotes."
)

_PREFIX = re.compile(r"^(?:standalone\s+)?(?:search\s+)?query\s*:\s*", re.IGNORECASE)


@dataclass(frozen=True)
class RewriteResult:
    query: str
    rewritten: bool  # the query differs from the user's wording
    failed: bool = False  # the model call or its output was unusable


def build_rewrite_prompt(
    question: str,
    history: Sequence[dict[str, Any]],
    max_turns: int = 3,
    max_answer_chars: int = 300,
) -> tuple[str, str]:
    lines = []
    for turn in list(history)[-max_turns:]:
        answer = str(turn.get("answer", ""))[:max_answer_chars]
        lines.append(f"User: {turn.get('question', '')}\nAssistant: {answer}")
    user = (
        "Conversation so far:\n"
        + "\n".join(lines)
        + f"\n\nLatest message: {question}\n\nStandalone search query:"
    )
    return REWRITE_SYSTEM, user


def clean_rewrite(raw: str, max_chars: int = 300) -> str | None:
    """First non-empty line, without a 'Query:' label or quotes. None if unusable."""
    stripped = raw.strip()
    if not stripped:
        return None
    text = _PREFIX.sub("", stripped.splitlines()[0].strip()).strip().strip("\"'` ")
    return text if 0 < len(text) <= max_chars else None


class QueryRewriter:
    def __init__(self, llm: LLM, max_turns: int = 3, max_chars: int = 300) -> None:
        self.llm, self.max_turns, self.max_chars = llm, max_turns, max_chars

    def rewrite(self, question: str, history: Sequence[dict[str, Any]] | None) -> RewriteResult:
        if not history:
            return RewriteResult(question, rewritten=False)
        system, user = build_rewrite_prompt(question, history, self.max_turns)
        try:
            raw = self.llm.complete(system, user, 80)
        except Exception:  # network errors, rate limits, anything: fall back to the original
            return RewriteResult(question, rewritten=False, failed=True)
        cleaned = clean_rewrite(raw, self.max_chars)
        if cleaned is None:
            return RewriteResult(question, rewritten=False, failed=True)
        return RewriteResult(cleaned, rewritten=cleaned.lower() != question.lower())
