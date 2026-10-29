"""Layer 1 and Layer 2 of the anti-poisoning pipeline.

Layer 1 detects known prompt-injection phrasings with regular expressions and strips them.
Layer 2 turns the detections into a trust score. Any detection caps trust at 40, so a
five-word injection inside a fifty-thousand-word document is not diluted by document length.

This is a pattern matcher. It does not catch paraphrased, encoded, or non-English attacks.
Run `python -m evals.injection_eval` to measure what it misses.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_I = re.IGNORECASE

# name -> (regex, flags). DAN is case-sensitive so the given name "Dan" does not match.
INJECTION_PATTERNS: dict[str, tuple[str, int]] = {
    "ignore_previous": (
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+)?"
        r"(?:previous|prior|above|earlier|preceding)\s+"
        r"(?:instructions?|prompts?|rules?|context|messages?|directions?)",
        _I,
    ),
    "forget_everything": (
        r"\bforget\s+(?:everything|all\s+(?:your\s+)?(?:rules|instructions|training))",
        _I,
    ),
    "ignore_all": (r"\bignore\s+(?:all|everything)\s+(?:above|before|else|you\s+were)", _I),
    "role_switch": (r"\byou\s+are\s+now\s+(?:a|an|the|in|my)\b", _I),
    "unrestricted_persona": (
        r"\b(?:act|behave|respond|pretend)\s+(?:as|like)\s+(?:if\s+you\s+(?:are|were)\s+)?"
        r"(?:a|an)\s+(?:unrestricted|unfiltered|jailbroken|uncensored)",
        _I,
    ),
    "new_persona": (r"\bnew\s+persona\b", _I),
    "developer_mode": (r"\bdeveloper\s+mode\b", _I),
    "dan_jailbreak": (r"\bDAN\b|\bdo\s+anything\s+now\b", 0),
    "prompt_extraction": (
        r"\b(?:reveal|show|print|repeat|output|leak|disclose|display)\s+(?:me\s+)?"
        r"(?:your|the)\s+(?:system\s+prompt|hidden\s+(?:prompt|instructions)|"
        r"initial\s+instructions|original\s+instructions)",
        _I,
    ),
    "prompt_question": (
        r"\bwhat\s+(?:is|are|were)\s+your\s+(?:system\s+prompt|initial\s+instructions)\b",
        _I,
    ),
    "role_tag": (r"(?m)^\s*(?:system|assistant)\s*:\s*\S", _I),
    "chat_template_token": (r"<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?INST\]", _I),
    "note_to_llm": (
        r"\bnote\s+(?:for|to)\s+(?:the\s+)?(?:ai|llm|language\s+models?|assistant|model|chatbot)s?\b",
        _I,
    ),
    "instructions_for_ai": (
        r"\b(?:attention|instructions?\s+for)\s*:?\s*(?:the\s+)?(?:ai|llm|assistant|model)\b",
        _I,
    ),
    "secret_request": (
        r"\b(?:provide|give|send|reveal|output|print|share)\s+(?:me\s+)?(?:the\s+|your\s+)?"
        r"(?:\w+\s+){0,2}(?:api\s*key|password|secret|credentials)s?\b",
        _I,
    ),
    "exfiltrate_url": (
        r"\b(?:send|post|append|forward|exfiltrate|upload)\s+[^.\n]{0,60}\s+to\s+https?://",
        _I,
    ),
    "jailbreak_word": (
        r"\b(?:jailbreak|jailbreaked)\s+(?:the\s+)?"
        r"(?:model|assistant|llm|chatbot|system)\b|\b(?:use|try)\s+"
        r"(?:this\s+|a\s+)?jailbreak\b",
        _I,
    ),
}
_COMPILED = {name: re.compile(rx, flags) for name, (rx, flags) in INJECTION_PATTERNS.items()}

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)

TRUST_CAP = 40.0


@dataclass(frozen=True)
class SanitizeResult:
    clean_text: str
    matched: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.matched)


def _detection_view(text: str) -> str:
    """Normalised copy used for matching: NFKC plus zero-width removal."""
    return unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)


def _is_quoted_example(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 100) : start]
    cue = re.search(
        r"\b(?:instructions?|prompts?|attacks?|examples?)\s+"
        r"(?:such\s+as|like|including)\s*[,;:]?\s*(['\"“‘])\s*$",
        prefix,
        _I,
    )
    if cue is None:
        return False
    closing_quote = {"“": "”", "‘": "’"}.get(cue.group(1), cue.group(1))
    closing = text.find(closing_quote, end, min(len(text), end + 300))
    return closing >= 0


def detect(text: str) -> list[str]:
    view = _detection_view(text)
    return [name for name, rx in _COMPILED.items() if rx.search(view)]


def sanitize(text: str) -> SanitizeResult:
    """Strip detected injections.

    When nothing matches the original text is kept untouched apart from whitespace
    collapsing, so legitimate characters are never rewritten by NFKC.
    """
    view = _detection_view(text)
    matched: list[str] = []
    stripped = False
    for name, rx in _COMPILED.items():
        matches = list(rx.finditer(view))
        examples_only = (
            name == "ignore_previous"
            and bool(matches)
            and all(_is_quoted_example(view, m.start(), m.end()) for m in matches)
        )
        view, n = rx.subn(" ", view)
        stripped = stripped or bool(n)
        if n and not examples_only:
            matched.append(name)
    if not matched and not stripped:
        return SanitizeResult(clean_text=re.sub(r"\s+", " ", text).strip(), matched=[])
    return SanitizeResult(clean_text=re.sub(r"\s+", " ", view).strip(), matched=matched)


def trust_score(matched: list[str]) -> float:
    """100 for a clean document. Any detection caps the score at 40, below the 50 threshold."""
    if not matched:
        return 100.0
    return float(max(0.0, min(TRUST_CAP, 100.0 - 20.0 * len(matched))))
