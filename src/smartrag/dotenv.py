"""Minimal .env loader. Real environment variables always win over the file."""

from __future__ import annotations

import os
from pathlib import Path


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_dotenv(*paths: str | Path) -> list[str]:
    """Load the first existing file among paths. Returns the keys that were set."""
    candidates = paths or (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env")
    for path in candidates:
        p = Path(path)
        if p.is_file():
            loaded = []
            for key, value in parse_dotenv(p.read_text(encoding="utf-8-sig")).items():
                if key not in os.environ:
                    os.environ[key] = value
                    loaded.append(key)
            return loaded
    return []
