from __future__ import annotations

import pymupdf


def normalize_text(raw: str) -> str:
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract(doc: pymupdf.Document) -> str:
    pages: list[str] = []
    try:
        for page in doc:
            pages.append(page.get_text())
    finally:
        doc.close()
    return "\n".join(pages)


def extract_pdf_bytes(data: bytes) -> str:
    """Extract text page by page so only one page object is alive at a time."""
    return _extract(pymupdf.open(stream=data, filetype="pdf"))


def extract_pdf_path(path: str) -> str:
    return _extract(pymupdf.open(path))
