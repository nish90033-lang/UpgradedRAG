import io
import ssl
import urllib.error
import zipfile

import pytest

from scripts import download_beir


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", "{}\n")
    return buffer.getvalue()


def test_download_uses_verified_tls_and_extracts_dataset(tmp_path, monkeypatch):
    seen = {}

    def open_url(url, *, context):
        seen["url"] = url
        seen["context"] = context
        return io.BytesIO(_archive())

    monkeypatch.setattr(download_beir.urllib.request, "urlopen", open_url)
    download_beir.main("scifact", str(tmp_path))

    assert seen["url"] == download_beir.BASE.format(name="scifact")
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED
    assert seen["context"].check_hostname
    assert (tmp_path / "scifact" / "corpus.jsonl").read_text() == "{}\n"
    assert not (tmp_path / "scifact.zip.part").exists()


def test_certificate_failure_explains_fix_and_removes_partial_download(tmp_path, monkeypatch):
    def fail_download(url, *, context):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("unknown issuer"))

    monkeypatch.setattr(download_beir.urllib.request, "urlopen", fail_download)
    (tmp_path / "scifact.zip.part").write_bytes(b"incomplete")

    with pytest.raises(RuntimeError, match="SSL_CERT_FILE"):
        download_beir.main("scifact", str(tmp_path))

    assert not (tmp_path / "scifact.zip.part").exists()
    assert not (tmp_path / "scifact.zip").exists()
