import os

from smartrag.dotenv import load_dotenv, parse_dotenv


def test_parse_handles_quotes_comments_export_and_ampersands():
    text = (
        "# comment\n\nDATABASE_URL=postgresql://u:p@h/db?sslmode=require&channel_binding=require\n"
        "export GROQ_API_KEY='abc'\nNAME=\"two words\"\nEMPTY=\nnot a line\n"
    )
    got = parse_dotenv(text)
    assert got["DATABASE_URL"].endswith("sslmode=require&channel_binding=require")
    assert got["GROQ_API_KEY"] == "abc" and got["NAME"] == "two words" and got["EMPTY"] == ""
    assert "not a line" not in got


def test_load_sets_missing_keys_but_never_overrides_the_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("\ufeffA_KEY=from_file\nB_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("B_KEY", "from_env")
    monkeypatch.delenv("A_KEY", raising=False)
    assert load_dotenv(f) == ["A_KEY"]
    assert os.environ["A_KEY"] == "from_file" and os.environ["B_KEY"] == "from_env"
    monkeypatch.delenv("A_KEY")


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []
