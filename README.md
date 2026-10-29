# SmartRAG v2

A document question answering system with hybrid retrieval (BM25 and dense embeddings fused by
reciprocal rank fusion), cross-encoder reranking, a prompt-injection trust filter, per-user
access control, incremental re-indexing, and an evaluation suite that anyone can rerun.

SmartRAG v1 was a lexical-only system (hashed term frequency plus BM25) evaluated on a query set
written by its author. v2 keeps the safety design and replaces the retrieval and evaluation
foundations. The v1 scorer is kept in `smartrag.retrieval.lexical_v1` as the baseline row in every
ablation, and a test proves it matches the original v1 function.

## Quick start

```bash
pip install -e ".[dev,ml]"
docker compose up -d db                      # PostgreSQL 16 with pgvector
export DATABASE_URL=postgresql://smartrag:smartrag@localhost:5432/smartrag
export GROQ_API_KEY=...                      # any OpenAI-compatible LLM can be swapped in
uvicorn smartrag.api:create_app --factory --port 8000
```

Tests (no models, no network, no API keys needed):

```bash
pip install -e ".[dev]"
TEST_DATABASE_URL=postgresql://smartrag:smartrag@localhost:5432/smartrag_test python -m pytest
```

Without a reachable PostgreSQL the database tests skip and everything else still runs.

## Frontend

`frontend/` is the v1 React app ported to the v2 API. It keeps the v1 look and adds what v2 made
possible: clickable citations that jump to their source, a visible "Blocked" state for documents
the safety filter rejected, delete and share per document, a note on how many chunks an edit
re-embedded, and sign-out when the server rejects an expired token.

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

The frontend uses `http://localhost:8000` by default. If the API is at another URL,
create `frontend/.env` and set `VITE_API_URL` there.

To try the interface without PostgreSQL, models, or API keys, run the demo backend. It uses the
real API, retrieval pipeline, trust filter, and access control, but a hashing embedder and an
extractive stand-in for the LLM, so its answers say nothing about quality:

```bash
python -m smartrag.demo     # http://127.0.0.1:8000, in-memory, resets on restart
```

Set `SMARTRAG_CORS_ORIGINS` (comma separated) when the frontend is served from another origin.

Moving from the v1 API:

| v1 | v2 |
|---|---|
| `POST /upload` | `POST /documents/pdf` (multipart `file`, optional `name`) |
| `POST /upload-text` `{text, doc_name}` | `POST /documents/text` `{name, text}` |
| documents have `doc_id`, `filename` | documents have `id`, `name`, `patterns`, `shared` |
| `POST /query` `{question, use_llm, doc_ids}` | `POST /query` `{question, doc_ids, history}`; sources carry `n`, `doc_name`, `score`, `trust`; the response adds `rewritten_query` |
| `POST /auth/reset` | removed (v2 has no password reset yet) |
| none | `DELETE /documents/{id}`, `POST /documents/{id}/share`, `GET /auth/public-key` |

Frontend tests: `npm test` runs 36 unit and component tests. `npm run test:e2e` drives the real
UI against a running `python -m smartrag.demo`.

## Architecture

```
upload -> extract -> sanitise (L1) -> trust score (L2) -> chunk -> embed (only new chunks)
query  -> rewrite (follow-ups only) -> access filter -> trust filter (L3) -> BM25 + dense -> RRF -> rerank -> LLM with citations
```

| Part | Where | Notes |
|---|---|---|
| Chunking | `ingest/chunker.py` | 200 word windows, 40 word overlap, content hash per chunk |
| Sanitiser and trust | `ingest/sanitize.py` | 17 regex patterns, NFKC and zero-width normalisation, any hit caps trust at 40 |
| BM25 | `retrieval/bm25.py` | Sparse matrix, verified equal to `rank_bm25` in okapi mode |
| Dense search | `storage/postgres.py` | pgvector, exact scan by default so access filters cannot reduce recall |
| Fusion | `retrieval/fusion.py` | Reciprocal rank fusion with tunable k and weights |
| Query rewriting | `rewrite.py` | Turns a follow-up into a standalone search query, only when there is conversation history |
| Reranker | `retrieval/rerank.py` | Cross-encoder on the top N fused candidates. Off by default: set `SMARTRAG_RERANK_MODEL` to a model name to enable it |
| Access control | `storage/*` | Owner access plus explicit shares, enforced inside SQL |
| Incremental sync | `service.py` | Unchanged chunks keep their stored embedding, only new text is embedded |
| LLM | `llm.py` | Numbered sources, mandatory citations, refusal text, sources treated as untrusted data |
| Auth | `auth.py` | ES256 JWT, bcrypt with a SHA-256 prehash |

## Reproducing the evaluation

Every number in a write-up should come from one of these commands.

**1. Retrieval ablation on a public benchmark** (needs network and the `ml` extra):

```bash
python scripts/download_beir.py scifact data
python -m evals.run_ablation --data data/scifact --out results
```

The downloader verifies HTTPS certificates using `certifi`. If you're behind a corporate proxy,
set `SSL_CERT_FILE` to your organization's CA bundle; do not disable certificate verification.

Protocol: queries split into dev and test with a fixed seed; the RRF constant and BM25 weight
are tuned on dev only; every system is scored on test only; differences get a paired bootstrap
95% interval; latency is measured per query after warm-up. Output goes to
`results/scifact/results.md` and `results.json` together with the library versions and git commit.
If a test-double embedder or reranker is used the output folder gets a `-smoke` suffix and the
report says so on its first line, so a plumbing check can never be mistaken for a result.

**2. Injection sanitiser**:

```bash
python -m evals.injection_eval --corpus data/scifact
```

Reports detection per attack category with Wilson intervals, false positives on benign text,
what was missed, and the false positive rate on a real corpus. The bundled cases were written
by the same assistant that wrote the sanitiser, so they are not an independent test. Ask other
people for cases, put their name in the `author` field, and pass them with `--extra`.

**3. Answer quality on your own documents**:

```bash
python -m evals.rag_eval --docs data/docs --goldset data/goldset.jsonl --llm gemini \
    --skip-judge
```

For Gemini, set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` in the root `.env`. The evaluator
uses `gemini-2.5-flash` if `GEMINI_MODEL` is not set. Omit `--skip-judge` to use the same Gemini
model for faithfulness judgments, or pass `--judge-model` to choose another Gemini model.
Gemini requests allow 180 seconds by default; use `--llm-timeout 300` to allow up to five minutes.
For a diagnostic run, `--rerank-top-n 8` changes how many fused candidates are cross-encoder
reranked without changing the application's default retrieval setting. Treat tuning on a small
goldset as exploratory; validate the setting on held-out questions before adopting it.

Create `data/docs` and put your own `.pdf`, `.txt`, or `.md` files in it; create
`data/goldset.jsonl` with questions and evidence phrases from those documents. The downloaded
BEIR dataset in `data/scifact` is for the retrieval ablation, not this command's document input.

Goldset lines look like this. Write the evidence phrases yourself, from the documents.

```json
{"id": "q01", "question": "...", "answerable": true, "evidence": ["exact phrase from a document"]}
{"id": "q51", "question": "An out-of-scope question", "answerable": false}
```

It reports retrieval hit rate and MRR (the reciprocal rank at which all gold evidence is first
retrieved, with Unicode normalization for PDF ligatures and typographic quotes), refusal
quality, citation validity, and faithfulness from a model judge. Its
`rag.json` includes the top five retrieved chunks per question with their source, score, trust,
and excerpt for diagnosing misses. It exports 10 judged answers to `handcheck.csv` for you to
verify by eye, and a
`ragas.jsonl` file in the layout RAGAS expects if you want a second opinion from that library.
`--calibrate` chooses the refusal threshold on half the questions and reports it on the other half.
If the provider rate-limits the additional judge requests, rerun with `--skip-judge` to keep
answer and retrieval metrics without faithfulness scoring. If a judge request receives HTTP 429
during a run, the evaluator stops further judging, reports partial faithfulness status, and still
saves the other metrics.

## What the tests cover

Metrics are hand-computed and cross-checked against scikit-learn. BM25 equals `rank_bm25`.
The v1 baseline equals the original v1 function. Both stores pass one shared contract suite,
including isolation, sharing, deletion, filters, and exact agreement of pgvector search with
brute-force cosine. The service tests cover incremental sync (an edit to the last 100 words of
an 800 word document re-embeds 1 of 5 chunks), poisoned documents never reaching the model,
tag-breakout attempts, and citation validation. The auth tests include algorithm-confusion
attacks.

## Reranking is opt-in

The cross-encoder is disabled by default (`SMARTRAG_RERANK_MODEL=none`). In the author's SciFact
run it did not improve nDCG@10 over BM25 plus dense fusion and added several seconds per query on
CPU; the numbers are in `results/scifact/results.md`. A reranker can still help on other
domains, so decide with data: run `evals.rag_eval` on your own documents with and without
`--reranker`, and keep it only if the hit rate improves enough to justify the latency.

Without a reranker there is no relevance score to refuse on, so out-of-scope questions are caught
by the model's own refusal rather than a threshold. `evals.rag_eval --calibrate` needs a reranker.

## Query rewriting

A follow-up such as "what about its cost?" has no searchable meaning alone. The client sends the
last few answered turns as `history`, and `QueryRewriter` asks the model for one standalone search
query. The rewritten text drives retrieval and the answer prompt. Chat history keeps the user's own
wording, and the UI shows "Searched for: ..." so the rewrite is never hidden.

Guard rails: it runs only when history exists, so a first question costs no extra model call; the
output must be one line of at most 300 characters; any failure falls back to the original question
and the request still succeeds. Its latency appears as `rewrite_ms` in the response timings.

It has not been shown to help. Public benchmarks such as SciFact are single-turn, so they cannot
measure it. To test it, add follow-up rows with a `history` field to your goldset (see the
`evals.rag_eval` docstring). The report then compares retrieval hit rate with and without the
rewrite on those rows. Report that comparison, with its interval, or make no claim.

## Known limits

- The sanitiser is a pattern matcher. Paraphrased, encoded, and non-English attacks pass it
  (see the injection report for current numbers), and it flags some legitimate text about
  security. Trust scoring reduces risk, it does not remove it.
- The v1 "TF-IDF" component was hashed term frequency with L2 normalisation. HashingVectorizer
  applies no idf weighting, and `tests/test_v1_fidelity.py` demonstrates this.
- Dense retrieval needs an embedding model. A 512 MB free tier cannot host the local models,
  so deploy with an embedding API (`OpenAICompatEmbedder`) or on a larger instance.
- The API does not stream tokens, and there is no rate limiting, password reset, or
  document-level audit log.
- Database connections use a small pool and the schema is created with `CREATE TABLE IF NOT
  EXISTS`. There is no migration tool yet.
