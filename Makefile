.PHONY: install db test lint smoke injection scifact demo frontend-test e2e

install:
	pip install -e ".[dev,ml]"

db:
	docker compose up -d db

test:
	python -m pytest -q

lint:
	ruff check . && ruff format --check .

# Pipeline check with test doubles. The output is labelled SMOKE TEST and proves nothing about quality.
smoke:
	python -c "from evals.synthetic import make_synthetic_beir; make_synthetic_beir('data/toy')"
	python -m evals.run_ablation --data data/toy --embedder hashing:128 --reranker overlap --out results

injection:
	python -m evals.injection_eval

# Real retrieval results. Needs network access and the ml extra.
scifact:
	python scripts/download_beir.py scifact data
	python -m evals.run_ablation --data data/scifact --out results
	python -m evals.injection_eval --corpus data/scifact

# Backend with a hashing embedder and extractive answers, for trying the interface.
demo:
	python -m smartrag.demo

frontend-test:
	cd frontend && npm ci && npm run build && npm test

# Starts the demo backend, runs the UI against it, then stops it.
e2e:
	python -m smartrag.demo & pid=$$!; sleep 3; (cd frontend && npm run test:e2e); status=$$?; kill $$pid; exit $$status
