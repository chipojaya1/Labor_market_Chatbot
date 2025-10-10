# New Prototype: Data Science Career Copilot

To support aspiring data scientists, the repository now also contains a lightweight Retrieval-Augmented Generation (RAG) prototype built with Flask (`app/`). The service indexes curated Bureau of Labor Statistics (BLS) and Glassdoor insights stored in `data/rag_corpus.json` and exposes a `/ask` endpoint plus a minimal chat interface. Each response surfaces bullet-point guidance with explicit source citations, aligning with the transparency requirement for career-planning research.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FLASK_APP=app.app flask run
```

### Using the SQLite DB for RAG

If you prefer a persisted, queryable knowledge store (recommended for larger
corpora), you can create a SQLite database from the CSV datasets and the
existing `data/rag_corpus.json`. The repository includes a helper script that
builds `data/labor_market.db` and a `rag_documents` table used by the RAG
pipeline.

WSL / Linux (recommended):

```bash
# Activate your venv (or use the WSL venv created by the setup steps above)
source /home/<user>/.venv_labormarket/bin/activate  # or use your project venv
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas scikit-learn
python scripts/create_db.py

# Run the Flask app using the DB-backed corpus
export RAG_DB_PATH=$(pwd)/data/labor_market.db
export FLASK_APP=app.app
flask run --host=127.0.0.1
```

PowerShell (Windows):

```powershell
# From the project root (PowerShell)
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas scikit-learn
python .\scripts\create_db.py

# Tell the Flask app to use the DB
$env:RAG_DB_PATH = (Resolve-Path .\data\labor_market.db).Path
$env:FLASK_APP = 'app.app'
flask run
```


## Installation (conda, PowerShell venv, WSL venv)

This project works best with Python 3.11 (recommended) or 3.10. Some binary packages (numpy, scikit-learn) may not have prebuilt wheels for very new Python versions (e.g. 3.13). If you see long pip build steps or compiler errors, prefer the conda instructions below.

1) Recommended: Conda (cross-platform, uses prebuilt binaries)

```powershell
# New Prototype: Data Science Career Copilot

This repository contains a lightweight Retrieval-Augmented Generation (RAG) prototype built with Flask (in `app/`). The service indexes curated BLS and Glassdoor insights from CSV/JSON data and exposes a `/ask` API plus a minimal chat UI. The code supports either a simple in-memory JSON corpus or a persisted SQLite DB with FTS indexing for faster, scalable retrieval.

This README consolidates everything you need to: create a venv (WSL or PowerShell), build the SQLite DB from the CSVs, generate expanded RAG documents, run the supervised Flask server (stable cross-host access from Windows ↔ WSL), test the endpoint from PowerShell, and next steps (LLM polishing, tests).

---

## Quickstart (minimal, WSL or macOS)

These steps get the app running quickly using the JSON corpus (no DB):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export FLASK_APP=app.app
flask run --host=127.0.0.1
```

Open http://127.0.0.1:5000 in your browser (or the Flask chat UI in `app/static/index.html`) and use `/ask` for programmatic queries.

---

## Using the SQLite DB for a persisted RAG corpus (recommended)

For a larger or evolving corpus, create the SQLite DB from the CSV files and the existing `data/rag_corpus.json`. The repo includes helper scripts that:

- create `data/labor_market.db` and CSV-derived tables
- create a `rag_documents` table used by the RAG pipeline
- optionally create an FTS5 virtual table `rag_fts` if SQLite in your environment supports FTS5

### WSL / Linux (recommended)

1. Activate your WSL venv (or use the project venv inside WSL):

```bash
source /home/<user>/.venv_labormarket/bin/activate  # or cd into project and use .venv
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas scikit-learn   # required for DB ingestion and RAG
python scripts/create_db.py
```

2. Start Flask with the DB-backed corpus (bind to local loopback so Windows can reach it from WSL2):

```bash
export RAG_DB_PATH=$(pwd)/data/labor_market.db
export FLASK_APP=app.app
flask run --host=127.0.0.1 --port=5001
```

Note: if you prefer to run the server supervised (restarts on crash and binds 0.0.0.0), see the Supervisor section below.

### Windows PowerShell

From the project root in PowerShell:

```powershell
# Create & activate a PowerShell venv
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1

# Install required packages for DB creation + RAG
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas scikit-learn
python .\scripts\create_db.py

# Point Flask at the DB and run
$env:RAG_DB_PATH = (Resolve-Path .\data\labor_market.db).Path
$env:FLASK_APP = 'app.app'
flask run --host=127.0.0.1 --port=5001
```

If `Activate.ps1` is blocked by policy, you can use the venv executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install pandas scikit-learn
.\.venv\Scripts\python.exe .\scripts\create_db.py
.\.venv\Scripts\python.exe -m flask run --host=127.0.0.1 --port=5001
```

---

## Supervisor: keep Flask supervised and accessible from Windows

On WSL there is a helper `scripts/supervise_flask.py` that restarts the Flask process on failure and can bind to all interfaces (0.0.0.0) so the Windows host can reach it in WSL2. Use it when you want a persistent local service.

Start supervisor in WSL (example):

```bash
# start under your WSL venv
/home/<user>/.venv_labormarket/bin/python scripts/supervise_flask.py --port 5001 &
# or run in foreground for logs
/home/<user>/.venv_labormarket/bin/python scripts/supervise_flask.py --port 5001
```

Notes:

- If you previously had Flask instances running, kill them first to avoid "address already in use":

```bash
pkill -f 'flask' || true
pkill -f 'labor_chatbot' || true
```

- The supervisor writes logs to `/tmp/labor_chatbot_supervised.log` by default — check that file for startup errors.

---

## Health-check and testing from PowerShell

Once the server is listening on `127.0.0.1:5001` you can POST from PowerShell to `/ask`:

```powershell
$body = @{ question = 'What are the top skills for Data Scientists?'; max_sources = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:5001/ask' -ContentType 'application/json' -Body $body
```

If you see connection refused, ensure the supervisor / Flask process is running in WSL and is bound to `127.0.0.1` or `0.0.0.0`, and that no other process is claiming the port.

---

## API behavior and useful flags

The `/ask` endpoint accepts JSON like:

```json
{
  "question": "What skills are needed for Data Scientists?",
  "max_sources": 3,
  "include_table_rows": true
}
```

- `max_sources` (int): number of top sources to surface.
- `include_table_rows` (bool): when true (default for DB-backed sources) the response may include a short CSV-like snippet (1–3 rows) for sources that map to CSV tables (OEWS, glassdoor, etc.). Use false to suppress long table snippets.

Response shape (example):

```json
{
  "answer": "<string: synthesized answer>",
  "sources": [ { "id": "...", "title": "...", "score": 0.12, "source_name": "..." }, ... ]
}
```

The RAG pipeline prefers DB-backed generated documents when `RAG_DB_PATH` is set. If not present, it falls back to `data/rag_corpus.json`.

---

## Generate expanded RAG documents from CSVs (optional)

To create additional per-occupation and per-region summary documents and upsert them into the DB, run:

```bash
python scripts/generate_rag_from_csv.py
```

This writes `data/rag_corpus_generated.json` and upserts generated docs into `rag_documents` in `data/labor_market.db` (if present). These generated docs improve answer relevance for occupation- and region-specific queries.

---

## LLM-based summary polishing (optional)

There is a scaffold `scripts/generate_summaries_llm.py` to polish generated summaries using an LLM (OpenAI or HF). It is intentionally opt-in and requires API keys.

Usage (example with OpenAI-style env var):

```bash
export OPENAI_API_KEY='sk-...'
python scripts/generate_summaries_llm.py --input data/rag_corpus_generated.json --output data/rag_corpus_llm.json --provider openai
```

If you want I can implement a default OpenAI/requests implementation; currently the script is a safe no-op when no key/provider is provided.

---

## Reporting and analysis

There are helper scripts under `scripts/`:

- `scripts/produce_reports.py` — generates CSVs and PNGs for top occupations/states found in the CSV datasets into an `outputs/` folder.
- `scripts/test_rag_db.py` — quick test harness that loads the DB-backed RagPipeline and runs sample queries.
- `scripts/demo_query.py` — small client to POST to the `/ask` endpoint (can be used from PowerShell or bash).

---

## Troubleshooting

- "python not found" in WSL: use `python3` or install the distribution Python. Prefer creating venvs inside WSL's native filesystem (e.g., `/home/<user>/.venv_labormarket`) because Windows-mounted drives (`/mnt/c/...`) sometimes block virtualenv ensurepip or symlink operations.
- "Operation not permitted" when creating venv on `/mnt/c`: create venv inside WSL's home and use that venv for running scripts.
- "Address already in use" when starting supervisor: kill leftover Flask processes first (see Supervisor section) and re-run. I can also harden `scripts/supervise_flask.py` to include a pre-check and exponential backoff on request.
- Pip build failures for binary packages: use conda/miniforge or install system build dependencies in WSL as shown in the WSL setup section.

If pip fails during a verbose install, save the pip log and paste the last ~200 lines when asking for help:

```bash
pip install -r requirements.txt -v 2>&1 | tee pip-install.log
tail -n 200 pip-install.log
```

---

## Tests and CI (suggested)

Recommended quick checks you can add to CI or run locally:

1. Unit test for `RagPipeline` retrieval (DB-backed): ensure it returns >0 sources for a basic query.
2. Lightweight integration test: start supervisor in background, POST to `/ask` and assert HTTP 200 + answer string.

I can add these tests if you'd like; they are small and quick to run with the existing venv.

---

## Developer notes & next steps

- Add `include_table_rows` to the UI controls so users can toggle snippet in the chat UI.
- Stabilize `scripts/supervise_flask.py` (add pidfile, backoff, rotation); this is already on the todo list.
- Implement LLM polishing in `scripts/generate_summaries_llm.py` once you provide an API key.

If you want, I will now:
1) Harden `scripts/supervise_flask.py` to avoid address-in-use restart loops and add a pidfile (recommended), and
2) add the `include_table_rows` API flag to `app.app` and the UI.

Tell me which of these you'd like me to implement next.
