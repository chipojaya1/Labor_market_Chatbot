# New Prototype: Data Science Career Copilot

To support aspiring data scientists, the repository now also contains a lightweight Retrieval-Augmented Generation (RAG) prototype built with Flask (`app/`). The service indexes curated Bureau of Labor Statistics (BLS) and Glassdoor insights stored in `data/rag_corpus.json` and exposes a `/ask` endpoint plus a minimal chat interface. Each response surfaces bullet-point guidance with explicit source citations, aligning with the transparency requirement for career-planning research.

The chatbot now also ships with a lightweight SQLite layer that mirrors the three cleaned CSV datasets (`Glassdoor_Salary_Cleaned_Version.csv`, `bls_macro_indicators_cleaned.csv`, and `oews_cleaned_2024.csv`). This enables tool-calling agents to run precise analytical queries. Visit `GET /schema` to inspect the generated schema and the accompanying function-calling specification, or `POST /function/query` with a JSON payload such as `{"sql": "SELECT * FROM oews_salary LIMIT 5"}` to execute read-only SQL.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FLASK_APP=app.app flask run
```


## Installation (conda, PowerShell venv, WSL venv)

This project works best with Python 3.11 (recommended) or 3.10. Some binary packages (numpy, scikit-learn) may not have prebuilt wheels for very new Python versions (e.g. 3.13). If you see long pip build steps or compiler errors, prefer the conda instructions below.

1) Recommended: Conda (cross-platform, uses prebuilt binaries)

```powershell
# Create the environment (PowerShell)
conda create -n laboriq python=3.11 -c conda-forge -y
conda activate laboriq

# Install core binary packages from conda-forge
conda install -c conda-forge scikit-learn flask numpy pandas matplotlib wordcloud -y

# Install remaining pip-only requirements without re-resolving conda deps
pip install -r .\requirements.txt --no-deps
# New Prototype: Data Science Career Copilot

To support aspiring data scientists, this repository contains a lightweight Retrieval-Augmented Generation (RAG) prototype built with Flask (in `app/`). The service indexes curated BLS and Glassdoor insights stored in `data/rag_corpus.json` and exposes a `/ask` endpoint plus a minimal chat interface. Each response surfaces bullet-point guidance with explicit source citations.

## Quickstart

Run the app locally using a virtual environment (bash / macOS / WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export FLASK_APP=app.app
flask run
```

On Windows PowerShell use the PowerShell venv activation steps in the Installation section below.

## Installation (recommended: conda; alternatives: PowerShell venv, WSL venv)

This project is best run with Python 3.11 or 3.10. Newer CPython releases (3.13+) may not have prebuilt wheels for some packages and pip may try to compile parts of numpy / scikit-learn, which can fail or take a long time.

1) Recommended — Conda (cross-platform, uses prebuilt binary packages):

```powershell
# Create env (PowerShell)
conda create -n laboriq python=3.11 -c conda-forge -y
conda activate laboriq

# Install binary packages from conda-forge
conda install -c conda-forge scikit-learn flask numpy pandas matplotlib wordcloud -y

# Install any remaining pip-only requirements without re-resolving conda deps
python -m pip install --no-deps -r .\requirements.txt

# Quick import check
python - <<'PY'
import sys, numpy as np, sklearn
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__)
print('scikit-learn:', sklearn.__version__)
PY

# Run the app
$env:FLASK_APP = 'app.app'
flask run
```

2) Windows PowerShell venv (no conda):

```powershell
# Create venv (PowerShell)
python -m venv .venv

# Temporarily allow script activation for this session and activate the venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1

# Upgrade packaging tools and install requirements
pip install --upgrade pip setuptools wheel
pip install -r .\requirements.txt

# Quick import check
python - <<'PY'
import sys, numpy as np, sklearn
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__)
print('scikit-learn:', sklearn.__version__)
PY

# Run the app
$env:FLASK_APP = 'app.app'
flask run
```

If you cannot run `Activate.ps1` because of execution policy, you can use the venv executables directly:

```powershell
.\.venv\Scripts\pip.exe install --upgrade pip setuptools wheel
.\.venv\Scripts\pip.exe install -r .\requirements.txt
.\.venv\Scripts\python.exe -m flask run
```

3) WSL / Bash venv (Linux):

```bash
# (Debian/Ubuntu example) Install system build deps if you expect to build packages
# sudo apt update
# sudo apt install -y python3 python3-venv python3-pip build-essential python3-dev gfortran libopenblas-dev liblapack-dev

# Create and activate venv in WSL
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip / wheel and install requirements (prefer binary wheels)
pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt -v 2>&1 | tee pip-install.log

# Quick import check
python - <<'PY'
import sys, numpy as np, sklearn
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__)
print('scikit-learn:', sklearn.__version__)
PY

# Run the app in WSL
export FLASK_APP=app.app
flask run --host=127.0.0.1
```

Quick note: the repository includes `scripts/setup.sh --wsl` which automates creating a WSL venv and installing `requirements.txt` from the Windows host by invoking WSL. Use that if you prefer automation.

## Troubleshooting

- LAiSER version error: If you see `No matching distribution found for laiser>=0.4.2` the latest published version is 0.3.0. The included `requirements.txt` uses `laiser>=0.3.0`.

- Flask import error: If Flask reports `Could not import 'app.app'`, ensure `FLASK_APP` is set correctly:

```bash
# Bash
export FLASK_APP=app.app
flask run

# PowerShell
$env:FLASK_APP = 'app.app'
flask run
```

- Build failures when pip compiles packages: these usually show C/C++/Fortran compilation errors. Fixes:
  - Use conda/miniforge (recommended) which provides prebuilt wheels.
  - Install system build tools (see WSL example above) and retry pip install.

If pip fails during a verbose install, save the pip log and paste the last ~200 lines when asking for help:

```bash
pip install -r requirements.txt -v 2>&1 | tee pip-install.log
tail -n 200 pip-install.log
```

## Job Skill Analysis CLI

The repository bundles a command-line version of the Job Skill Analysis notebook powered by LAiSER. Example usage:

```bash
python scripts/job_skill_analysis.py \
  --input path/to/job_postings.csv \
  --model-id <model-or-empty> \
  --hf-token $HUGGINGFACE_TOKEN \
  --output-dir outputs/skill_analysis
```

This saves:
- `skill_summary.csv`
- `top_skills.png`
- `skill_wordcloud.png`

Supply your Hugging Face token via `--hf-token` or `HUGGINGFACEHUB_API_TOKEN` env var. Add `--use-gpu` if GPU inference is available.
