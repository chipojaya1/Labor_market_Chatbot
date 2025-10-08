# New Prototype: Data Science Career Copilot

To support aspiring data scientists, the repository now also contains a lightweight Retrieval-Augmented Generation (RAG) prototype built with Flask (`app/`). The service indexes curated Bureau of Labor Statistics (BLS) and Glassdoor insights stored in `data/rag_corpus.json` and exposes a `/ask` endpoint plus a minimal chat interface. Each response surfaces bullet-point guidance with explicit source citations, aligning with the transparency requirement for career-planning research.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app.app run
```

Open <http://127.0.0.1:5000> to chat with the assistant. Provide questions about entry-level salaries, required skills, hiring trends, or interview preparation to receive sourced answers.

## Troubleshooting

If pip fails with a Cython / build error while installing packages (for example errors mentioning "Cython" or "sklearn/... .pyx"), pip is trying to compile a package from source because a prebuilt wheel wasn't available for your Python/macOS combination. Common causes: using a very new Python (e.g. 3.13) or an Apple Silicon macOS where wheels may be on conda-forge first.

Try one of these fixes (ordered by reliability):

- Use conda (recommended on macOS, especially M1/M2): it installs prebuilt binary wheels.

```bash
# create a conda environment with a supported Python and install prebuilt packages
conda create -n chat-env python=3.11 -y
conda activate chat-env
conda install -c conda-forge scikit-learn flask
# then install remaining Python-only requirements (if any) from pip
pip install -r requirements.txt --no-deps
```

- Use a supported CPython (3.11/3.10) virtualenv so pip can find compatible wheels:

```bash
# use a system python version that has prebuilt wheels available
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

- If you must build from source, install build tools and prerequisites first:

```bash
# macOS command line tools (provides compilers)
xcode-select --install
# preinstall common build deps
pip install --upgrade pip setuptools wheel cython numpy scipy
pip install -r requirements.txt
```

Notes:
- If you are on Apple Silicon (M1/M2), prefer conda-forge / miniforge for reliable binary packages.
- The error in your trace showing Cython compilation for `sklearn/... .pyx` indicates scikit-learn was attempted to be compiled — switching to a Python version or channel with prebuilt wheels avoids this.

For step-by-step install commands, see `docs/INSTALL.md`.
