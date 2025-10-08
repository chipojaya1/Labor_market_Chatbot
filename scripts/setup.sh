#!/usr/bin/env bash
# Setup script for the Data Science Career Copilot project
# Usage:
#   scripts/setup.sh --conda        # create conda env and install packages
#   scripts/setup.sh --venv         # create virtualenv with python3.11 and pip install
#   scripts/setup.sh --verify       # run import verification in existing env
#   scripts/setup.sh --help         # show this message

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
ENV_NAME="chat-env"
PYTHON_VERSION="3.11"

print_help() {
  sed -n '1,120p' "$0" | sed -n '1,12p'
}

verify_imports() {
  echo "Running import verification (numpy, scikit-learn)..."
  python - <<'PY'
import sys
try:
    import numpy as np
    import sklearn
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    # quick sanity check
    a = np.arange(6).reshape(2,3)
    print('NumPy test array:', a.shape)
    from sklearn.feature_extraction.text import TfidfVectorizer
    print('sklearn import OK')
except Exception as e:
    print('Import test failed:', e)
    raise
PY
}

if [[ ${#@} -eq 0 ]]; then
  print_help
  exit 1
fi

case "$1" in
  --conda)
    if ! command -v conda >/dev/null 2>&1; then
      echo "conda not found on PATH. Install Miniconda/Miniforge and try again." >&2
      exit 2
    fi

    echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "Activating $ENV_NAME..."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    echo "Installing numpy, scikit-learn, flask from conda-forge..."
    conda install -c conda-forge numpy scikit-learn flask -y

    echo "Installing remaining requirements via pip (no-deps)..."
    if [[ -f "$REQUIREMENTS" ]]; then
      pip install -r "$REQUIREMENTS" --no-deps
    fi

    echo "Environment created. Run: conda activate $ENV_NAME && flask --app app.app run"
    ;;

  --venv)
    if ! command -v python3.11 >/dev/null 2>&1; then
      echo "python3.11 not found on PATH. Install Python 3.11 or use --conda." >&2
      exit 2
    fi

    echo "Creating venv .venv with python3.11..."
    python3.11 -m venv .venv
    echo "Activating venv..."
    # shellcheck disable=SC1091
    source .venv/bin/activate

    echo "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel

    echo "Installing requirements..."
    if [[ -f "$REQUIREMENTS" ]]; then
      pip install -r "$REQUIREMENTS"
    fi

    echo "Venv ready. Activate with: source .venv/bin/activate && flask --app app.app run"
    ;;

  --verify)
    verify_imports
    ;;

  --help|-h)
    print_help
    ;;

  *)
    echo "Unknown option: $1" >&2
    print_help
    exit 3
    ;;

esac
