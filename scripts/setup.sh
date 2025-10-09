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
YES_FLAG=0

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

# parse optional --yes to skip prompts
for arg in "$@"; do
  if [[ "$arg" == "--yes" || "$arg" == "-y" ]]; then
    YES_FLAG=1
  fi
done

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
      # use pip from the conda env
      python -m pip install -r "$REQUIREMENTS" --no-deps --prefer-binary
    fi

    echo "Environment created. Run: conda activate $ENV_NAME && FLASK_APP=app.app flask run"
    ;;

  --venv)
    # Find python3 or fallback to python/py
    PY_CMD=""
    if command -v python3 >/dev/null 2>&1; then
      PY_CMD=python3
    elif command -v python >/dev/null 2>&1; then
      PY_CMD=python
    elif command -v py >/dev/null 2>&1; then
      # Use py launcher on Windows
      PY_CMD="py -3"
    fi

    if [[ -z "$PY_CMD" ]]; then
      echo "No Python interpreter found. Install Python 3.11 (recommended) or use --conda." >&2
      exit 2
    fi

    echo "Creating venv .venv with: $PY_CMD"
    # create venv using chosen python
    $PY_CMD -m venv .venv

    # define venv-local pip/python paths
    if [[ -x ".venv/bin/pip" ]]; then
      VENV_PIP=.venv/bin/pip
      VENV_PY=.venv/bin/python
    else
      VENV_PIP=.venv/Scripts/pip.exe
      VENV_PY=.venv/Scripts/python.exe
    fi

    echo "Upgrading pip, setuptools, wheel inside .venv..."
    "$VENV_PIP" install --upgrade pip setuptools wheel

    echo "Installing requirements into .venv (preferring binary wheels)..."
    if [[ -f "$REQUIREMENTS" ]]; then
      "$VENV_PIP" install --prefer-binary -r "$REQUIREMENTS"
    fi

    echo "Venv ready. Activation instructions:"
    echo "  Bash / Git-bash / WSL: source .venv/bin/activate && export FLASK_APP=app.app && flask run"
  echo '  PowerShell (Windows): .\.venv\Scripts\Activate.ps1; $env:FLASK_APP='\''app.app'\''; flask run'
    ;;

  --wsl)
    # Run the venv creation and pip install inside WSL
    if ! command -v wsl >/dev/null 2>&1; then
      echo "wsl command not found. Install WSL or run the --venv flow inside WSL manually." >&2
      exit 2
    fi

    # Convert current project root (Windows path) to WSL absolute path
    WSL_PATH=$(wsl wslpath -a "${PROJECT_ROOT//\\/\\\\}")
    echo "Running WSL install in: $WSL_PATH"

    wsl bash -lc "set -e; cd '$WSL_PATH'; python3 -m venv .venv || python -m venv .venv; source .venv/bin/activate; pip install --upgrade pip setuptools wheel; pip install --prefer-binary -r requirements.txt"

    echo "WSL venv created. In WSL: cd '$WSL_PATH' && source .venv/bin/activate && export FLASK_APP=app.app && flask run"
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
