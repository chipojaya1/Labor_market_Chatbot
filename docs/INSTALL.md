Installation notes

If you hit Cython / scikit-learn build errors on macOS, prefer one of the following:

1) Conda (recommended for Apple Silicon)

```bash
conda create -n chat-env python=3.11 -y
conda activate chat-env
conda install -c conda-forge scikit-learn flask
pip install -r requirements.txt --no-deps
```

2) Use a CPython with prebuilt wheels (Python 3.11 recommended)

```bash
# Installation notes

If you hit Cython / scikit-learn build errors on macOS, prefer one of the following:

1. Conda (recommended for Apple Silicon)

```bash
conda create -n chat-env python=3.11 -y
conda activate chat-env
conda install -c conda-forge scikit-learn flask
pip install -r requirements.txt --no-deps
```

1. Use a CPython with prebuilt wheels (Python 3.11 recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

1. Build from source (slow; install toolchain first)

```bash
xcode-select --install
pip install --upgrade pip setuptools wheel cython numpy scipy
pip install -r requirements.txt
```

Notes:

- Prefer conda-forge/miniforge on Apple Silicon for the broadest wheel support.
- If you still see errors that mention `.pyx` files (Cython), that means pip is compiling scikit-learn from source.

## Binary incompatibility (numpy.dtype size changed)

If you see a runtime error like:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

This means a compiled extension (scikit-learn, murmurhash, etc.) was built against a different ABI of NumPy than the NumPy currently installed. Fixes:

- Using conda (recommended): create a fresh environment and install matching binary packages from conda-forge:

```bash
conda create -n chat-env python=3.11 -y
conda activate chat-env
conda install -c conda-forge numpy scikit-learn flask
pip install -r requirements.txt --no-deps
```

- If using pip/venv: upgrade/downgrade NumPy to match the compiled extensions, or reinstall scikit-learn after installing NumPy:

```bash
# inside your venv
pip uninstall -y numpy scikit-learn
pip install numpy==1.26.4  # pick a NumPy version compatible with your scikit-learn wheel
pip install scikit-learn==1.3.2
pip install -r requirements.txt --no-deps
```

Note: exact version combinations depend on your Python version and platform. If unsure, the conda-forge channel is easiest since it keeps compatible builds together.
