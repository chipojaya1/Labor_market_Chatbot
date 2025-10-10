"""
Simple supervisor for the Flask app. Runs the app and restarts if it dies.
Binds to 0.0.0.0 so Windows can access the service when using WSL2.

Usage:
    python scripts/supervise_flask.py --port 5001
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path("/home/chipo/.venv_labormarket/bin/python")
LOG = Path("/tmp/labor_chatbot_supervised.log")


def run_supervisor(port: int = 5001) -> None:
    cmd = [str(VENV_PY), "-m", "flask", "run", "--host=0.0.0.0", f"--port={port}"]
    env = {**dict(), "FLASK_APP": "app.app", "RAG_DB_PATH": str(ROOT / "data" / "labor_market.db"), "PYTHONPATH": str(ROOT)}
    print(f"Supervisor starting Flask on 0.0.0.0:{port} (logs -> {LOG})")
    while True:
        with LOG.open("ab") as fp:
            proc = subprocess.Popen(cmd, stdout=fp, stderr=fp, env={**env, **dict()})
            try:
                ret = proc.wait()
                print(f"Flask exited with {ret}; restarting in 2s...")
                time.sleep(2)
            except KeyboardInterrupt:
                proc.terminate()
                print("Supervisor exiting")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    run_supervisor(args.port)
