"""
Optional script to improve generated summaries by calling an LLM.

This is intentionally optional and doesn't require an API by default. Set
HUGGINGFACE_API_KEY or OPENAI_API_KEY in the environment and implement the
preferred provider logic inside `call_llm`.

Usage:
    python scripts/generate_summaries_llm.py --input data/rag_corpus_generated.json --output data/rag_corpus_llm.json

If no API key is provided the script copies input => output unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def call_llm(prompt: str) -> str:
    # Placeholder: no-op if no API keys configured. You can implement OpenAI
    # or HuggingFace calls here. For now we just return the prompt (or a
    # shortened version) to avoid requiring API credentials.
    if os.getenv("OPENAI_API_KEY") or os.getenv("HUGGINGFACE_API_KEY"):
        # Implement provider call here if desired.
        return ""  # implement API call
    return prompt[:1000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    docs = json.loads(inp.read_text(encoding="utf-8"))
    enhanced: List[Dict[str, str]] = []
    for d in docs:
        text = d.get("content", "")
        prompt = f"Summarize the following dataset insight in 2-3 concise bullets:\n\n{text}"
        summary = call_llm(prompt)
        d2 = dict(d)
        d2["summary_llm"] = summary
        enhanced.append(d2)

    out.write_text(json.dumps(enhanced, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote enhanced corpus to {out}")
