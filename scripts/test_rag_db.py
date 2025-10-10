"""Quick test that the RagPipeline can load documents from the created SQLite DB."""
from __future__ import annotations

import json
from pathlib import Path

from app.rag import RagPipeline

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "labor_market.db"

if not DB.exists():
    raise SystemExit(f"DB not found: {DB}")

rp = RagPipeline(db_path=str(DB))
query = "What are current hiring trends for data scientists?"
res = rp.answer(query)
print(json.dumps(res, indent=2, ensure_ascii=False))
