"""
Create a SQLite database combining the CSV datasets and the RAG corpus.

This script creates a database file at `data/labor_market.db` containing:
- tables: glassdoor_salary, bls_macro_indicators, oews_salary (from CSVs)
- table: rag_documents (id, title, summary, content, source_name, source_url, last_updated)

Usage:
    python scripts/create_db.py

"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = DATA / "labor_market.db"

CSV_TABLES = {
    "glassdoor_salary": "Glassdoor_Salary_Cleaned_Version.csv",
    "bls_macro_indicators": "bls_macro_indicators_cleaned.csv",
    "oews_salary": "oews_cleaned_2024.csv",
}

RAG_CORPUS = DATA / "rag_corpus.json"


def create_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        print(f"Removing existing DB: {db_path}")
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        # Ingest CSV tables
        for table_name, filename in CSV_TABLES.items():
            csv_path = DATA / filename
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CSV: {csv_path}")
            print(f"Loading {csv_path} into table {table_name}...")
            df = pd.read_csv(csv_path)
            # normalise column names minimally
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            df.to_sql(table_name, conn, index=False)
            # Do not create an index on the implicit rowid pseudo-column; some
            # SQLite builds and CSV-to-SQL flows make this invalid. If you need
            # indexes for specific columns add them here based on the dataset.

        # Ingest RAG corpus as a documents table
        if not RAG_CORPUS.exists():
            raise FileNotFoundError(f"Missing RAG corpus: {RAG_CORPUS}")

        with RAG_CORPUS.open("r", encoding="utf-8") as fp:
            docs = json.load(fp)

        print(f"Creating rag_documents table with {len(docs)} documents...")
        conn.execute(
            "CREATE TABLE rag_documents (id TEXT PRIMARY KEY, title TEXT, summary TEXT, content TEXT, source_name TEXT, source_url TEXT, last_updated TEXT)"
        )

        insert_sql = "INSERT OR REPLACE INTO rag_documents (id, title, summary, content, source_name, source_url, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)"
        for d in docs:
            conn.execute(
                insert_sql,
                (
                    d.get("id"),
                    d.get("title"),
                    d.get("summary"),
                    d.get("content"),
                    d.get("source_name"),
                    d.get("source_url"),
                    d.get("last_updated"),
                ),
            )
        # Content-based indexes require SQLite compile-time features (FTS)
        # so we avoid creating a prefix index here. Use an external full-text
        # search (FTS5) index if you need faster content retrieval.
        conn.commit()
        # Try to create an FTS5 virtual table for fast full-text search over
        # rag_documents.content. Not all SQLite builds include FTS5; if it
        # fails we emit a warning and continue without FTS.
        try:
            print("Attempting to create FTS5 virtual table 'rag_fts'...")
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(id, content, title, summary, source_name, tokenize='porter')")
            conn.execute("DELETE FROM rag_fts")
            conn.execute("INSERT INTO rag_fts (id, content, title, summary, source_name) SELECT id, content, title, summary, source_name FROM rag_documents")
            conn.commit()
            print("FTS5 table created: rag_fts")
        except sqlite3.OperationalError as exc:
            print(f"FTS5 not available or creation failed: {exc}; continuing without FTS")

        print(f"Database created: {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    create_db()
