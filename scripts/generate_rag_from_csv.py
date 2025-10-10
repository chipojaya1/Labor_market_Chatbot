"""
Generate RAG documents from the CSV datasets and insert them into the
SQLite `data/labor_market.db` under the `rag_documents` table.

This script creates two documents per CSV file:
- a schema document listing columns and types
- a summary document with row counts, top categories and sample rows

Run from project root (inside the venv):
    python scripts/generate_rag_from_csv.py

The script writes `data/rag_corpus_generated.json` and upserts documents into
`rag_documents` in the DB so the RAG pipeline (when pointed at the DB) can
retrieve these generated insights.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = DATA / "labor_market.db"
OUT_JSON = DATA / "rag_corpus_generated.json"

CSV_TABLES = {
    "glassdoor_salary": "Glassdoor_Salary_Cleaned_Version.csv",
    "bls_macro_indicators": "bls_macro_indicators_cleaned.csv",
    "oews_salary": "oews_cleaned_2024.csv",
}


def describe_dataframe(df: pd.DataFrame, max_samples: int = 3) -> Dict[str, Any]:
    row_count = len(df)
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}

    categorical_summary = {}
    for c in cols:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            top = df[c].dropna().astype(str).value_counts().head(5).to_dict()
            if top:
                categorical_summary[c] = top

    numeric_summary = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if not s.empty:
                numeric_summary[c] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }

    samples = df.head(max_samples).to_dict(orient="records")
    return {
        "row_count": row_count,
        "columns": cols,
        "dtypes": dtypes,
        "categorical_summary": categorical_summary,
        "numeric_summary": numeric_summary,
        "samples": samples,
    }


def render_schema_doc(table_name: str, desc: Dict[str, Any]) -> Dict[str, Any]:
    lines = [f"Table: {table_name}", f"Rows: {desc['row_count']}", "Columns:"]
    for col in desc["columns"]:
        lines.append(f"- {col} ({desc['dtypes'].get(col, 'unknown')})")
    content = "\n".join(lines)
    return {
        "id": f"generated_schema::{table_name}",
        "title": f"Schema: {table_name}",
        "summary": f"Schema and column list for table {table_name} ({desc['row_count']} rows)",
        "content": content,
        "source_name": table_name,
        "source_url": "",
        "last_updated": "generated",
    }


def render_summary_doc(table_name: str, desc: Dict[str, Any]) -> Dict[str, Any]:
    lines: List[str] = [f"Summary: {table_name}", f"Rows: {desc['row_count']}"]

    if desc["categorical_summary"]:
        lines.append("Top categorical values (sample):")
        for col, top in list(desc["categorical_summary"].items())[:5]:
            top_lines = ", ".join([f"{k} ({v})" for k, v in top.items()])
            lines.append(f"- {col}: {top_lines}")

    if desc["numeric_summary"]:
        lines.append("Numeric summaries (sample):")
        for col, stats in list(desc["numeric_summary"].items())[:5]:
            lines.append(f"- {col}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, min={stats['min']}, max={stats['max']}")

    lines.append("Sample rows:")
    for s in desc["samples"]:
        lines.append(f"- {s}")

    content = "\n".join(lines)
    return {
        "id": f"generated_summary::{table_name}",
        "title": f"Dataset summary: {table_name}",
        "summary": f"Top-level summary for {table_name} ({desc['row_count']} rows)",
        "content": content,
        "source_name": table_name,
        "source_url": "",
        "last_updated": "generated",
    }


def main() -> None:
    docs: List[Dict[str, Any]] = []

    for table_name, filename in CSV_TABLES.items():
        csv_path = DATA / filename
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        print(f"Processing {csv_path} -> {table_name}...")
        df = pd.read_csv(csv_path)
        desc = describe_dataframe(df)
        schema_doc = render_schema_doc(table_name, desc)
        summary_doc = render_summary_doc(table_name, desc)
        docs.extend([schema_doc, summary_doc])

        # Additional generated documents: per-occupation and per-region analysis
        try:
            if table_name == "oews_salary":
                # Per-occupation top summaries (limit top 10 occupations by jobs)
                grp = df.groupby("Occupation", dropna=True, as_index=False)
                occ_stats = grp.agg({"Total_Jobs": "sum", "Average_Salary": "mean", "Median_Salary": "mean"})
                occ_stats = occ_stats.sort_values("Total_Jobs", ascending=False).head(10)
                for _, row in occ_stats.iterrows():
                    occ = row["Occupation"]
                    content = (
                        f"Occupation: {occ}\nTotal jobs (approx): {int(row['Total_Jobs'])}\n"
                        f"Avg salary: {row['Average_Salary']:.2f} | Median salary: {row['Median_Salary']:.2f}"
                    )
                    docs.append(
                        {
                            "id": f"oews_occ::{occ}",
                            "title": f"OEWS summary: {occ}",
                            "summary": f"OEWS aggregate for {occ}",
                            "content": content,
                            "source_name": "BLS OEWS",
                            "source_url": "",
                            "last_updated": "generated",
                        }
                    )

                # Per-region top locations (limit top 6 locations)
                loc_grp = df.groupby("Location", dropna=True, as_index=False)
                loc_stats = loc_grp.agg({"Total_Jobs": "sum", "Average_Salary": "mean"})
                loc_stats = loc_stats.sort_values("Total_Jobs", ascending=False).head(6)
                for _, row in loc_stats.iterrows():
                    loc = row["Location"]
                    content = (
                        f"Location: {loc}\nTotal jobs (approx): {int(row['Total_Jobs'])}\n"
                        f"Avg salary: {row['Average_Salary']:.2f}"
                    )
                    docs.append(
                        {
                            "id": f"oews_loc::{loc}",
                            "title": f"OEWS location summary: {loc}",
                            "summary": f"OEWS aggregate for {loc}",
                            "content": content,
                            "source_name": "BLS OEWS",
                            "source_url": "",
                            "last_updated": "generated",
                        }
                    )

            if table_name == "glassdoor_salary":
                # Per-state skill prevalence (based on skill flags like python_yn, R_yn, aws, spark, excel)
                if "job_state" in df.columns:
                    skill_cols = [c for c in ["python_yn", "R_yn", "spark", "aws", "excel"] if c in df.columns]
                    state_grp = df.groupby("job_state", dropna=True)
                    state_counts = df["job_state"].value_counts().head(8)
                    for state in state_counts.index:
                        sub = df[df["job_state"] == state]
                        lines = [f"State: {state}", f"Total listings: {len(sub)}"]
                        for sc in skill_cols:
                            pct = 100.0 * sub[sc].dropna().astype(float).sum() / max(1, len(sub))
                            lines.append(f"- {sc}: {pct:.1f}%")
                        content = "\n".join(lines)
                        docs.append(
                            {
                                "id": f"glassdoor_state::{state}",
                                "title": f"Glassdoor skills: {state}",
                                "summary": f"Skill prevalence in {state} Glassdoor listings",
                                "content": content,
                                "source_name": "Glassdoor",
                                "source_url": "",
                                "last_updated": "generated",
                            }
                        )
        except Exception as exc:
            print(f"Warning: additional doc generation failed for {table_name}: {exc}")

    # Write generated corpus to JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=2)
    print(f"Wrote generated corpus: {OUT_JSON} ({len(docs)} docs)")

    # Upsert into DB rag_documents
    if not DB_PATH.exists():
        print(f"DB missing: {DB_PATH}; create it first with scripts/create_db.py")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS rag_documents (id TEXT PRIMARY KEY, title TEXT, summary TEXT, content TEXT, source_name TEXT, source_url TEXT, last_updated TEXT)")
        insert_sql = "INSERT OR REPLACE INTO rag_documents (id, title, summary, content, source_name, source_url, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)"
        for d in docs:
            conn.execute(insert_sql, (d["id"], d["title"], d["summary"], d["content"], d["source_name"], d["source_url"], d["last_updated"]))
        conn.commit()
        print(f"Upserted {len(docs)} generated docs into {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
