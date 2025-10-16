"""Generate a RAG corpus from the curated 2024 job postings dataset.

The resulting JSON corpus is used by the Flask application when a database
backend is unavailable.  Each job posting is converted into a compact document
highlighting the most important skills so retrieval works well for questions
such as "What skills are required for an entry-level data scientist?".

Usage (run from the project root)::

    python scripts/build_jobpostings_corpus.py

This script writes ``data/rag_corpus.json`` and prints a short summary of the
extracted skill distribution.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JOB_POSTINGS_PATH = DATA_DIR / "Jobpostings.csv"
OUTPUT_PATH = DATA_DIR / "rag_corpus.json"

# Hand-curated list of high-signal data science skills that show up frequently in
# 2024 job postings.  The regular expressions are intentionally simple so they
# remain fast when applied to every job description.
_RAW_SKILL_PATTERNS: Dict[str, str] = {
    "Python": r"\bpython\b",
    "SQL": r"\bsql\b",
    "R": r"(?<!\w)r(?:\s+programming)?(?!\w)",
    "Tableau": r"\btableau\b",
    "Power BI": r"power\s+bi",
    "Excel": r"\bexcel\b",
    "Looker": r"\blooker\b",
    "SAS": r"\bsas\b",
    "MATLAB": r"\bmatlab\b",
    "Scala": r"\bscala\b",
    "Java": r"\bjava\b",
    "C++": r"(?<!\w)c\+\+(?!\w)",
    "C#": r"(?<!\w)c#(?!\w)",
    "Julia": r"\bjulia\b",
    "Spark": r"\bspark\b",
    "Hadoop": r"\bhadoop\b",
    "Kafka": r"\bkafka\b",
    "Airflow": r"\bairflow\b",
    "dbt": r"(?<!\w)dbt(?!\w)",
    "Snowflake": r"\bsnowflake\b",
    "Databricks": r"\bdatabricks\b",
    "AWS": r"\baws\b",
    "Azure": r"\bazure\b",
    "GCP": r"\bgcp\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b",
    "Machine learning": r"machine\s+learning",
    "Deep learning": r"deep\s+learning",
    "Natural language processing": r"natural\s+language\s+processing|\bnlp\b",
    "Statistics": r"\bstatistics?\b",
    "Probability": r"\bprobability\b",
    "A/B testing": r"a/b\s+testing",
    "Experiment design": r"experiment(al)?\s+design",
    "Feature engineering": r"feature\s+engineering",
    "Predictive modelling": r"predictive\s+model(ing|ling)",
    "Time series": r"time\s+series",
    "Data visualisation": r"data\s+visuali[sz]ation",
    "Business intelligence": r"business\s+intelligence",
    "ETL": r"(?<!\w)etl(?!\w)",
    "Data warehousing": r"data\s+warehous",
    "BigQuery": r"big\s*query",
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "scikit-learn": r"scikit-?learn",
    "TensorFlow": r"tensorflow",
    "PyTorch": r"pytorch",
    "Visualization dashboards": r"dashboard(s)?",
    "Communication": r"communication\s+skills?",
}

_SKILL_PATTERNS: Dict[str, re.Pattern[str]] = {
    skill: re.compile(pattern, re.IGNORECASE)
    for skill, pattern in _RAW_SKILL_PATTERNS.items()
}


@dataclass(frozen=True)
class Document:
    """Representation of a RAG document."""

    id: str
    title: str
    summary: str
    content: str
    source_name: str
    source_url: str
    last_updated: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "last_updated": self.last_updated,
        }


def load_job_postings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Job postings file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = {"job_id", "description"} - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Job postings CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    subset = df[["job_id", "description"]].dropna()
    subset = subset[subset["description"].str.strip() != ""]
    if subset.empty:
        raise ValueError("Job postings dataset has no non-empty descriptions.")

    return subset


def extract_skills(text: str) -> List[str]:
    """Return a sorted, de-duplicated list of skills detected in ``text``."""

    if not text:
        return []

    lower = text.lower()
    found: List[str] = []
    seen = set()
    for skill, pattern in _SKILL_PATTERNS.items():
        if skill in seen:
            continue
        if pattern.search(lower):
            seen.add(skill)
            found.append(skill)
    return found


def _extract_location(lines: Sequence[str]) -> str | None:
    for line in lines:
        lower = line.lower()
        if lower.startswith("location:"):
            return line.split(":", 1)[1].strip()
    return None


def make_job_document(row: pd.Series, skill_counts: Counter) -> Tuple[Document, List[str]]:
    job_id = str(row["job_id"]).strip()
    description = str(row["description"]).strip()

    lines = [line.strip() for line in description.splitlines() if line.strip()]
    title_line = lines[0] if lines else "Data Science Job Posting"
    location = _extract_location(lines)

    skills = extract_skills(description)
    skill_counts.update(skills)

    summary_parts: List[str] = [f"Job posting {job_id}: {title_line}"]
    if location:
        summary_parts.append(f"Location: {location}")
    if skills:
        summary_parts.append("Key skills: " + ", ".join(skills[:10]))
    else:
        summary_parts.append("Highlights responsibilities and qualifications for the role.")

    summary = " | ".join(summary_parts)

    content_lines = [description]
    if skills:
        content_lines.append("\nExtracted skills: " + ", ".join(skills))
    content = "\n".join(content_lines)

    return Document(
        id=f"job::{job_id}",
        title=f"Job posting {job_id}",
        summary=summary,
        content=content,
        source_name="2024 job postings",
        source_url="",
        last_updated="2024",
    ), skills


def build_skill_distribution_doc(skill_counts: Counter, total_jobs: int) -> Document | None:
    if not skill_counts:
        return None

    top_skills = skill_counts.most_common(25)
    lines = [
        "Skill frequency across 2024 job postings",
        f"Analysed {total_jobs} job descriptions.",
        "Top skills and counts:",
    ]
    for skill, count in top_skills:
        lines.append(f"- {skill}: {count} postings")

    summary = "Most frequently requested skills include " + ", ".join(
        skill for skill, _ in top_skills[:8]
    )

    return Document(
        id="jobpostings::top_skills",
        title="Top skills in 2024 job postings",
        summary=summary,
        content="\n".join(lines),
        source_name="2024 job postings",
        source_url="",
        last_updated="2024",
    )


def build_overview_doc(total_jobs: int, average_skills: float) -> Document:
    lines = [
        "Overview of the 2024 job postings dataset",
        f"Total postings analysed: {total_jobs}",
        f"Average unique skills detected per posting: {average_skills:.1f}",
        "Dataset fields include job_id and free-form job description text.",
    ]
    return Document(
        id="jobpostings::overview",
        title="2024 job postings overview",
        summary=(
            f"Analysed {total_jobs} job postings with an average of {average_skills:.1f} "
            "skills mentioned per role."
        ),
        content="\n".join(lines),
        source_name="2024 job postings",
        source_url="",
        last_updated="2024",
    )


def write_corpus(documents: Iterable[Document], output_path: Path) -> None:
    serialised = [doc.to_dict() for doc in documents]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serialised, handle, ensure_ascii=False, indent=2)


def main() -> None:
    postings = load_job_postings(JOB_POSTINGS_PATH)
    skill_counts: Counter[str] = Counter()
    documents: List[Document] = []
    total_skill_mentions = 0

    for _, row in postings.iterrows():
        doc, skills = make_job_document(row, skill_counts)
        documents.append(doc)
        total_skill_mentions += len(skills)

    job_doc_count = len(documents)
    average_skills = total_skill_mentions / job_doc_count if job_doc_count else 0.0
    documents.append(build_overview_doc(job_doc_count, average_skills))

    skill_doc = build_skill_distribution_doc(skill_counts, job_doc_count)
    if skill_doc:
        documents.append(skill_doc)

    write_corpus(documents, OUTPUT_PATH)

    top_preview = ", ".join(f"{skill} ({count})" for skill, count in skill_counts.most_common(10))
    print(f"Wrote {len(documents)} documents to {OUTPUT_PATH}")
    if top_preview:
        print(f"Top skills: {top_preview}")


if __name__ == "__main__":
    main()
