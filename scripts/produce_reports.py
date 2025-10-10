"""
Produce simple analytical reports (CSV + PNG) for top occupations and regions.
Outputs are written to `outputs/`.

Usage:
    python scripts/produce_reports.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CSV_TABLES = {
    "glassdoor_salary": "Glassdoor_Salary_Cleaned_Version.csv",
    "oews_salary": "oews_cleaned_2024.csv",
}


def report_top_occupations_oews():
    path = DATA / CSV_TABLES["oews_salary"]
    df = pd.read_csv(path)
    grp = df.groupby("Occupation", dropna=True).agg({"Total_Jobs": "sum", "Average_Salary": "mean"})
    top = grp.sort_values("Total_Jobs", ascending=False).head(15)
    top.to_csv(OUT / "oews_top_occupations.csv")

    plt.figure(figsize=(8, 6))
    top["Total_Jobs"].sort_values().plot.barh()
    plt.title("Top occupations by estimated employment (OEWS)")
    plt.tight_layout()
    plt.savefig(OUT / "oews_top_occupations.png")
    plt.close()


def report_top_states_glassdoor():
    path = DATA / CSV_TABLES["glassdoor_salary"]
    df = pd.read_csv(path)
    if "job_state" in df.columns:
        top = df["job_state"].value_counts().head(20)
        top.to_csv(OUT / "glassdoor_top_states.csv")

        plt.figure(figsize=(8, 6))
        top.sort_values().plot.barh()
        plt.title("Top states by Glassdoor listing count")
        plt.tight_layout()
        plt.savefig(OUT / "glassdoor_top_states.png")
        plt.close()


def main() -> None:
    report_top_occupations_oews()
    report_top_states_glassdoor()
    print(f"Wrote reports to {OUT}")


if __name__ == "__main__":
    main()
