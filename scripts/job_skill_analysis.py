"""CLI tool to analyze job descriptions for in-demand skills.

This script wraps the LAiSER ``SkillExtractorRefactored`` pipeline so the
project's users can replicate the "Job Skill Analysis for Job Seekers" notebook
from Satya Phanindra Kumar Kalaga directly from the command line.  It mirrors
the steps from the original notebook:

1. Load a CSV file containing job descriptions.
2. Run the Hugging Face powered skill extractor.
3. Aggregate the detected skills.
4. Export helpful visualisations (bar chart + word cloud) alongside a CSV
   summary of skill frequencies.

Usage example::

    python scripts/job_skill_analysis.py \
        --input data/linkedin_jobs.csv \
        --model-id mistralai/Mistral-7B-Instruct-v0.1 \
        --hf-token $HUGGINGFACE_TOKEN \
        --output-dir outputs/skill_analysis

The Hugging Face token is sensitive; prefer passing it via the ``--hf-token``
flag or the ``HUGGINGFACEHUB_API_TOKEN`` environment variable instead of
hard-coding it inside the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from laiser.skill_extractor_refactored import SkillExtractorRefactored
from wordcloud import WordCloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a CSV file containing job descriptions. The file must have "
            "a column named 'description'."
        ),
    )
    parser.add_argument(
        "--id-column",
        default="job_id",
        help=(
            "Name of the column that uniquely identifies each job posting. "
            "Defaults to 'job_id'."
        ),
    )
    parser.add_argument(
        "--text-column",
        default="description",
        help="Name of the column that stores the job description text.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model identifier, e.g. 'mistralai/Mistral-7B-Instruct-v0.1'.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token. If omitted the script will fall back to "
            "the HUGGINGFACEHUB_API_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/job_skill_analysis"),
        help="Directory where visualisations and summary files will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top skills to include in the bar chart. Defaults to 20.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU inference if the environment supports it.",
    )
    return parser.parse_args()


def load_job_postings(path: Path, id_column: str, text_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    missing_columns = {id_column, text_column} - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    subset = df[[id_column, text_column]].dropna()
    if subset.empty:
        raise ValueError("No job postings with descriptions were found in the dataset.")

    return subset


def initialise_extractor(model_id: str, hf_token: str | None, use_gpu: bool) -> SkillExtractorRefactored:
    token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face access token must be provided via --hf-token or the "
            "HUGGINGFACEHUB_API_TOKEN environment variable."
        )

    return SkillExtractorRefactored(model_id=model_id, hf_token=token, use_gpu=use_gpu)


def extract_skills(
    extractor: SkillExtractorRefactored,
    job_postings: pd.DataFrame,
    id_column: str,
    text_column: str,
) -> pd.DataFrame:
    extracted = extractor.extract_and_align(
        job_postings,
        id_column=id_column,
        text_columns=[text_column],
        input_type="job_desc",
    )

    if extracted.empty:
        raise ValueError("The skill extractor returned an empty result set.")

    return extracted


def save_skill_summary(skills: pd.DataFrame, output_dir: Path) -> Path:
    summary_path = output_dir / "skill_summary.csv"
    counts = skills["skill"].value_counts().rename_axis("skill").reset_index(name="count")
    counts.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return summary_path


def plot_top_skills(skills: pd.DataFrame, top_n: int, output_dir: Path) -> Path:
    top_counts = skills["skill"].value_counts().nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(12, 8))
    top_counts.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {len(top_counts)} Most In-Demand Skills")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Skill")
    fig.tight_layout()

    chart_path = output_dir / "top_skills.png"
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path


def plot_wordcloud(skills: pd.DataFrame, output_dir: Path) -> Path:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(skills["skill"].dropna())
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Extracted Skills")
    fig.tight_layout()

    image_path = output_dir / "skill_wordcloud.png"
    fig.savefig(image_path)
    plt.close(fig)
    return image_path


def save_metadata(
    output_dir: Path,
    input_path: Path,
    model_id: str,
    top_n: int,
    generated_files: Iterable[Path],
) -> Path:
    metadata = {
        "input_csv": str(input_path),
        "model_id": model_id,
        "top_n": top_n,
        "artifacts": [str(path) for path in generated_files],
    }
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    return metadata_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading job postings…")
    postings = load_job_postings(args.input, args.id_column, args.text_column)
    print(f"Loaded {len(postings)} postings with descriptions.")

    print("Initialising LAiSER skill extractor…")
    extractor = initialise_extractor(args.model_id, args.hf_token, args.use_gpu)
    print("Skill extractor ready.")

    print("Extracting skills… this may take a moment depending on model size.")
    extracted = extract_skills(extractor, postings, args.id_column, args.text_column)
    print(f"Extraction complete. Found {len(extracted)} skill mentions.")

    print("Saving aggregated outputs…")
    summary_path = save_skill_summary(extracted, args.output_dir)
    bar_chart_path = plot_top_skills(extracted, args.top_n, args.output_dir)
    wordcloud_path = plot_wordcloud(extracted, args.output_dir)
    metadata_path = save_metadata(
        args.output_dir,
        args.input,
        args.model_id,
        args.top_n,
        [summary_path, bar_chart_path, wordcloud_path],
    )

    print("Artifacts written:")
    for path in [summary_path, bar_chart_path, wordcloud_path, metadata_path]:
        print(f" - {path}")

    print("Done. Review the PNG files for visuals and the CSV for exact counts.")


if __name__ == "__main__":
    main()

