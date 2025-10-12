# -*- coding: utf-8 -*-
"""Utilities for loading the labor market CSV datasets into an in-memory database.

The web chatbot relies on a structured view of the curated datasets so that an
LLM (or other orchestration layer) can reason about the available tables and run
read-only analytical queries.  This module exposes two key building blocks:

* :class:`LaborMarketDatabase` – loads the three cleaned CSV exports into an
  in-memory SQLite database and exposes helpers for schema inspection and safe
  SQL execution.
* :func:`build_call_function_spec` – helper that produces the JSON schema used
  when registering an OpenAI/Anthropic "function call" tool, describing how to
  invoke the SQL query helper exposed by :class:`LaborMarketDatabase`.

This module originated from a Colab notebook located at
https://colab.research.google.com/drive/1BEfY0oUB4vtOAf9XRlHkrXHXa3lq-FWv
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd


_NORMALISE_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata describing a column inside a table."""

    name: str
    original_name: str
    sql_type: str


@dataclass(frozen=True)
class TableMetadata:
    """Metadata describing a table loaded from a CSV file."""

    name: str
    source_path: Path
    row_count: int
    columns: List[ColumnMetadata]


class LaborMarketDatabase:
    """Lightweight SQLite wrapper over the curated labor market CSV datasets."""

    def __init__(
        self,
        data_dir: Path | str = Path("data"),
        table_files: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.table_files: Mapping[str, str] = table_files or {
            "glassdoor_salary": "Glassdoor_Salary_Cleaned_Version.csv",
            "bls_macro_indicators": "bls_macro_indicators_cleaned.csv",
            "oews_salary": "oews_cleaned_2024.csv",
        }

        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row

        self._table_metadata: MutableMapping[str, TableMetadata] = {}
        self._column_name_lookup: Dict[str, str] = {}
        self._column_order_lookup: Dict[str, int] = {}
        self._load_tables()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def tables(self) -> List[str]:
        return list(self._table_metadata.keys())

    @property
    def metadata(self) -> List[TableMetadata]:
        return list(self._table_metadata.values())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_schema_as_text(self) -> str:
        """Return a human-readable representation of the database schema."""

        lines: List[str] = [
            "Labor Market SQLite schema (derived from curated CSV exports):"
        ]
        for table in self.metadata:
            lines.append(
                f"\n• {table.name} — source: {table.source_path.name} ({table.row_count} rows)"
            )
            for column in table.columns:
                lines.append(
                    f"    - {column.name} ({column.sql_type}) — original column '{column.original_name}'"
                )
        return "\n".join(lines)

    def get_schema_as_dict(self) -> List[Dict[str, Any]]:
        """Return the schema metadata as JSON-serialisable dictionaries."""

        return [
            {
                "name": table.name,
                "source": str(table.source_path),
                "row_count": table.row_count,
                "columns": [
                    {
                        "name": column.name,
                        "original_name": column.original_name,
                        "sql_type": column.sql_type,
                    }
                    for column in table.columns
                ],
            }
            for table in self.metadata
        ]

    def execute_query(self, sql: str, max_rows: int | None = None) -> Dict[str, Any]:
        """Execute a read-only SQL query against the in-memory database.

        Parameters
        ----------
        sql:
            SQL statement to execute. Only ``SELECT``/``WITH`` statements are
            permitted.  ``PRAGMA`` calls are rejected to keep the surface area
            minimal for the chatbot.
        max_rows:
            Optional maximum number of rows to fetch. Defaults to 50 and is
            capped at 200 to avoid overwhelming the chat responses.
        """

        if not sql or not sql.strip():
            raise ValueError("A non-empty SQL query must be provided.")

        normalised = sql.strip()
        # Allow a trailing semicolon, but disallow multiple statements.
        if normalised.endswith(";"):
            normalised = normalised[:-1].rstrip()
        if ";" in normalised:
            raise ValueError("Only a single SQL statement is permitted per call.")

        first_token = normalised.split(None, 1)[0].lower()
        if first_token not in {"select", "with"}:
            raise ValueError("Only read-only SELECT queries are supported.")

        limit = 50 if max_rows is None else max(1, min(int(max_rows), 200))

        cursor = self.connection.execute(normalised)
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        if truncated:
            rows = rows[:limit]

        serialised_rows = [dict(row) for row in rows]
        return {
            "rows": serialised_rows,
            "row_count": len(serialised_rows),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _introspect_table(self, table_name: str, csv_path: Path) -> TableMetadata:
        pragma_rows = self.connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns: List[ColumnMetadata] = []
        type_map = {row[1]: row[2] or "TEXT" for row in pragma_rows}
        matching_keys = [
            key
            for key in self._column_name_lookup
            if key.startswith(f"{table_name}::")
        ]
        matching_keys.sort(key=lambda key: self._column_order_lookup.get(key, 0))

        for lookup_key in matching_keys:
            column_key = lookup_key.split("::", 1)[1]
            original = self._column_name_lookup[lookup_key]
            sql_type = type_map.get(column_key, "TEXT")
            columns.append(
                ColumnMetadata(
                    name=column_key,
                    original_name=original,
                    sql_type=sql_type,
                )
            )

        row_count = self.connection.execute(
            f"SELECT COUNT(*) FROM '{table_name}'"
        ).fetchone()[0]

        return TableMetadata(
            name=table_name,
            source_path=csv_path,
            row_count=row_count,
            columns=columns,
        )

    def _normalise_columns(self, column_names: Iterable[str]) -> List[str]:
        """Convert raw CSV column names into safe SQL identifiers."""

        normalised_columns: List[str] = []
        seen: Dict[str, int] = {}

        for original_name in column_names:
            base = _NORMALISE_PATTERN.sub("_", original_name.strip()).strip("_").lower()
            if not base:
                base = "column"

            count = seen.get(base, 0)
            seen[base] = count + 1
            final_name = base if count == 0 else f"{base}_{count}"
            normalised_columns.append(final_name)

            # Store lookup so we can recover the original column label when
            # building the schema metadata.
            lookup_key = f"{self._current_table}::{final_name}"
            self._column_name_lookup[lookup_key] = original_name
            self._column_order_lookup[lookup_key] = len(normalised_columns) - 1

        return normalised_columns

    # ------------------------------------------------------------------
    # Context manager helpers used during column normalisation
    # ------------------------------------------------------------------
    def __enter__(self) -> "LaborMarketDatabase":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        self.connection.close()

    def _prepare_column_lookup(self, table_name: str) -> None:
        self._current_table = table_name

    # Override of _load_tables to utilise lookup helper
    def _load_tables(self) -> None:  # type: ignore[override]
        self._column_name_lookup: Dict[str, str] = {}
        raw_frames: Dict[str, pd.DataFrame] = {}
        for table_name, filename in self.table_files.items():
            csv_path = self.data_dir / filename
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Expected dataset not found: {csv_path}"
                )

            raw_frames[table_name] = pd.read_csv(csv_path)

        star_schema_tables = self._build_star_schema(raw_frames)

        for table_name, (frame, source_path) in star_schema_tables.items():
            self._prepare_column_lookup(table_name)
            columns = self._normalise_columns(frame.columns)
            frame = frame.rename(columns=dict(zip(frame.columns, columns)))
            frame.to_sql(table_name, self.connection, if_exists="replace", index=False)

            metadata = self._introspect_table(table_name, Path(source_path))
            self._table_metadata[table_name] = metadata

    def _build_star_schema(
        self, raw_frames: Mapping[str, pd.DataFrame]
    ) -> Dict[str, tuple[pd.DataFrame, str]]:
        """Transform the raw CSV exports into the dimensional model."""

        glassdoor = raw_frames["glassdoor_salary"].copy()
        oews = raw_frames["oews_salary"].copy()
        bls = raw_frames["bls_macro_indicators"].copy()

        # ------------------------------------------------------------------
        # Dimension tables
        # ------------------------------------------------------------------
        dim_time = self._build_time_dimension(bls)
        dim_company = self._build_company_dimension(glassdoor)
        dim_location = self._build_location_dimension(glassdoor)
        dim_occupation = self._build_occupation_dimension(glassdoor)
        dim_skill, bridge_job_skill = self._build_skill_bridge(glassdoor)

        # ------------------------------------------------------------------
        # Fact table
        # ------------------------------------------------------------------
        fact_job_market = self._build_fact_table(
            glassdoor,
            oews,
            bls,
            dim_time,
            dim_location,
            dim_company,
            dim_occupation,
        )

        return {
            "fact_job_market": (fact_job_market, "<derived:fact_job_market>"),
            "dim_time": (dim_time, "<derived:dim_time>"),
            "dim_location": (dim_location, "<derived:dim_location>"),
            "dim_company": (dim_company, "<derived:dim_company>"),
            "dim_occupation": (dim_occupation, "<derived:dim_occupation>"),
            "dim_skill": (dim_skill, "<derived:dim_skill>"),
            "bridge_job_skill": (
                bridge_job_skill,
                "<derived:bridge_job_skill>",
            ),
        }

    def _build_time_dimension(self, bls: pd.DataFrame) -> pd.DataFrame:
        frame = bls.copy()
        frame = frame[frame["period"].str.startswith("M")].copy()
        frame["month"] = frame["period"].str[1:].astype(int)
        frame["quarter"] = ((frame["month"] - 1) // 3) + 1
        time_dim = frame[["year", "month", "quarter"]].drop_duplicates().sort_values(
            ["year", "month"]
        )
        time_dim.insert(0, "time_key", range(1, len(time_dim) + 1))
        return time_dim.reset_index(drop=True)

    def _build_company_dimension(self, glassdoor: pd.DataFrame) -> pd.DataFrame:
        columns = {
            "Company Name": "company_name",
            "Industry": "industry",
            "Type of ownership": "ownership_type",
        }
        company_dim = glassdoor[list(columns.keys())].rename(columns=columns)
        company_dim = (
            company_dim.fillna({col: "Unknown" for col in company_dim.columns})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        company_dim.insert(0, "company_key", range(1, len(company_dim) + 1))
        return company_dim

    def _build_location_dimension(self, glassdoor: pd.DataFrame) -> pd.DataFrame:
        state_lookup = self._state_lookup()

        def split_location(value: Any) -> tuple[str, str]:
            if isinstance(value, str) and "," in value:
                city, state = [part.strip() for part in value.split(",", 1)]
                return city, state
            if isinstance(value, str):
                return value.strip(), ""
            return "Unknown", ""

        locations = glassdoor[["Location", "job_state"]].copy()
        locations["city"], locations["state_code_from_location"] = zip(
            *locations["Location"].map(split_location)
        )
        locations["state_code"] = locations["job_state"].fillna(
            locations["state_code_from_location"]
        )
        locations["state_code"] = locations["state_code"].str.upper().str.strip()
        locations["state_name"] = locations["state_code"].map(state_lookup).fillna(
            "Unknown"
        )
        dim_location = (
            locations[["state_name", "state_code", "city"]]
            .fillna({"city": "Unknown"})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        dim_location.insert(0, "location_key", range(1, len(dim_location) + 1))
        return dim_location

    def _build_occupation_dimension(self, glassdoor: pd.DataFrame) -> pd.DataFrame:
        occupation_series = glassdoor["Job Title"].fillna("Unknown")
        occupation_dim = pd.DataFrame(
            {
                "job_title": occupation_series,
                "occupation": occupation_series.map(self._derive_occupation),
            }
        ).drop_duplicates()
        occupation_dim.insert(0, "occupation_key", range(1, len(occupation_dim) + 1))
        return occupation_dim.reset_index(drop=True)

    def _build_skill_bridge(
        self, glassdoor: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        skill_columns = {
            "python_yn": "Python",
            "R_yn": "R",
            "spark": "Apache Spark",
            "aws": "AWS",
            "excel": "Microsoft Excel",
        }

        dim_skill = pd.DataFrame(
            {
                "skill_key": range(1, len(skill_columns) + 1),
                "skill_name": list(skill_columns.values()),
            }
        )

        skill_lookup = dim_skill.set_index("skill_name")["skill_key"].to_dict()

        bridge_rows: List[Dict[str, Any]] = []
        for job_id, (_, row) in enumerate(glassdoor.iterrows(), start=1):
            for column, skill_name in skill_columns.items():
                if column in row and pd.notna(row[column]) and int(row[column]) == 1:
                    bridge_rows.append(
                        {
                            "job_id": job_id,
                            "skill_key": int(skill_lookup[skill_name]),
                            "relevance_score": 1.0,
                        }
                    )

        bridge = pd.DataFrame(bridge_rows)
        return dim_skill, bridge

    def _build_fact_table(
        self,
        glassdoor: pd.DataFrame,
        oews: pd.DataFrame,
        bls: pd.DataFrame,
        dim_time: pd.DataFrame,
        dim_location: pd.DataFrame,
        dim_company: pd.DataFrame,
        dim_occupation: pd.DataFrame,
    ) -> pd.DataFrame:
        latest_time_key = dim_time["time_key"].max()
        state_lookup = self._state_lookup()

        oews_state = oews[oews["Location_Type"].str.lower() == "state"].copy()
        oews_state["state_code"] = (
            oews_state["Location"].map({v: k for k, v in state_lookup.items()})
        )
        oews_national = oews[oews["Location_Type"].str.lower() == "national"].copy()

        occupation_map = (
            dim_occupation.set_index("job_title")["occupation"].to_dict()
        )

        company_lookup = dim_company.set_index("company_name")["company_key"].to_dict()
        location_lookup = (
            dim_location.set_index(["state_code", "city"])["location_key"].to_dict()
        )
        occupation_lookup = (
            dim_occupation.set_index(["occupation", "job_title"])["occupation_key"].to_dict()
        )

        bls_numeric = bls.copy()
        bls_numeric["value"] = pd.to_numeric(bls_numeric["value"], errors="coerce")
        latest_bls = (
            bls_numeric.sort_values("date")
            .groupby("series_name")
            .tail(1)
            .set_index("series_name")["value"]
        )

        metrics_defaults = {
            "Average Hourly Earnings: Total Private": None,
            "Consumer Price Index (CPI-U All Items)": None,
            "Job Openings: Total Nonfarm": None,
            "Labor Force Participation Rate": None,
            "Unemployment Rate": None,
        }
        metrics: Dict[str, Optional[float]] = {}
        for name, default in metrics_defaults.items():
            value = latest_bls.get(name)
            metrics[name] = float(value) if pd.notna(value) else default

        rows: List[Dict[str, Any]] = []
        for idx, job in glassdoor.reset_index(drop=True).iterrows():
            job_id = idx + 1
            company_name = job.get("Company Name")
            if pd.isna(company_name):
                company_name = "Unknown"
            company_key = company_lookup.get(company_name, None)

            city, state_code = self._extract_location(job)
            location_key = location_lookup.get((state_code, city))

            job_title = job.get("Job Title", "Unknown")
            if pd.isna(job_title):
                job_title = "Unknown"
            occupation_name = occupation_map.get(job_title, "Unknown")
            occupation_key = occupation_lookup.get((occupation_name, job_title))

            oews_match = oews_state[
                (oews_state["Occupation"] == occupation_name)
                & (oews_state["state_code"] == state_code)
            ]
            if oews_match.empty:
                oews_match = oews_national[oews_national["Occupation"] == occupation_name]

            median_salary = oews_match["Median_Salary"].astype(float).mean()
            median_salary = float(median_salary) if pd.notna(median_salary) else None
            avg_salary_oews = oews_match["Average_Salary"].astype(float).mean()
            avg_salary_oews = (
                float(avg_salary_oews) if pd.notna(avg_salary_oews) else None
            )
            total_employment = oews_match["Total_Jobs"].astype(float).mean()
            total_employment = (
                float(total_employment) if pd.notna(total_employment) else None
            )

            avg_salary_value = job.get("avg_salary")
            avg_salary = (
                float(avg_salary_value) * 1000.0
                if pd.notna(avg_salary_value)
                else avg_salary_oews
            )

            rows.append(
                {
                    "job_id": job_id,
                    "avg_salary": avg_salary,
                    "median_salary": median_salary,
                    "total_employment": total_employment,
                    "location_quotient": None,
                    "company_rating": float(job.get("Rating"))
                    if pd.notna(job.get("Rating"))
                    else None,
                    "hourly_earnings": metrics["Average Hourly Earnings: Total Private"],
                    "cpi": metrics["Consumer Price Index (CPI-U All Items)"],
                    "job_openings": metrics["Job Openings: Total Nonfarm"],
                    "labor_force_rate": metrics["Labor Force Participation Rate"],
                    "unemployment_rate": metrics["Unemployment Rate"],
                    "time_key": latest_time_key,
                    "location_key": location_key,
                    "company_key": company_key,
                    "occupation_key": occupation_key,
                }
            )

        return pd.DataFrame(rows)

    def _derive_occupation(self, job_title: str) -> str:
        title = job_title.lower()
        if "data scientist" in title:
            return "Data Scientists"
        if "machine learning" in title:
            return "Machine Learning Specialists"
        if "data engineer" in title:
            return "Database Architects"
        if "analyst" in title:
            return "Data Analysts"
        if "manager" in title:
            return "Information Systems Managers"
        return "Other Occupations"

    def _extract_location(self, job_row: pd.Series) -> tuple[str, str]:
        state_code = str(job_row.get("job_state", "")).upper().strip()
        location = job_row.get("Location", "")
        city = "Unknown"
        if isinstance(location, str) and "," in location:
            city_part, state_part = [part.strip() for part in location.split(",", 1)]
            city = city_part or "Unknown"
            if not state_code:
                state_code = state_part.upper()
        elif isinstance(location, str) and location:
            city = location.strip()
        if not state_code:
            state_code = ""
        return city, state_code

    def _state_lookup(self) -> Dict[str, str]:
        return {
            "AL": "Alabama",
            "AK": "Alaska",
            "AZ": "Arizona",
            "AR": "Arkansas",
            "CA": "California",
            "CO": "Colorado",
            "CT": "Connecticut",
            "DE": "Delaware",
            "FL": "Florida",
            "GA": "Georgia",
            "HI": "Hawaii",
            "ID": "Idaho",
            "IL": "Illinois",
            "IN": "Indiana",
            "IA": "Iowa",
            "KS": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "ME": "Maine",
            "MD": "Maryland",
            "MA": "Massachusetts",
            "MI": "Michigan",
            "MN": "Minnesota",
            "MS": "Mississippi",
            "MO": "Missouri",
            "MT": "Montana",
            "NE": "Nebraska",
            "NV": "Nevada",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NY": "New York",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "OR": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TN": "Tennessee",
            "TX": "Texas",
            "UT": "Utah",
            "VT": "Vermont",
            "VA": "Virginia",
            "WA": "Washington",
            "WV": "West Virginia",
            "WI": "Wisconsin",
            "WY": "Wyoming",
            "DC": "District of Columbia",
        }


def build_call_function_spec(schema_text: str) -> Dict[str, Any]:
    """Return a JSON schema describing the query function for tool-calling."""

    return {
        "name": "query_labor_market_database",
        "description": (
            "Run a read-only SQL query against the curated labor market database. "
            "The database is derived from BLS macro indicators, BLS Occupational "
            "Employment and Wage Statistics (OEWS), and Glassdoor salary "
            "postings. Use this to answer quantitative questions that require "
            "precise values. Only SELECT/CTE queries are allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "SQL query to execute. Must be a single read-only SELECT "
                        "or WITH statement targeting the available tables.\n\n"
                        f"Available tables and columns:\n{schema_text}"
                    ),
                },
                "max_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": (
                        "Optional upper bound on the number of rows to return. "
                        "Defaults to 50 and is capped at 200."
                    ),
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    }


def create_database_with_spec(
    data_dir: Path | str = Path("data"),
) -> tuple[LaborMarketDatabase, Dict[str, Any]]:
    """Convenience helper returning the database instance and function spec."""

    database = LaborMarketDatabase(data_dir=data_dir)
    schema_text = database.get_schema_as_text()
    function_spec = build_call_function_spec(schema_text)
    return database, function_spec


__all__ = [
    "ColumnMetadata",
    "TableMetadata",
    "LaborMarketDatabase",
    "build_call_function_spec",
    "create_database_with_spec",
]
