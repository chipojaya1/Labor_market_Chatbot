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
        for table_name, filename in self.table_files.items():
            csv_path = self.data_dir / filename
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Expected dataset not found: {csv_path}"
                )

            frame = pd.read_csv(csv_path)
            self._prepare_column_lookup(table_name)
            columns = self._normalise_columns(frame.columns)
            frame.columns = columns
            frame.to_sql(table_name, self.connection, if_exists="replace", index=False)

            metadata = self._introspect_table(table_name, csv_path)
            self._table_metadata[table_name] = metadata


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
