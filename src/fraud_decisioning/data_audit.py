"""Lightweight, repeatable first-pass audit for CSV and Parquet datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_CANDIDATES = {
    "fraud",
    "isfraud",
    "is_fraud",
    "fraud_flag",
    "is_fraudulent",
    "isfraudulent",
    "label",
    "target",
    "class",
}
TIME_HINTS = ("time", "date", "timestamp", "datetime", "_at", "_ts")


def load_dataset(path: str | Path, sample_rows: int | None = None) -> pd.DataFrame:
    """Load a CSV or Parquet file; optional row limit is for schema exploration only."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path, nrows=sample_rows, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(dataset_path)
        return frame.head(sample_rows) if sample_rows is not None else frame

    raise ValueError(
        f"Unsupported file type '{suffix}'. This starter supports .csv, .parquet, and .pq."
    )


def _candidate_target(columns: list[str]) -> str | None:
    for column in columns:
        normalized = column.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in LABEL_CANDIDATES:
            return column
    return None


def summarize_dataframe(frame: pd.DataFrame, source: str = "in-memory dataframe") -> dict[str, Any]:
    """Return JSON-serializable schema, quality, target, and timestamp hints."""
    total_rows = int(len(frame))
    columns = [str(column) for column in frame.columns]
    target_column = _candidate_target(columns)

    inventory: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        missing_count = int(series.isna().sum())
        try:
            unique_count = int(series.nunique(dropna=True))
        except TypeError:
            unique_count = int(series.astype("string").nunique(dropna=True))
        inventory.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_pct": round((missing_count / total_rows * 100) if total_rows else 0.0, 4),
                "unique_count": unique_count,
            }
        )

    timestamp_candidates = [
        column
        for column in columns
        if any(hint in column.lower() for hint in TIME_HINTS)
    ]

    target_distribution: dict[str, int] | None = None
    if target_column is not None:
        values = frame[target_column].astype("string").fillna("<NULL>").value_counts(dropna=False)
        target_distribution = {str(key): int(value) for key, value in values.head(50).items()}

    return {
        "source": source,
        "rows_loaded": total_rows,
        "column_count": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()) if total_rows else 0,
        "target_column_candidate": target_column,
        "target_distribution_loaded_rows": target_distribution,
        "timestamp_column_candidates": timestamp_candidates,
        "columns": inventory,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Format a concise audit summary for console or a report file."""
    lines = [
        "# Dataset audit",
        "",
        f"- **Source:** `{summary['source']}`",
        f"- **Rows loaded:** {summary['rows_loaded']:,}",
        f"- **Columns:** {summary['column_count']:,}",
        f"- **Duplicate rows in loaded data:** {summary['duplicate_rows']:,}",
        f"- **Possible target column:** `{summary['target_column_candidate'] or 'not detected — confirm manually'}`",
        f"- **Timestamp candidates:** {', '.join(f'`{x}`' for x in summary['timestamp_column_candidates']) or 'none detected by name'}",
        "",
        "## Columns",
        "",
        "| Column | Type | Missing | Missing % | Unique values |",
        "|---|---|---:|---:|---:|",
    ]
    for column in summary["columns"]:
        lines.append(
            f"| `{column['column']}` | `{column['dtype']}` | "
            f"{column['missing_count']:,} | {column['missing_pct']:.4f}% | {column['unique_count']:,} |"
        )

    distribution = summary.get("target_distribution_loaded_rows")
    if distribution is not None:
        lines.extend(["", "## Candidate target distribution (loaded rows)", ""])
        lines.extend(f"- `{label}`: {count:,}" for label, count in distribution.items())
    lines.extend(
        [
            "",
            "> This is a descriptive first pass, not a cleaning or modelling decision. Confirm the target semantics, prediction time, sampling method, and leakage risks before training a model.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to a .csv or .parquet file")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row limit for a quick schema check. CSV reads the first N rows; this may be biased.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the Markdown audit report.",
    )
    args = parser.parse_args()

    if args.sample_rows is not None and args.sample_rows <= 0:
        parser.error("--sample-rows must be a positive integer")

    try:
        frame = load_dataset(args.dataset, args.sample_rows)
        summary = summarize_dataframe(frame, source=str(Path(args.dataset)))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    report = render_markdown(summary)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
