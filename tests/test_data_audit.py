from pathlib import Path

import pandas as pd
import pytest

from fraud_decisioning.data_audit import load_dataset, render_markdown, summarize_dataframe


def test_summary_detects_target_and_timestamp_hints():
    frame = pd.DataFrame(
        {
            "transaction_time": ["2026-01-01", "2026-01-02", "2026-01-02"],
            "amount": [10.0, 20.0, 20.0],
            "isFraud": [0, 1, 1],
        }
    )

    summary = summarize_dataframe(frame, source="test.csv")

    assert summary["rows_loaded"] == 3
    assert summary["column_count"] == 3
    assert summary["duplicate_rows"] == 1
    assert summary["target_column_candidate"] == "isFraud"
    assert summary["target_distribution_loaded_rows"] == {"1": 2, "0": 1}
    assert summary["timestamp_column_candidates"] == ["transaction_time"]


def test_load_csv_with_sample_limit(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"amount": [1, 2, 3], "fraud": [0, 0, 1]}).to_csv(path, index=False)

    frame = load_dataset(path, sample_rows=2)

    assert len(frame) == 2
    assert list(frame.columns) == ["amount", "fraud"]


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataset(path)


def test_markdown_report_includes_caution():
    summary = summarize_dataframe(pd.DataFrame({"fraud": [0, 1]}))

    markdown = render_markdown(summary)

    assert "Dataset audit" in markdown
    assert "Confirm the target semantics" in markdown
