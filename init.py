"""Starter baseline for fraud-risk modeling and decision-policy experiments.

Run from the repository root with::

    python init.py

The default dataset path matches the local PaySim CSV. This script is an
educational baseline: it does not make production decisions, and its metrics
must not be treated as evidence of real-world financial impact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
LABEL_NAMES = {"fraud", "isfraud", "is_fraud", "fraud_flag", "label", "target", "class"}
TIME_NAMES = ("step", "transaction_time", "timestamp", "datetime", "date")


def find_target(columns: list[str], requested: str | None) -> str:
    """Find a likely binary fraud label or validate the explicitly requested one."""
    if requested:
        if requested not in columns:
            raise ValueError(f"Target column {requested!r} is not in the dataset.")
        return requested

    for column in columns:
        normalized = column.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in LABEL_NAMES:
            return column
    raise ValueError(
        "Could not identify a fraud label. Pass its column name with --target."
    )


def find_time_column(columns: list[str], requested: str | None) -> str | None:
    """Prefer a known event-time field for a forward-in-time holdout."""
    if requested:
        if requested not in columns:
            raise ValueError(f"Time column {requested!r} is not in the dataset.")
        return requested
    normalized = {column.lower().replace(" ", "_"): column for column in columns}
    return next((normalized[name] for name in TIME_NAMES if name in normalized), None)


def make_temporal_split(
    frame: pd.DataFrame, time_column: str | None
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Reserve the latest 20% of observed time for evaluation when possible."""
    if time_column is not None:
        values = frame[time_column]
        numeric_time = pd.to_numeric(values, errors="coerce")
        if numeric_time.notna().mean() >= 0.8:
            order = numeric_time
        else:
            order = pd.to_datetime(values, errors="coerce", utc=True)

        valid = order.notna()
        if valid.sum() >= 10 and order[valid].nunique() >= 2:
            cutoff = order[valid].quantile(0.8)
            train_mask = valid & (order < cutoff)
            test_mask = valid & (order >= cutoff)
            train, test = frame.loc[train_mask].copy(), frame.loc[test_mask].copy()
            if train.empty or test.empty:
                raise ValueError("Could not split the data around the selected time column.")
            return train, test, f"time split on {time_column!r} at {cutoff}"

    # Fallback for datasets without usable time: sort-independent seeded split.
    shuffled = frame.sample(frac=1, random_state=RANDOM_STATE)
    cut = max(1, int(len(shuffled) * 0.8))
    return shuffled.iloc[:cut].copy(), shuffled.iloc[cut:].copy(), "random 80/20 split (no usable time column)"


def bounded_sample(frame: pd.DataFrame, target: str, limit: int | None) -> pd.DataFrame:
    """Bound runtime while sampling each label in proportion to its prevalence."""
    if limit is None or len(frame) <= limit:
        return frame

    groups = [group for _, group in frame.groupby(target, dropna=False, sort=False)]
    exact_quotas = [limit * len(group) / len(frame) for group in groups]
    quotas = [int(quota) for quota in exact_quotas]
    remainder = limit - sum(quotas)
    order = sorted(
        range(len(groups)), key=lambda index: exact_quotas[index] - quotas[index], reverse=True
    )
    for index in order[:remainder]:
        quotas[index] += 1

    pieces: list[pd.DataFrame] = []
    for group, take in zip(groups, quotas):
        if take == 0:
            continue
        pieces.append(group.sample(n=take, random_state=RANDOM_STATE))
    return pd.concat(pieces).sample(frac=1, random_state=RANDOM_STATE)


def build_model(features: pd.DataFrame) -> Pipeline:
    """Create an imbalanced-label logistic-regression baseline."""
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def summarize_policy(
    labels: np.ndarray, scores: np.ndarray, threshold: float, review_capacity: float
) -> dict[str, Any]:
    """Report simple review/decline rates without assuming business costs."""
    predicted_fraud = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted_fraud, labels=[0, 1]).ravel()

    review_count = max(1, math.ceil(len(scores) * review_capacity))
    review_indexes = np.argsort(scores)[-review_count:]
    reviewed_fraud = int(labels[review_indexes].sum())
    total_fraud = int(labels.sum())

    return {
        "threshold": threshold,
        "decline_rate": float(predicted_fraud.mean()),
        "precision_at_threshold": float(precision_score(labels, predicted_fraud, zero_division=0)),
        "recall_at_threshold": float(recall_score(labels, predicted_fraud, zero_division=0)),
        "f1_at_threshold": float(f1_score(labels, predicted_fraud, zero_division=0)),
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "review_capacity": review_capacity,
        "reviewed_transactions": review_count,
        "fraud_captured_in_review_queue": reviewed_fraud,
        "review_queue_precision": float(reviewed_fraud / review_count),
        "review_queue_fraud_recall": float(reviewed_fraud / total_fraud) if total_fraud else 0.0,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    default_data = repo_root / "data/raw/PS_20174392719_1491204439457_log.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_data, help="Local CSV or Parquet file")
    parser.add_argument("--target", help="Fraud label column; auto-detected when possible")
    parser.add_argument("--time-column", help="Event-time column used for a forward-in-time holdout")
    parser.add_argument("--max-train-rows", type=int, default=500_000, help="Maximum training rows; use 0 for all rows")
    parser.add_argument("--max-test-rows", type=int, default=200_000, help="Maximum evaluation rows; use 0 for all rows")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for the illustrative decline policy")
    parser.add_argument("--review-capacity", type=float, default=0.02, help="Share of transactions available for manual review")
    parser.add_argument("--output", type=Path, default=repo_root / "reports/baseline_metrics.json")
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if not 0 < args.review_capacity <= 1:
        parser.error("--review-capacity must be greater than 0 and at most 1")
    if args.max_train_rows < 0 or args.max_test_rows < 0:
        parser.error("row limits must be non-negative; use 0 for no limit")
    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")

    suffix = args.dataset.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(args.dataset, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(args.dataset)
    else:
        parser.error("Dataset must be a CSV or Parquet file")

    target = find_target(frame.columns.tolist(), args.target)
    time_column = find_time_column(frame.columns.tolist(), args.time_column)
    frame = frame.loc[frame[target].notna()].copy()
    labels = pd.to_numeric(frame[target], errors="raise")
    unique_labels = set(labels.unique().tolist())
    if not unique_labels.issubset({0, 1}) or len(unique_labels) != 2:
        parser.error(f"Target {target!r} must contain both binary labels 0 and 1; found {unique_labels}")
    frame[target] = labels.astype("int8")

    train, test, split_description = make_temporal_split(frame, time_column)
    train_limit = args.max_train_rows or None
    test_limit = args.max_test_rows or None
    train = bounded_sample(train, target, train_limit)
    test = bounded_sample(test, target, test_limit)

    # Exclude outcome-derived or identity-like columns from this first baseline.
    excluded = {target}
    for column in frame.columns:
        normalized = column.lower().replace("_", "").replace(" ", "")
        if normalized in {"isflaggedfraud", "newbalanceorig", "newbalancedest"}:
            excluded.add(column)
        elif any(token in normalized for token in ("nameorig", "namedest", "customerid", "accountid", "transactionid")):
            excluded.add(column)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        parser.error("No feature columns remain after excluding the target and identifier/leakage fields")

    x_train, y_train = train[feature_columns], train[target].to_numpy()
    x_test, y_test = test[feature_columns], test[target].to_numpy()
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        parser.error("Both train and evaluation partitions need fraud and non-fraud examples")

    model = build_model(x_train)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]

    report = {
        "dataset": str(args.dataset),
        "target": target,
        "time_column": time_column,
        "split": split_description,
        "rows_used": {"train": int(len(train)), "evaluation": int(len(test))},
        "fraud_rate": {
            "train": float(y_train.mean()),
            "evaluation": float(y_test.mean()),
        },
        "features_used": feature_columns,
        "excluded_columns": sorted(excluded),
        "next_project_ideas": [
            "Compare the logistic-regression baseline with a tree-based model using the same temporal split.",
            "Calibrate risk scores and compare review/decline policies at fixed approval and review-capacity levels.",
            "Add cost-sensitive evaluation after documenting fraud loss, false-decline, and review costs.",
            "Audit feature leakage, subgroup performance, and score drift before considering deployment.",
        ],
        "evaluation": {
            "average_precision": float(average_precision_score(y_test, scores)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "decision_policy": summarize_policy(
                y_test, scores, args.threshold, args.review_capacity
            ),
        },
        "caveats": [
            "Logistic regression is a baseline, not a production fraud system.",
            "The decision threshold and review capacity are examples; choose them using business costs and operational constraints.",
            "Validate feature availability at decision time; post-transaction fields are excluded in this starter.",
            "Use a documented data source and license, temporal validation, calibration, and fairness checks before drawing conclusions.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nMetrics written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
