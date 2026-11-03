# Fraud Decisioning & Model Optimization

A portfolio project that builds an end-to-end fraud-risk workflow: dataset understanding, leakage-safe feature engineering, supervised risk scoring, decision-threshold optimization, and measurable trade-offs between fraud prevention and legitimate-customer approvals.

> **Portfolio / research project:** this repository is not a production fraud service and must not be used to make real financial decisions. Use public or synthetic data only. Never commit Varo or other employer data, credentials, customer information, or licensed datasets whose terms prohibit redistribution.

## Problem statement

Can a transaction-risk model and a configurable decision policy reduce expected fraud loss while preserving the approval rate for legitimate transactions and respecting a review/verification capacity?

## Planned workflow

1. **Understand the data** — identify the target, time column, transaction amount, feature types, data quality issues, label prevalence, and dataset limitations.
2. **Build a baseline** — create a simple benchmark and define leakage-safe train/validation/test splits.
3. **Engineer features** — develop historical, velocity, and behavioral features using only information available at decision time.
4. **Train and validate** — compare an interpretable baseline with a tree-based model; evaluate temporal generalization and calibration.
5. **Simulate decisions** — compare approve, review/step-up, and decline policies across thresholds and capacity constraints.
6. **Measure outcomes** — report fraud recall, precision, false-positive rate, approval rate, expected loss, and incremental value versus a baseline.
7. **Prepare for production** — add reproducible pipelines, tests, model/feature monitoring, and a scoring API if justified by the findings.

## Repository layout

```text
.
├── config/                 # Project/model configuration
├── data/
│   ├── raw/                # Local source data (git-ignored)
│   └── processed/          # Local derived data (git-ignored)
├── docs/                   # Project plan and data dictionary
├── notebooks/              # Exploratory notebooks and experiments
├── reports/figures/        # Generated plots (keep only shareable outputs)
├── src/fraud_decisioning/  # Reusable project code
├── tests/                  # Automated tests
├── .github/workflows/      # CI checks
├── requirements.txt
└── README.md
```

## Quickstart

Requires Python 3.11 or 3.12.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Put your downloaded dataset under data/raw/ (the file is intentionally not tracked by Git)
# 4. Audit a CSV or Parquet dataset
python src/fraud_decisioning/data_audit.py data/raw/YOUR_DATASET.csv
```

Optional: `--sample-rows 100000` limits a CSV audit to its first 100,000 rows. Use this only for a quick schema check: the first rows may be time-ordered or otherwise unrepresentative, so do not use a sample to report overall fraud prevalence.

To run unit tests:

```bash
pytest
```

## Evaluation principles

- Prefer time-based splitting when the dataset contains a reliable event timestamp.
- Do not use post-outcome fields or information unavailable at decision time as model features.
- Use precision-recall metrics for highly imbalanced labels; do not rely on accuracy alone.
- Compare strategies at matched approval rates, review capacity, or expected cost where appropriate.
- Treat results on public benchmarks as dataset-specific evidence, not a claim of production savings.
- Document assumptions, limitations, and the impact on legitimate customers.

## Current status

**Milestone 2 — first baseline.** The PaySim dataset has been audited and documented. `init.py` applies explicit target validation, leakage exclusions, training-only preprocessing, and an exploratory logistic-regression baseline with a time-based holdout. It writes metrics to `reports/baseline_metrics.json`. Next, build and test transaction-history features that are safe at decision time.

## Data source

The project uses the synthetic [PaySim dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), listed as CC BY-SA 4.0. See [`docs/data_dictionary.md`](docs/data_dictionary.md) for attribution, audit results, feature notes, and limitations. The raw CSV stays local and is excluded from Git; download it separately into `data/raw/`.
