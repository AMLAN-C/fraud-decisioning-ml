# Notebooks

Use notebooks for focused exploration and communicate reusable logic back into `src/fraud_decisioning/` once it stabilizes.

Suggested sequence:

1. `01_dataset_audit.ipynb` — schema, labels, missingness, time range, duplicates, and leakage questions.
2. `02_baseline_model.ipynb` — baseline benchmark with a leakage-safe split.
3. `03_model_comparison.ipynb` — feature/model comparison and calibration.
4. `04_decision_simulation.ipynb` — approval, review, decline, and expected-loss trade-offs.

Avoid committing notebooks containing raw data, credentials, or large outputs. Do not start modelling until the label and prediction point are understood.
