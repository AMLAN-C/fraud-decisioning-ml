# Project plan and milestones

## Objective

Build a reproducible fraud decisioning prototype that connects model quality to real decision outcomes. The key question is not merely whether a model ranks fraud well, but whether a decision policy improves fraud outcomes at a clearly stated approval rate, review capacity, or cost assumption.

## Milestones

### M0 — Repository and environment
- [x] Create a clean repository scaffold.
- [x] Add data exclusions and a starter data-audit script.
- [x] Create the GitHub repository and connect the local project.
- [x] Confirm Python environment and install requirements.

### M1 — Dataset audit and problem framing
- [x] Record source URL, licence/terms, and dataset limitations.
- [x] Inspect row/column counts, data types, missingness, duplicates, and label distribution.
- [x] Confirm what the positive label means and when it becomes observable.
- [x] Identify the event timestamp, amount field, entity identifiers, and potential post-outcome leakage.
- [x] Write a compact data dictionary and a one-paragraph modelling question.

**Exit criteria:** target and prediction time are unambiguous; the split strategy is documented; known leakage risks are listed.

### M2 — Data quality and baseline
- [x] Establish reproducible cleaning and validation rules.
- [x] Build a simple baseline (e.g., logistic regression or a transparent heuristic).
- [x] Split data chronologically when a reliable event timestamp exists.
- [x] Save baseline metrics with the evaluation set and assumptions clearly stated.

### M3 — Feature engineering
- [ ] Create transaction-level features available at decision time.
- [ ] Add safe velocity/behavior features only when timestamps and entity identifiers support them.
- [ ] Add tests for feature windows, ordering, null handling, and leakage prevention.

### M4 — Model comparison and calibration
- [ ] Compare a simple baseline with a gradient-boosted tree model.
- [ ] Evaluate average precision, precision/recall at operating points, calibration, and temporal performance.
- [ ] Examine subgroup/cohort performance and data coverage.

### M5 — Decision and economic optimization
- [ ] Implement configurable approve / review-or-step-up / decline thresholds.
- [ ] Document the cost assumptions for fraud losses, reviews, and false declines.
- [ ] Compare policies at matched approval rates or review capacity.
- [ ] Quantify incremental impact against a clearly specified baseline.

### M6 — Monitoring and communication
- [ ] Create a self-serve dashboard or concise report.
- [ ] Add data-quality and drift checks.
- [ ] Write a model card, limitations, and client-style executive summary.

### M7 — Optional deployment extension
- [ ] Package inference separately from training.
- [ ] Add an API, input validation, unit tests, and latency measurement.
- [ ] Document deployment assumptions; do not represent this prototype as a production control.

## Guardrails

- Use only public or synthetic data that may legally be used for this project.
- Never commit proprietary data, personal information, credentials, or employer code.
- Avoid random splits when they would leak time or entity history across train/test.
- Do not infer business savings without explicit assumptions and a credible counterfactual.
- Record failed experiments and limitations as well as the best result.
