# Roadmap

Updated: 2026-11-13.

The repository now has synthetic data generation, local CSV/XLSX ingestion, verified single-file snapshots, saved ingestion receipts, explicit tags, and the first runnable forecasting workflow. The training-mean baseline uses daily expanding histories and saves predictions, scores, required checks, failures, and a pending review. A separate report draws a chart from saved artifacts. Broader model support remains proposed.

## Resume here

The current modeling slice is on `feature/first-forecast-workflow`. It builds on the completed ingestion/snapshot/tag workflow and receipt enhancement; it does not replace them.

1. **Review the working baseline.** Follow [the forecasting walkthrough](forecasting.md), then read `models.py`, `backtest.py`, and `reporting.py`. The first prediction uses 30 days to predict day 31; remaining predictions expand the training history one day at a time.
2. **Inspect the saved evidence.** The runner resolves a supplied tag once, pins the dataset and snapshot ID, validates the input, and automatically saves forecasts, training coordinates, scores, required checks, and a pending review. The script also draws a chart; library callers invoke reporting separately. Input preparation and human interpretation remain explicit caller actions.
3. **Review before extending.** Decide whether the request/model boundary and saved files are clear enough to keep. The first run is development evidence, with immediate synthetic availability and no final holdout. Define a real-data task's timing and final evaluation boundaries before adding it; add Bayesian diagnostics when the Bayesian adapter is selected. Broader versioning and a general plugin/check framework are not prerequisites.

Do not rebuild the completed snapshot/tag functions or introduce a larger storage framework as the next step. Keep work in small pieces the user can read and understand. git-annex remains optional; DVC or another dependency/stage/execution tool is deferred.

## Verification at this checkpoint

- The suite contains 78 behavioral tests: 17 ingestion, 27 snapshot, 13 tag, and 21 forecasting/reporting checks. Forecast checks include known answers, future-value isolation, immutable history, input/target validation, tag moves/store relocation, invalid outputs and failed origins, scoring overflow, artifact preservation, and reporting without model calls.
- The mean-shift example completed 170/170 one-day forecasts from 200 observations with the default minimum history (MAE 2.4197, RMSE 2.9462). The saved-result chart shows the baseline adapting slowly after the mean shift. Its outputs stay local under `runs/`; human review remains pending.
- matplotlib is added for saved-result PNG charts; dependency declarations, lockfile, and pip export are updated together. Rerun the suite after code changes:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The supported local scope is one file per snapshot, one writer at a time, local writable storage, and integrity verification on retrieval. The first backtest adds daily CSV validation and fixed required checks; broader quality evidence/eligibility, configurable check plans, multi-file identity, processing lineage, concurrent writers, remote storage, and crash recovery remain future work. Local data and records are ignored by Git and are not uploaded with code.

## 0. Establish the design and development environment

- [x] Write the project purpose and initial scope.
- [x] Describe application, adapter, and shared-pipeline responsibilities.
- [x] Preserve the Bayesian and non-Bayesian checking approach.
- [x] Create a Git-ready repository layout and task/review templates.
- [x] Select CPython 3.14 with a 3.14.10 development pin.
- [x] Declare project metadata and the supported interpreter range in `pyproject.toml`.
- [x] Provide a uv lockfile and a pip-compatible dependency export.
- [x] Document both venv/pip and uv setup, with local environments excluded from Git.
- [x] Recreate and verify the pinned environment on the development machine (CPython 3.14.10).
- [x] Select and lock the first runtime packages (pandas/openpyxl) and use standard-library unittest for behavioral checks.
- [x] Define the first synthetic baseline and verify its observable acceptance criteria.

## Data foundations — implemented locally

- [x] Reproducible synthetic generator and two starting CSV datasets.
- [x] Local-file copying with an acquisition receipt and overwrite protection.
- [x] CSV/XLSX table reading with worksheet and header selection.
- [x] Single-file SHA-256 snapshot registration, preserved bytes, and JSON manifests.
- [x] Separate registration records and optional notes, including repeated identical acquisitions.
- [x] Optional original ingestion receipt saved with each registration, including source and copy completion time.
- [x] Retrieval with integrity checks and storage-location-independent dataset/ID references.
- [x] Snapshot inventory and chronological acquisition-history listing.
- [x] Explicit dataset-scoped tag assignment, resolution, current-tag listing, and assignment history.
- [x] Caller context and optional tag notes; registration never automatically moves tags.
- [x] Code comments, descriptive names, an everyday function reference, and behavioral tests.
- [x] Provide code walkthroughs, explanations, and the source-directory reference for user review.
- [x] Prepare the reviewed snapshot/tagging slice for the user's requested commit and push.

## 1. Small synthetic experiment

Seeded synthetic data → training-mean baseline → rolling backtest → saved results → one chart.

Implemented in [the first forecasting workflow](forecasting.md):

- [x] Daily one-step task with explicit initial history and availability assumptions.
- [x] Training-mean model behind an immutable history/target-time request.
- [x] Tag resolution once and permanent dataset/snapshot pinning.
- [x] Expanding-history backtest, frozen predictions, MAE/RMSE, and coverage counts.
- [x] Required input/alignment/output/coverage outcomes and visible failed origins.
- [x] JSON/CSV evidence, pending review record, and chart from saved results.
- [x] Behavioral checks for future-value isolation, reproduction, invalid output, and reporting.
- [ ] Human review of the harness and the resulting evidence.

The current checks are fixed in the runner. A configurable check plan and missing specialized-check handling will be designed when another model requires them; they are not claimed as implemented.

Review the harness before adding model complexity: keep, simplify, or restart using the evidence preserved from this slice.

## 2. One real-data application

Add one source and one numerical target with explicit timing, calendar, snapshot, and availability assumptions. A next-session market-return task is a candidate; the instrument, source, and dates remain undecided.

Hold the task and comparison protocol fixed before interpreting model improvements. Forecasting performance and trading performance are separate evaluations.

## 3. First Bayesian adapter

Add a local-level BDLM through the same runner and storage path. Begin with fixed observation/evolution variances and explicit initial-state assumptions. Document uncertainty omitted by that simplification.

Include prior predictive simulation, synthetic inference verification, fitted-model criticism/sensitivity, and compatible held-out scores. Compare with the baseline without rewriting shared execution. MCMC diagnostics are inapplicable to an analytic filter.

## 4. Prove the boundary with a contrasting model

Add a point-only model such as gradient boosting. Supply its feature, fitting, and model-criticism checks through the same check boundary. Keep uncertainty evaluation explicitly inapplicable unless supported outputs are added.

## Later, when a need is demonstrated

Multiple horizons, richer Bayesian models, an operational-data application, a run index, dashboards, scheduled execution, and monitoring policies. Each extension needs a stated problem and evidence that it helps.

## Further versioning enhancements — not selected for implementation

Local ingestion receipts are now carried into registration. Remote source-specific context, processing lineage, multi-file collections, quality-check evidence, additional source functions, and storage/recovery improvements remain possible later slices. Implement them only when a concrete need is selected; the list is not a requirement to finish every feature before learning from a baseline.

See [the current snapshot/tag behavior](snapshots.md), [the decision log](decisions.md), and [the broader versioning requirements and historical plan](data-versioning.md). The roadmap's resume section is the current continuation order.

Turn each deliverable into a small GitHub issue using the repository's task template. This roadmap is not a commitment to implement all possible extensions.
