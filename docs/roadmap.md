# Roadmap

These milestones describe proposed outcomes. The repository contains design documents, environment configuration, a synthetic generator, and local CSV/XLSX ingestion with behavioral tests. The forecasting milestones have not been implemented.

## 0. Establish the design and development environment

- [x] Write the project purpose and initial scope.
- [x] Describe application, adapter, and shared-pipeline responsibilities.
- [x] Preserve the Bayesian and non-Bayesian checking approach.
- [x] Create a Git-ready repository layout and task/review templates.
- [x] Select CPython 3.14 with a 3.14.10 development pin.
- [x] Declare project metadata and the supported interpreter range in `pyproject.toml`.
- [x] Provide a uv lockfile and a pip-compatible dependency export.
- [x] Document both venv/pip and uv setup, with local environments excluded from Git.
- [ ] Recreate and verify the pinned environment on the development machine.
- [ ] Agree on the first runtime modules and development tools, then lock and verify their versions.
- [ ] Agree on the first synthetic experiment and its observable acceptance criteria.

## 1. Small synthetic experiment

Seeded synthetic data → training-mean baseline → rolling backtest → saved results → one chart.

The initial implementation should demonstrate correct target alignment, no future information in prediction, explicit failed folds, and reconstruction from preserved inputs/configuration. A deliberately invalid output or missing required check must be visible. The report reads saved results without fitting.

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

## Next design slice

Review the local implementation of [Local ingestion: first slice and later enhancements](ingestion.md): one local-file ingestion function in a small Python module, a receipt describing the completed copy, and a companion CSV/XLSX table reader with worksheet selection. Behavioral tests cover temporary files and a multi-sheet workbook. This slice is ready for code review and remains independent of snapshot registration and model execution.

The [2026-11-09 decision](decisions.md#ingestion-first-preservation-and-execution-later--2026-11-09) selects git-annex as the direction for later preservation and defers DVC or another dependency/stage/execution tool. Do not implement a custom snapshot storage system as a prerequisite for the local ingestion function. Preserve the [data-versioning requirements](data-versioning.md) for later integration.

Then define the smallest baseline experiment together: target, issuance/cutoff, minimum history, origins, point statistic, scores, required checks, and expected artifacts. Permanent input references and training selection remain runner requirements to address when that integration is designed. Select and verify packages as implementation needs become concrete.

Turn each deliverable into a small GitHub issue using the repository's task template. This roadmap is not a commitment to implement all possible extensions.
