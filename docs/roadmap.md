# Roadmap

Updated: 2026-11-09.

The repository now has synthetic data generation, local CSV/XLSX ingestion, single-file snapshots, registration history and notes, optional ingestion receipts saved with registrations, and explicit dataset-scoped tags. The forecasting milestones below remain proposed outcomes; no model runner or completed experiment exists.

## Resume here

The current checkpoint includes ingestion, snapshots, inspection functions, notes, tags, tests, clearer code comments/names, and the function reference. Ingestion was first published in `685b7fb` (`Add local file ingestion and CSV/XLSX table reading`); snapshots and tags followed in `41a4120`. The user then selected carrying the ingestion receipt into snapshot registration. That local change adds the optional `ingestion_receipt` argument and saves the original copy's source, destination, UTC completion time, and byte count with each registration.

1. **Resume from the implemented checkpoint.** Start with the [function reference beside the code](../src/model_pipelining/README.md), then `ingest.py`, `snapshots.py`, and `tags.py`. The guide explains return values, the complete workflow, tag moves, permanent references, and the saved files. Keep future changes small and readable, with reviewable code and useful comments.
2. **Review the implemented receipt enhancement.** Pass `ingestion_receipt=receipt` alongside `receipt.destination`. Receipt context belongs to the registration, independently of the snapshot ID and manifest. Destination and byte count must match; older records remain readable with `ingestion_receipt=None`. The complete workflow in the function reference includes this argument.
3. **Define the baseline experiment next, when ready.** Set the target, issuance/cutoff, minimum history, historical forecast origins, point statistic, scores, required checks, and expected artifacts. Its future runner must pin the dataset plus resolved snapshot ID and record the training selection. Neither this design nor runner integration is implemented yet. Further versioning enhancements are optional later slices.

Do not rebuild the completed snapshot/tag functions or introduce a larger storage framework as the next step. Keep work in small pieces the user can read and understand. git-annex remains optional; DVC or another dependency/stage/execution tool is deferred.

## Verification at this stopping point

- The receipt enhancement passed all 57 tests: 17 ingestion, 27 snapshot, and 13 tag tests. Receipt checks include retained source/time context, repeated identical bytes, old records, mismatched destinations and sizes, timezone handling, and store relocation. Snapshot and tag fixtures use temporary stores, including simulated copy/publication failures.
- The complete example in the source-directory README ran with 200 freshly generated observations in a temporary store. Its relative file links were also checked.
- The earlier snapshot/tag publication also passed dependency compatibility and lockfile checks. This enhancement changes no dependencies. Rerun the suite after code changes:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The supported local scope is one file per snapshot, one writer at a time, local writable storage, and integrity verification on retrieval. Multi-file identity, processing lineage, quality eligibility, concurrent writers, remote storage, and crash recovery are not implemented. Local data and records are ignored by Git and are not uploaded with code.

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
- [ ] Agree on the first synthetic experiment and its observable acceptance criteria.

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

## Further versioning enhancements — not selected for implementation

Local ingestion receipts are now carried into registration. Remote source-specific context, processing lineage, multi-file collections, quality-check evidence, additional source functions, and storage/recovery improvements remain possible later slices. Implement them only when a concrete need is selected; the list is not a requirement to finish every feature before learning from a baseline.

See [the current snapshot/tag behavior](snapshots.md), [the decision log](decisions.md), and [the broader versioning requirements and historical plan](data-versioning.md). The roadmap's resume section is the current continuation order.

Turn each deliverable into a small GitHub issue using the repository's task template. This roadmap is not a commitment to implement all possible extensions.
