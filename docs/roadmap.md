# Roadmap

Updated: 2026-11-13. This page records implemented scope and review priorities.
For usage, start with [the walkthrough](forecasting.md). For past choices, use
[the decision history](decisions.md); historical proposals are not build checklists.

## Resume here

The current priority is to understand and review the modeling infrastructure
before selecting more implementation work:

1. Follow [the workflow](forecasting.md) to see how a preserved input becomes
   two repeatable runs and saved reports.
2. Review [the local-level model](local-level.md): its fixed settings, initial-prior
   timing, forward calculation, and independently checked results.
3. Review [shared uncertainty](uncertainty.md): next-observation semantics, HPD
   priority, capabilities, frozen outputs, compatible evaluation, and failures.
4. Use [the API reference](../src/model_pipelining/README.md) and
   [test map](../tests/README.md) to inspect exact interfaces and supporting checks.

Implementation acceptance means correct calculations, no future data in model
requests, explicit failures, and reproducible saved results. Better scores,
faster reactions, or a particular empirical interval coverage are not acceptance
thresholds. Human interpretation of assumptions and evidence remains open.

The baseline workflow was introduced in `eabd66e`. The current checkout extends
it with the local-level adapter and shared uncertainty. Use Git status/history
to distinguish local edits from committed/published code.

## Implemented scope

| Area | Implemented | Detailed reference |
|---|---|---|
| Environment | CPython 3.14, development pin 3.14.10, uv lock and pip export, unittest | [Setup](../README.md#development-environment) |
| Synthetic sources | Two seeded 200-day series and generation manifest | [Data](../data/README.md) |
| Ingestion | Local copy receipts and CSV/XLSX table reading | [Ingestion](ingestion.md) |
| Preservation | One-file SHA-256 snapshots, verified retrieval, repeat registration events, optional receipt context | [Snapshots](snapshots.md) |
| References | Dataset-scoped explicit tags, saved assignment history, permanent run input IDs | [Tags](snapshots.md#assign-resolve-and-inspect-tags) |
| Execution | One daily numerical series, expanding histories, next-day forecasts, fixed required checks, visible failures | [Walkthrough](forecasting.md) |
| Models | Training mean; fixed-variance Gaussian local level with exact filtering | [Model guide](local-level.md) |
| Uncertainty | Observation quantiles and highest-density regions; disconnected regions supported by the shared format | [Uncertainty](uncertainty.md) |
| Evaluation | Point errors, quantile/equal-tail scores, separate HPD coverage/total width, explicit case completion | [Uncertainty evaluation](uncertainty.md#validation-and-evaluation) |
| Reporting | Saved-result charts/comparisons; HPD preferred; version-1 point artifacts still readable | [Run files](../runs/README.md) |

## Verification at this checkpoint

The suite currently has 118 tests. [The test guide](../tests/README.md) maps them
to behavior: data preservation, independent posterior/predictive calculations,
requested uncertainty validation, disjoint region membership/display, future
isolation, frozen-file replay, legacy compatibility, and failure handling.

Both models were exercised on both existing synthetic series, with 170 common
forecast dates per series. The uncertainty run was replayed exactly, including
quantiles, HPD segments, evaluations, and scores. The documentation walkthrough
is also checked in temporary storage. These observations establish the stated
local behavior, not calibration or approval of a real-data model.

The local-level/shared-uncertainty work adds no dependency beyond the existing
pandas/openpyxl/matplotlib environment. The filter uses the standard library;
independent batch-conditioning tests use NumPy already installed through pandas.

## Next decisions, not automatic implementation

- **Model evidence:** decide the required prior-sensitivity and fitted-model
  criticism checks, how they are saved, and which outcomes should block execution
  or release. The runner's current check list is fixed; a configurable check plan
  and explicit experimental overrides are not implemented.
- **Real-data application:** choose a target/source and define its availability,
  calendar, revisions, benchmark, and final evaluation period. The synthetic
  immediate-availability assumption is not enough for this application.
- **Richer model uncertainty:** decide whether to infer variance parameters or
  introduce predictive samples/full densities, with corresponding verification
  and evaluation. Current uncertainty is conditional on fixed settings.
- **Another adapter:** choose a concrete need before adding a contrasting model
  such as boosting; preserve point-only capability handling when appropriate.

The Bayesian adapter was selected before the real-data application. This is an
order of learning tasks, not a requirement to finish every possible storage or
modeling feature first.

## Later, when a need is demonstrated

Multi-file snapshots, processing lineage, richer acquisition provenance, remote
sources/storage, concurrent writers, crash recovery, backup/retention automation,
multiple horizons, richer calendars, tuning, a run index, dashboards, scheduling,
and monitoring remain possible extensions. git-annex is optional; DVC or another
execution tool is unselected. Neither is a prerequisite to reviewing the current
working path. See [the historical storage proposal](data-versioning.md) for the
broader requirements and alternatives that have been retained for reference.
