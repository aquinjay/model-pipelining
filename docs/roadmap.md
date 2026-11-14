# Roadmap

Updated: 2026-11-14. This page records implemented scope and review priorities.
For usage, start with [the walkthrough](forecasting.md). For past choices, use
[the decision history](decisions.md); historical proposals are not build checklists.

## Resume here

Step 1, the computational review of the current modeling workflow, is complete.
The [review record](workflow-review.md) explains one concrete forecast, saved
checks, exact replay, and the remaining assumption/evidence gaps. All 118 tests
passed, and both models completed all 170 cases on both synthetic series.
Human model approval remains pending.

Step 2 now has an explicit [local-level diagnostic workflow](diagnostics.md):
caller-chosen prior/noise influence paths, conditional initial-mean weights,
sequential innovation/PIT criticism, saved failures, and pending human review.
The user clarified that sensitivity should reveal assumption influence, not
minimize it. Strong influence is not a computational failure; no magnitude or
configuration-agreement threshold was added. The workflow is separate from the
backtest and introduces no new dependency.

The user clarified that infrastructure, rather than application selection, is
the current goal. Step 3 now supplies [reusable experiment specs](experiments.md):
target/source descriptions and columns/units, issuance and as-of release/revision
selection, explicit daily calendar/gap validation, required benchmarks on matching
cases, and separate development/final target periods. The example uses known
synthetic fixtures; it does not choose a real application or certify untouched data.

The next modeling detour is [parameter uncertainty and MCMC](parameter-uncertainty.md):
start with an independently verifiable unknown-parameter model, integrate its
posterior into prediction, and add fit/chain evidence and relevant computational
diagnostics when introducing a sampler. The concrete model/backend remain open.
Prior influence remains descriptive, not a quantity to minimize. Application
selection, richer calendars, and a broader release/check framework can wait.

To understand or repeat the reviewed workflow:

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
| Execution | Daily model-independent experiments, release-aware expanding histories, one fixed positive-day horizon per run; defaults preserve the original next-day task | [Experiments](experiments.md) |
| Models | Training mean and last-value point models; fixed-variance Gaussian local level with exact filtering | [Model guide](local-level.md) |
| Uncertainty | Observation quantiles and highest-density regions; disconnected regions supported by the shared format | [Uncertainty](uncertainty.md) |
| Evaluation | Point errors, quantile/equal-tail scores, separate HPD coverage/total width, explicit case completion | [Uncertainty evaluation](uncertainty.md#validation-and-evaluation) |
| Reporting | Saved-result charts/comparisons; HPD preferred; version-1 point artifacts still readable | [Run files](../runs/README.md) |
| Model evidence | Explicit full-path prior/noise influence, conditional initial-mean weights, sequential criticism, saved-only diagnostic charts | [Diagnostics](diagnostics.md) |
| Task/evaluation contracts | Version-1 specs; manifest version 3, target roles, saved case exclusions, selected-history hashes, explicit actual vintage, required benchmarks | [Experiments](experiments.md) |

## Verification at this checkpoint

The suite currently has 146 tests. [The test guide](../tests/README.md) maps them
to behavior: data preservation, independent posterior/predictive calculations,
requested uncertainty validation, disjoint region membership/display, future
isolation, frozen-file replay, legacy compatibility, and failure handling.
The 12 diagnostic tests include independent batch-conditioning influence
calculations, strong persistent prior influence completing successfully,
prior/noise distinctions, saved failures, future isolation, replay, and CLI
scenario inheritance. Both synthetic series were exercised with the reference
and four illustrative alternatives over all 200 observations.

The 16 experiment tests cover delayed releases and revisions, distinct scoring
vintages, available-period counts, target-based development/final separation,
matching benchmark cases, all-forecast freezing, visible benchmark failures,
arbitrary positive-day horizons and adapter rejection, experiment replay,
history-identity comparison, version-2 reporting, and both CLI entry points.
The 200-day fixture also exercised both models with 130 development and 40
final-role cases, each with both point benchmarks; the Bayesian development
run's numerical artifacts replayed exactly. Final-role results are infrastructure
evidence on a known fixture, not an untouched scientific evaluation.

Both models were exercised on both existing synthetic series, with 170 common
forecast dates per series. The uncertainty run was replayed exactly, including
quantiles, HPD segments, evaluations, and scores. The documentation walkthrough
is also checked in temporary storage. These observations establish the stated
local behavior, not calibration or approval of a real-data model.

The local-level/shared-uncertainty work adds no dependency beyond the existing
pandas/openpyxl/matplotlib environment. The filter uses the standard library;
independent batch-conditioning tests use NumPy already installed through pandas.

## Next decisions, not automatic implementation

- **Further model evidence:** decide whether the application needs full-data
  posterior predictive replications or other criticism beyond the implemented
  influence/innovation evidence. A configurable shared check plan, experimental
  overrides, and application-specific human/release criteria remain open.
- **Parameter uncertainty and MCMC (next detour):** select the first verifiable
  unknown-parameter model, inference method, posterior/predictive evidence, and
  computational diagnostic requirements. Current local-level uncertainty remains
  conditional on fixed variances. [The plan](parameter-uncertainty.md) separates
  parameter uncertainty from Monte Carlo error and describes the proposed sequence.
- **Real-data application (deferred):** fill the implemented task/evaluation
  contract with an actual source and verified historical release claims when
  application choice becomes useful; it is not the immediate project objective.
- **Another adapter:** choose a concrete need before adding a contrasting model
  such as boosting; preserve point-only capability handling when appropriate.

The Bayesian adapter was selected before the real-data application. This is an
order of learning tasks, not a requirement to finish every possible storage or
modeling feature first.

## Later, when a need is demonstrated

Multi-file snapshots, processing lineage, richer acquisition provenance, remote
sources/storage, concurrent writers, crash recovery, backup/retention automation,
multiple simultaneous horizons, richer calendars, tuning, a run index, dashboards, scheduling,
and monitoring remain possible extensions. git-annex is optional; DVC or another
execution tool is unselected. Neither is a prerequisite to reviewing the current
working path. See [the historical storage proposal](data-versioning.md) for the
broader requirements and alternatives that have been retained for reference.
