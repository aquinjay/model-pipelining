# Roadmap

Updated: 2026-11-17. This page records implemented scope and review priorities.
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

The [parameter-uncertainty detour](parameter-uncertainty.md) is implemented:
`NormalMean` learns a constant mean, computes exact marginalized observation
forecasts, and optionally audits each fit with independent emcee Metropolis
chains and ArviZ diagnostics. Optional fit-evidence dispatch saves posterior,
chain/draw, warmup, seed, transform, precision/reference checks, and partial
failures before scoring. Required audit failures fail cases/runs. Both modes
use exact forecast integration; MCMC-only forecast/HPD estimation remains open.
Prior influence remains descriptive, not a quantity to minimize. Parameter
learning now also includes an unknown observation variance:
[NormalMeanVariance](normal-mean-variance.md) uses normal/inverse-gamma updates,
exact Student-t forecasts and an optional joint sampler audit. Heavy-tail moment
existence and the transform Jacobian are explicit. Local-level observation and
evolution variances remain fixed. Application selection and a broader
release/check framework can wait.

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

The baseline workflow was introduced in `eabd66e`. PR #1 merged unknown-mean
parameter learning and its sampler audit; the current checkout extends that
with joint mean/variance learning. Use Git status/history
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
| Parameter learning | Unknown constant Gaussian mean with known or learned observation variance; exact normal/Student-t predictions, optional independent-chain audits and saved fit evidence | [Parameter uncertainty](parameter-uncertainty.md), [joint model](normal-mean-variance.md) |
| Uncertainty | Observation quantiles and highest-density regions; disconnected regions supported by the shared format | [Uncertainty](uncertainty.md) |
| Evaluation | Point errors, quantile/equal-tail scores, separate HPD coverage/total width, explicit case completion | [Uncertainty evaluation](uncertainty.md#validation-and-evaluation) |
| Reporting | Saved-result charts/comparisons; HPD preferred; version-1 point artifacts still readable | [Run files](../runs/README.md) |
| Model evidence | Explicit full-path prior/noise influence, conditional initial-mean weights, sequential criticism, saved-only diagnostic charts | [Diagnostics](diagnostics.md) |
| Task/evaluation contracts | Version-1 specs; manifest version 3, target roles, saved case exclusions, selected-history hashes, explicit actual vintage, required benchmarks | [Experiments](experiments.md) |

## Verification at this checkpoint

The suite currently has 194 tests. [The test guide](../tests/README.md) maps them
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

The local-level filter still uses the standard library. The parameter-learning
slice declares NumPy, SciPy, emcee and pre-1.0 ArviZ explicitly, with regenerated
lockfile/pip export. CPython 3.14.10 imports, installed compatibility, and the full
suite passed. The 28 new tests cover independent quadrature, exact HPD semantics,
prior influence, sampler target ratios and posterior/predictive agreement,
biased/stuck/missing diagnostics, partial backend failures, global-RNG isolation,
fit persistence/freezing/replay, delayed releases/final roles, both inference CLI
modes, and saved-only traces. Human approval remains pending.

Saved local exercises under `runs/parameter-uncertainty-final-20261114/` completed
170 exact cases on each synthetic series. Each series also completed five
explicit final-role audit cases at target indices 100–104, with both point
benchmarks, and an exact matched-cohort run. Every fit/draw and numerical forecast
file replayed exactly; exact/audit forecasts and scores matched. The largest
R-hat was 1.003979 and smallest bulk/tail ESS were 1550.38/1329.21 across the
exercised diagnostics. Saved-only charts and comparisons succeeded. These known
fixtures demonstrate mechanics, not untouched final evaluation or scientific
model approval. Generated artifacts are local and ignored by Git.

The joint model adds 20 tests; all 194 tests passed. Both seeded synthetic
series completed 170 exact forecast cases, plus three explicitly selected
final-role audit cases at target indices 100–102 with both point benchmarks.
Exact/audit predictions and scores matched, and all numerical fit/draw and run
files replayed byte for byte (excluding manifest timestamps). Saved-only charts
and comparisons succeeded. Evidence is local under
`runs/variance-learning-20261117-t2f9oedw/`. Across those joint audits, maximum
R-hat was 1.005306 and minimum bulk/tail ESS were 2638.73/3362.73. These known
fixtures demonstrate mechanics, not untouched final evaluation or approval of
the constant-mean assumption. No new dependency was added.

## Next decisions, not automatic implementation

- **Further model evidence:** decide whether the application needs full-data
  posterior predictive replications or other criticism beyond the implemented
  influence/innovation evidence. A configurable shared check plan, experimental
  overrides, and application-specific human/release criteria remain open.
- **Further parameter learning:** unknown observation variance is now learned
  in the exact-reference constant-mean model. Next choose whether a richer
  unknown-variance local-level model is needed, with an independently validated reference
  and explicit inference evidence. Intractable sampled prediction needs its own
  quantile/HPD approximation and precision design; current audit forecasts remain
  exact. Local-level variances are still fixed. See [current scope](parameter-uncertainty.md).
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
