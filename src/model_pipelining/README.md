# Python API reference

Use this page to look up arguments, return types, and records. For a complete
example with explanations, start with [the forecasting walkthrough](../../docs/forecasting.md).
All relative paths below are relative to the repository root. Launch Python with:

```bash
PYTHONPATH=src .venv/bin/python
```

A `*` in a signature means the following arguments must be named. Public
functions are listed below; names starting with `_` are implementation helpers.

| Module | Responsibility |
|---|---|
| [ingest.py](ingest.py) | Copy a completed file and read a CSV/XLSX table |
| [snapshots.py](snapshots.py) | Preserve exact file bytes, verify retrieval, inspect registrations |
| [tags.py](tags.py) | Assign readable dataset references and retain assignment history |
| [models.py](models.py) | Shared requests/results, capabilities, training mean, and last-value baseline |
| [experiments.py](experiments.py) | Immutable versioned task/evaluation specs, release parsing, as-of case planning, and scoring vintage selection |
| [local_level.py](local_level.py) | Gaussian local-level adapter, filtering, prior simulation |
| [normal_mean.py](normal_mean.py) | Unknown constant mean, exact posterior/predictive integration, optional sampler audit |
| [normal_mean_variance.py](normal_mean_variance.py) | Joint unknown mean/variance, normal/inverse-gamma posterior, exact Student-t forecasts |
| [inference.py](inference.py) | Optional fitted-forecast/evidence records and runner-owned persistence |
| [sampling.py](sampling.py) | Independent emcee Metropolis chains, ArviZ diagnostics, exact-reference audits |
| [sampling_variance.py](sampling_variance.py) | Joint two-coordinate audit with Jacobian and moment-aware diagnostics |
| [diagnostics.py](diagnostics.py) | Explicit assumption influence and sequential criticism; plots from saved diagnostic paths |
| [uncertainty.py](uncertainty.py) | Shared output validation and compatible uncertainty evaluation |
| [backtest.py](backtest.py) | Daily history selection, model execution, frozen forecasts, scoring, checks |
| [reporting.py](reporting.py) | Read saved artifacts and draw/compare results without fitting |

## Terms and record boundaries

| Term | Meaning |
|---|---|
| Dataset | A named history of input versions, such as `synthetic-mean-shift` |
| Snapshot | One preserved file version; referenced by dataset plus snapshot ID |
| Snapshot ID | Lowercase 64-character SHA-256 of exact file bytes; identifies but cannot recreate missing bytes |
| Tag | A movable readable reference within one dataset |
| Store | Preserved files and their metadata; defaults to `data/versioned` |
| Registration/acquisition record | One successful `register_snapshot` event; may carry an earlier ingestion receipt |
| Snapshot manifest | File identity/layout record, format version 1 |
| Run manifest | Input/experiment/task/model/evaluation/environment/status record, format version 3 for new runs |
| Experiment specification | Version-1 model-independent target/timing, benchmark, and evaluation contract |
| Run | One model/configuration executed over selected historical forecast cases |

The different manifest versions apply to different record types. Adding run
format version 3 did not change snapshot, acquisition, or tag format version 1.
Acquisition time, registration time, and historical observation availability are
separate concepts. The first two do not establish the third.

## Copying and table reading

```text
ingest_local_file(source, destination) -> IngestResult
read_local_table(path, *, sheet=None, header=0) -> pandas.DataFrame
```

Source/destination/path arguments accept strings or `Path` objects.
`ingest_local_file` copies one regular file, creates missing parent directories,
and refuses existing destinations (including symlinks). A handled copy failure
removes only its partial output. The source must remain unchanged during copying.

`IngestResult` is a frozen dataclass:

| Field | Value |
|---|---|
| `source` | Resolved original local `Path` |
| `destination` | Resolved completed-copy `Path` |
| `acquired_at` | Timezone-aware UTC copy-completion time |
| `size_bytes` | Copied byte count |

The reader infers CSV/XLSX from the case-insensitive extension and returns one
DataFrame. For XLSX, `sheet` is a worksheet name or nonnegative zero-based index;
omission selects the first sheet. CSV rejects `sheet`. `header` is a nonnegative
zero-based row index or `None` for no header. pandas infers types/missing values;
the function does not clean data, convert application timestamps, or calculate
Excel formulas. Reading XLSX is supported here; the forecasting runner accepts
CSV only. [The ingestion guide](../../docs/ingestion.md) covers detailed behavior.

## One complete workflow

**This example covers data preparation only:** copy → preserve → tag → select →
read. Continue with [the forecasting walkthrough](../../docs/forecasting.md) to
run models. Generate the source first if absent with
`.venv/bin/python scripts/generate_synthetic.py`.

```python
from model_pipelining.ingest import ingest_local_file, read_local_table
from model_pipelining.snapshots import register_snapshot, retrieve_snapshot
from model_pipelining.tags import assign_tag, resolve_tag

receipt = ingest_local_file(
    "data/synthetic/mean_shift.csv",
    "data/ingested/api-reference-001/mean_shift.csv",
)
snapshot = register_snapshot(
    receipt.destination,
    dataset="synthetic-mean-shift",
    ingestion_receipt=receipt,
    note="Input prepared using the API reference",
)
assign_tag(
    dataset=snapshot.dataset, tag="baseline-input",
    snapshot_id=snapshot.snapshot_id, changed_by="api-example",
)
selected_snapshot_id = resolve_tag(dataset=snapshot.dataset, tag="baseline-input")
preserved_file = retrieve_snapshot(selected_snapshot_id, dataset=snapshot.dataset)
table = read_local_table(preserved_file)
```

Choose an unused ingestion destination if repeating this example. Copying does
not register/tag a snapshot. Passing `ingestion_receipt=receipt` makes registration
save that context automatically; no separate receipt-saving call is needed.
You can also register an existing completed file directly without a copy/receipt.

## Preserving and inspecting versions

All functions below accept optional keyword `store="data/versioned"`:

```text
register_snapshot(path, *, dataset, note=None, ingestion_receipt=None, store=...) -> Snapshot
retrieve_snapshot(snapshot_id, *, dataset, store=...) -> Path
list_snapshots(*, dataset, store=...) -> list[dict]
list_acquisitions(*, dataset, store=...) -> list[dict]
```

`Snapshot` is a frozen dataclass with `dataset`, `snapshot_id`, and `path`.
Keep the dataset/ID as the permanent reference. The path is the current local
location; retrieve again after moving the store. Retrieval checks the manifest,
size, and hash and raises on missing/tampered contents.

Identical bytes in the same dataset reuse a snapshot, even if renamed; the first
preserved filename stays. Every successful registration adds a separate event.
Changed bytes create a new snapshot. Different datasets have separate copies and
histories even when the content hash matches.

`note` accepts a string or `None` and belongs to the event, not snapshot identity.
An optional ingestion receipt must describe the registered destination/byte count
and have an aware acquisition time. Its original source need not still exist.
The receipt has no hash; the caller must keep the acquired copy unchanged until
registration, including against same-size edits.

| Listing | Contents and order |
|---|---|
| `list_snapshots` | `dataset`, `snapshot_id`, `filename`, `size_bytes`; sorted by ID |
| `list_acquisitions` | `format_version`, `acquisition_id`, `dataset`, `snapshot_id`, `source_path`, `registered_at`, `note`, `ingestion_receipt`; oldest UTC registration first, then ID for ties |

Listings return `[]` for absent data without creating directories. They inspect
metadata, not preserved bytes. Older optional note/receipt fields return `None`
without rewriting files. Historical receipt paths remain context after moving
the store. [Snapshot files and record fields](../../docs/snapshots.md#files-on-disk)
are described in the storage reference.

## Understanding tags

All tag functions accept optional keyword `store="data/versioned"`:

```text
assign_tag(*, dataset, tag, snapshot_id, changed_by, note=None, store=...) -> dict
resolve_tag(*, dataset, tag, store=...) -> str
list_tags(*, dataset, store=...) -> list[dict]
list_tag_history(*, dataset, tag, store=...) -> list[dict]
```

Assignment verifies the target in the same dataset, then records every explicit
assignment, including repeats. Registration never creates/moves tags. Multiple
tags may select one snapshot; different datasets may use the same tag name.
Moving a tag does not alter a previously resolved permanent ID or a saved run.

`changed_by` is required, nonblank, caller-supplied context rather than
authenticated identity. Optional `note` accepts a string or `None`. The returned
event contains `change_id`, `previous_snapshot_id`, `snapshot_id`, UTC
`changed_at`, `changed_by`, and `note`.

Resolution reads the current ID and raises `FileNotFoundError` for an unassigned
tag. Retrieval separately verifies its contents. `list_tags` returns current
target/last-assignment metadata sorted by tag name. `list_tag_history` returns
all events in saved sequence, including when the clock moves backwards. Empty
listings return `[]`; history for an unassigned tag is also `[]`. See
[the tag reference](../../docs/snapshots.md#assign-resolve-and-inspect-tags).

## Backtesting and reporting

```text
run_backtest(
    *, dataset, output_directory, snapshot_id=None, tag=None,
    minimum_history=None, experiment=None, model=None, interval_levels=(0.8, 0.95),
    store="data/versioned",
) -> Path

plot_run(run_directory, *, interval_level=None, interval_method="auto") -> Path
compare_runs(first_run, second_run, *, output_directory) -> Path
```

Import the runner from `model_pipelining.backtest` and reports from
`model_pipelining.reporting`. Supply exactly one of `snapshot_id` and `tag`; the
runner resolves a tag once and saves its permanent ID. `model=None` selects
`TrainingMean`. For the Bayesian adapter, pass `LocalLevel(...)` from
`model_pipelining.local_level`. Every output directory must be unused. Without
`experiment`, the effective minimum history is 30 unless `minimum_history` is
supplied. With an `ExperimentSpec`, omit that legacy argument and choose the
count in `experiment.task`.

`run_backtest` automatically verifies/parses the CSV, validates its daily calendar
and releases, saves the case schedule, selects as-of training versions, executes
the model and declared benchmarks, freezes outputs, scores against the declared
actual vintage, and saves checks/review. It returns an absolute run-directory `Path`, **including
when required cases failed**. Input/configuration errors raise before output
creation. Per-case forecast failures are retained and later origins continue.
Filesystem errors propagate and can leave a `running` directory.

Read `manifest.json.status` and `checks.json` before interpreting results. Point
`scores.complete` is not the overall result: required uncertainty scoring can
fail independently; a declared benchmark failure also fails the run. The [walkthrough](../../docs/forecasting.md#read-the-results)
explains status and [run-file reference](../../runs/README.md) lists artifacts.
The library does not acquire/register/tag data, simulate priors, or draw charts
inside `run_backtest`. The three run scripts call reporting automatically; the
local-level script also handles prior simulation/files.

`plot_run` reads saved evidence and returns the chart path. `"auto"` prefers the
widest saved highest-density region, then equal-tail intervals. Explicit
`interval_method` is `"highest_density"` or `"equal_tail"`; explicit levels must
have been saved. A point-only run has no band. Existing charts are rejected.

`compare_runs` reads completed runs with matching permanent inputs, experiment,
task, training coordinates/history hashes, target dates, and actual vintages. It saves JSON/PNG comparison
files and returns the absolute comparison-directory path. It does not fit,
tune, or select a model. Uncertainty comparisons use common supplied kinds and
levels only; a point-only baseline has no uncertainty comparison.

## Experiment specifications and planning

Import `ForecastTask`, `EvaluationPlan`, and `ExperimentSpec` from
`model_pipelining.experiments`. All are frozen keyword-only dataclasses.
`ForecastTask()` supplies the original daily immediate/unrevised task with
30 minimum observations. `EvaluationPlan()` supplies development-only evaluation
with no extra benchmarks. `ExperimentSpec(task=..., evaluation=...)` combines
them. See [fields, defaults, semantics, and examples](../../docs/experiments.md).

```text
ExperimentSpec.to_dict() -> dict
ExperimentSpec.from_dict(value) -> ExperimentSpec
ExperimentSpec.load(path) -> ExperimentSpec
read_task_data(path: Path, task: ForecastTask) -> tuple[ObservationVersion, ...]
plan_cases(versions, experiment) -> (tuple[ForecastCase, ...], list[dict])
select_actuals(versions, task) -> dict[str, ObservationVersion]
```

`load` reads version-1 JSON explicitly; unknown constructor fields/versions or
invalid settings raise. `to_dict` expands defaults, serializes UTC boundary
timestamps, and emits benchmark names as a list. `from_dict` restores them.
No function fetches data, executes a model, or automatically starts final evaluation.

The parsing/planning functions support inspection without executing a run.
`read_task_data` validates the ordered daily CSV and release table;
`ObservationVersion` holds `timestamp`, `value`, and `available_at`. Pass its
validated records to `plan_cases`. A `ForecastCase` holds `origin_index`,
`issued_at`, `target_time`, and selected `versions`; `.history` returns immutable
model observations and `.history_sha256` hashes timestamp/value/release triples.
The schedule records selected/excluded candidate targets. `select_actuals`
chooses the declared first/latest scoring vintage, keyed by UTC ISO target time;
the runner invokes it only after freezing forecasts. Record construction itself
does not validate raw data.

Only daily UTC with gap rejection is implemented. One fixed positive-day horizon
is allowed per run. Delayed releases, revisions, and issuance offsets can produce
requests unsupported by an adapter: the current `LocalLevel` still requires
contiguous daily history ending exactly at issuance and a one-day target.
Unsupported requests fail visibly rather than changing the declared experiment.

## Model requests and results

`Forecaster` is a Python protocol: adapters supply a nonempty `name`, JSON
`configuration` dictionary, and `forecast(request)`. Boolean `supports_quantiles`
and `supports_highest_density` flags declare available uncertainty. Absent flags
are treated as false for legacy adapters. `PointForecaster` remains an alias.
Models refit from each request, retaining no prior request's training data.

| Immutable record | Fields |
|---|---|
| `Observation` | `timestamp`, `value` |
| `ForecastRequest` | `history` (tuple of observations), `issued_at`, `target_time`, `quantile_levels=()`, `highest_density_levels=()` |
| `Forecast` | `mean`, `quantiles=()`, `highest_density_regions=()` |
| `Quantile` | `probability`, `value` |
| `HighestDensityRegion` | `level`, `segments` (tuple of `Interval` records) |
| `Interval` | `lower`, `upper` |

The runner requests all declared uncertainty kinds at `interval_levels`.
`LocalLevel.forecast` returns `Forecast`; direct callers use `.mean` for the point
value. `TrainingMean` and `LastValue` return numerical point forecasts and declare
no uncertainty. `LastValue` persists the latest eligible observation; both are
also selectable by name as experiment benchmarks.
Numeric legacy outputs are accepted for point-only adapters; declaring uncertainty
while returning only a number fails the required-output check. Direct callers constructing
a request must specify desired levels; the empty request defaults request none.

Record construction itself does not validate numerical contents; the runner
performs output validation before accepting/saving a forecast.

All shared uncertainty targets the future **observation**, not a hidden state
or parameter. [The uncertainty contract](../../docs/uncertainty.md) defines exact
validation/scoring behavior, including HPD segments and failure handling.

`LocalLevel` accepts fixed `initial_mean=0`, `initial_variance=100`,
`observation_variance=1`, and `evolution_variance=0.1` settings. Its methods also
include `predictive_distribution(request) -> NormalDist`,
`filter_history(history) -> tuple[FilterStep, ...]`, and
`simulate_prior(*, count, seed=0) -> tuple[PriorDraw, ...]`.
[The model guide](../../docs/local-level.md#library-interface-and-diagnostics)
defines the diagnostic fields, timing, and separate simulation/reporting calls.

## Unknown-mean model and optional fit evidence

Import `NormalMean` and `MeanPosterior` from `model_pipelining.normal_mean`,
`MCMCSettings` from `model_pipelining.sampling`, and `FitEvidence`,
`FittedForecast` from `model_pipelining.inference`.

```text
NormalMean(*, prior_mean=0, prior_variance=100, observation_variance=1, mcmc=None)
NormalMean.posterior(request) -> MeanPosterior
NormalMean.predictive_distribution(request) -> NormalDist
NormalMean.fit(request) -> FittedForecast
NormalMean.forecast(request) -> Forecast
plot_fit(run_directory, *, origin_index) -> Path
```

`MeanPosterior` fields are `mean`, `variance`, `prior_mean_weight`, and
`observation_variance`; its `predictive_variance` property adds observation and
parameter variances. Both configured variances must be positive. Unlike the
dynamic local-level model, this exchangeable constant-mean model accepts gapped
eligible histories, delayed issuance, and any future target. The runner still
enforces experiment coordinates. Forecasts always use exact integration.

`mcmc=MCMCSettings(...)` explicitly audits each fit; omitting it is exact-only.
Fields/defaults and all computation criteria are in [the model and sampler
guide](../../docs/parameter-uncertainty.md#computation-criteria-and-failures).
`configuration` saves the nested settings as JSON; reconstruct that nested
dictionary with `MCMCSettings(**settings)` before constructing the model.

`FittedForecast` contains `forecast` and `evidence`. `FitEvidence` contains
`metadata: dict`, `checks: tuple[dict, ...]`, `diagnostics: tuple[dict, ...]=()`,
and `draws: tuple[dict, ...]=()`. These frozen records contain trusted adapter
data; their nested dictionaries are not deeply immutable. Checks have unique
nonblank `name`, boolean `required`, and `status` in `passed/failed/not_applicable`.
At least one required check is needed; a required check cannot be inapplicable.
All evidence must serialize as finite JSON; missing diagnostics use null plus
a failed check. Draw rows must share one CSV schema.

The optional boolean `supports_fit_evidence=True` makes the runner call
`fit(request)` instead of `forecast(request)`. It saves runner-owned permanent
input/history/issuance/target identity with evidence before accepting outputs.
Required failed checks reject the candidate and fail that case while later cases
continue. A malformed/missing result fails visibly. Per-case evidence/index files
are frozen before actual-vintage selection. `implementation_files` can list
helper `Path`s for source fingerprints; `NormalMean` includes its sampler helper
when auditing. Existing adapters need neither optional capability.

For direct callers, `fit` retains failed audit evidence without persisting it;
`forecast` raises on failed required audits. `save_fit_evidence(result, directory,
*, identity) -> bool` validates/persists records and returns required-check
success; the runner invokes it automatically. Direct library callers choose
whether to persist. Persistence errors propagate, leaving possible partial
running output. This is one optional evidence boundary, not a configurable
shared check-plan engine.

`plot_fit` is imported from reporting and reads saved finished-run/fit files only.
It creates `fits/NNNNNN/trace.png` with excluded warmup and exact parameter/
predictive references. Exact-only fits have no trace. Existing charts are
rejected; partial failed audit traces can be drawn when some draws were saved.
The CLI requests this explicitly with `--plot-fit-origin`; it does not draw all
fit traces automatically. See [run files](../../runs/README.md) for retention.

## Joint mean and observation variance

Import `NormalMeanVariance` and `NormalInverseGammaPosterior` from
`model_pipelining.normal_mean_variance`.

```text
NormalMeanVariance(*, prior_mean=0, prior_mean_precision=0.01,
                   prior_shape=3, prior_scale=2, mcmc=None)
NormalMeanVariance.posterior(request) -> NormalInverseGammaPosterior
NormalMeanVariance.predictive_distribution(request) -> frozen scipy.stats.t
NormalMeanVariance.fit(request) -> FittedForecast
NormalMeanVariance.forecast(request) -> Forecast
```

The posterior fields are `mean`, `mean_precision`, `shape`, `scale`, and
`prior_mean_weight`. Properties are `degrees_of_freedom`, `mean_scale_squared`,
`predictive_scale_squared`, `observation_variance_mean`, `mean_variance`, and
`predictive_variance`. The last three return `None` when posterior shape is at
most 1. Scale squared is not variance. Positive finite prior precision, shape,
and inverse-gamma scale are required. The prior mean is finite. History/timing
requirements match `NormalMean`.

`predictive_distribution` supports SciPy `.pdf`, `.cdf`, `.ppf`, `.mean()`, and
`.var()`; unlike the nullable posterior properties, SciPy can return infinity
for an undefined variance. Both inference modes return exact Student-t
forecasts; shared records, dispatch, persistence and replay are unchanged.
Pass `mcmc=MCMCSettings(...)` explicitly to audit both parameters. The joint
helper and shared sampler source are fingerprinted. Undefined-moment audit
omissions have saved reasons; applicable missing/failed diagnostics still fail.

The existing CLI selects this model with `--learn-variance` and uses
`--prior-mean-precision`, `--prior-shape`, and `--prior-scale` instead of the
known-variance flags. `plot_fit` reads saved Student-t/inverse-gamma references
and draws three trace/density rows without refitting. See [equations, units,
commands, diagnostics and replay](../../docs/normal-mean-variance.md).

## Local-level diagnostics

```text
run_local_level_diagnostics(
    *, dataset, reference, scenarios, output_directory,
    snapshot_id=None, tag=None, store="data/versioned",
) -> Path

plot_local_level_diagnostics(diagnostics_directory) -> tuple[Path, Path]
```

Import both functions from `model_pipelining.diagnostics`. Supply a concrete
`LocalLevel` reference and a nonempty `dict[str, LocalLevel]` of alternatives;
subclasses are not supported. Scenario names must be nonblank strings other
than the reserved `reference`. Use exactly one input selector and an unused
output directory. The tag resolves once and the snapshot is verified/parsed
using the daily CSV rules. At least one observation is required.

The calculation saves every full forward path from the first observation,
pre-observation standardized innovations/PIT, update gain, updated level mean/SD,
next-observation SD, conditional initial-mean derivative, and reference
differences. Summaries preserve full settings, changed prior/noise groups,
criticism, influence, and per-configuration errors. No settings are selected
automatically; no influence magnitude or agreement threshold determines success.
Human review remains pending. See [diagnostic semantics and examples](../../docs/diagnostics.md).

Invalid options/input raises before output creation. Numerical configuration
failures are saved and later configurations continue. The returned path can
contain failed computation: inspect `diagnostics.json.status` and
`scenario_summaries.json`. If the reference fails, valid alternative paths are
retained with unavailable comparison differences. I/O errors propagate.

Plotting reads saved evidence only and returns `(sensitivity.png, innovations.png)`
paths. Either preexisting chart is rejected before rendering. The diagnostics
CLI reads `--config` JSON and calls both functions; existing backtest commands
do not call them. These are development diagnostics, not backtest forecasts,
posterior predictive replications, or automatic model approval.

## Naming, errors, and limits

Dataset/tag names are case-sensitive: start with an ASCII letter/digit, followed
by ASCII letters, digits, underscores, or hyphens. Invalid options/checks generally raise
`ValueError`; wrong note/receipt types raise `TypeError`; filesystem failures
retain their usual types. Malformed saved records raise rather than being
silently skipped. Individual forecast/evaluation failures are saved as described
above instead of propagating to stop all later origins.

Use completed, stable inputs and one writer. Preserved/run files remain writable;
retrieval verifies input integrity but does not recover lost bytes. Move the
whole snapshot store, including manifests/events/tags. Git excludes local data
and run evidence. Remote storage, crash recovery, retention, and backups remain
caller concerns or future work. The Python model boundary is trusted code, not a
security sandbox. [Tests](../../tests/README.md) map claims to supporting evidence.
