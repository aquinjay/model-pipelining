# Python API reference

Use this page to look up arguments, return types, and records. For a complete
example with explanations, start with [the forecasting walkthrough](../../docs/forecasting.md).
The [public NOAA example](../../docs/python312-public-forecast.md) reuses
`LocalLevel`, `run_backtest`, and `plot_run` on the focused Python 3.12 profile.
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
| [experiments.py](experiments.py) | Versioned task/evaluation/evidence requirements, verified development selection, as-of planning and actual vintages |
| [local_level.py](local_level.py) | Gaussian local-level adapter, filtering, prior simulation |
| [local_level_variance.py](local_level_variance.py) | Fixed evolution variance, learned observation variance, checked log-variance quadrature and conditional state inference |
| [sequential.py](sequential.py) | Freeze historical predictive laws and score PIT/errors on eligible development releases |
| [calibration.py](calibration.py) | Data-free posterior-CDF calibration, recovery and explicit failure evidence |
| [local_level_joint_variance.py](local_level_joint_variance.py) | Categorical W and continuous V joint learning, with full marginal forecasts |
| [innovation_reference.py](innovation_reference.py) | Eight-observation all-sequence small/large innovation reference, independent batch/adaptive and Kalman/quadrature calculations |
| [innovation_lattice.py](innovation_lattice.py) | Observation-conditioned state-grid proposal and probability intervals with explicit grid and tail error |
| [innovation_gibbs_check.py](innovation_gibbs_check.py) | Research sampler using a tridiagonal Gaussian precision solve, with an exact blocked label draw for short atom controls |
| [evidence_joint_level.py](evidence_joint_level.py) | Joint W/V/state replication and prior influence |
| [state_mixtures.py](state_mixtures.py) | Credible-state CDF/generalized quantiles with genuine probability atoms |
| [dynamics_study.py](dynamics_study.py) | Explicit dynamics comparisons on matched synthetic development cases |
| [mixtures.py](mixtures.py) | Weighted Gaussian PDF/CDF, marginal quantiles and numerically resolved disconnected HPD sets |
| [normal_mean.py](normal_mean.py) | Unknown constant mean, exact posterior/predictive integration, optional sampler audit |
| [normal_mean_variance.py](normal_mean_variance.py) | Joint unknown mean/variance, normal/inverse-gamma posterior, exact Student-t forecasts |
| [inference.py](inference.py) | Optional fitted-forecast/evidence records and runner-owned persistence |
| [sampling.py](sampling.py) | Independent emcee Metropolis chains, ArviZ diagnostics, exact-reference audits |
| [sampling_variance.py](sampling_variance.py) | Joint two-coordinate audit with Jacobian and moment-aware diagnostics |
| [evidence.py](evidence.py) | Provider protocol/metadata, replication records and structural validation |
| [evidence_normal.py](evidence_normal.py) | Exact conjugate simulation and prior/parameter/predictive summaries |
| [evidence_local_level.py](evidence_local_level.py) | Numerical local-level evidence, joint initial-state draws and fresh state paths |
| [evidence_workflows.py](evidence_workflows.py) | Explicit experiment-owned workflow execution, required outcomes and saved-only child plotting |
| [predictive_checks.py](predictive_checks.py) | Provider-based whole-dataset replication, descriptive discrepancies and saved-only plots |
| [prior_influence.py](prior_influence.py) | Provider-based prior/learning paths on common historical releases and descriptive reference differences |
| [diagnostics.py](diagnostics.py) | Explicit assumption influence and sequential criticism; plots from saved diagnostic paths |
| [uncertainty.py](uncertainty.py) | Shared output validation and compatible uncertainty evaluation |
| [backtest.py](backtest.py) | Daily history selection, model execution, frozen forecasts, scoring, checks |
| [distribution_scoring.py](distribution_scoring.py) | Frozen full-law validation, proper losses and saved score cohorts |
| [experiment_review.py](experiment_review.py) | Assemble compatible saved forecast/evidence/calibration for human review |
| [reporting.py](reporting.py) | Read saved artifacts and draw/compare results without fitting |
| [decision_value.py](decision_value.py) | Binary cost-loss and continuous asymmetric (newsvendor) expenses, relative economic value, and tail-concentration summaries |
| [decisions.py](decisions.py) | Model-independent `DecisionSpec`, frozen per-origin recommendation issuance, separate outcome scoring, saved-only charts, and paired comparisons |
| [robust_scoring.py](robust_scoring.py) | Robust error/loss summaries, worst-5% tail concentration, and deterministic Bartlett HAC paired-difference diagnostics |
| [probabilistic_baselines.py](probabilistic_baselines.py) | Zero-tuning parametric (`GaussianClimatology`, `ResidualPersistence`) and nonparametric (`EmpiricalClimatology`, `EmpiricalResidualPersistence`) probabilistic baseline adapters |
| [harmonic_regression.py](harmonic_regression.py) | Weekly Fourier-pair least-squares point forecasts for a second model family |
| [temperature_regression.py](temperature_regression.py) | Scalar annual Fourier and lag regressions with restorable settings |
| [noaa.py](noaa.py) | Validate a caller-supplied metric NOAA daily TMIN or TMAX archive |
| [temperature_evaluation.py](temperature_evaluation.py) | Match and summarize saved scalar temperature forecasts |
| [artifacts.py](artifacts.py) | Shared JSON/CSV writers for saved manifests, checks and tables |
| [adapters.py](adapters.py) | Name-to-class catalog that scripts use to rebuild adapters from saved settings |

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
| Experiment specification | Version-2 task/evaluation/evidence requirements; version 1 loads with default evidence requirements |
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
inside `run_backtest`. The four run scripts call reporting automatically; the
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

Import `ForecastTask`, `EvaluationPlan`, `EvidencePlan`, and `ExperimentSpec` from
`model_pipelining.experiments`. All are frozen keyword-only dataclasses.
`ForecastTask()` supplies the original daily immediate/unrevised task with
30 minimum observations. `EvaluationPlan()` supplies development-only evaluation
with no extra benchmarks. `EvidencePlan()` requires no separate workflows or
named fit checks. `ExperimentSpec(task=..., evaluation=..., evidence=...)` combines
them. See [fields, defaults, semantics, and examples](../../docs/experiments.md).

```text
ExperimentSpec.to_dict() -> dict
ExperimentSpec.from_dict(value) -> ExperimentSpec
ExperimentSpec.load(path) -> ExperimentSpec
read_task_data(path: Path, task: ForecastTask) -> tuple[ObservationVersion, ...]
read_observations(path: Path) -> tuple[Observation, ...]
select_history_as_of(versions, *, issued_at, period_end=None) -> tuple[ObservationVersion, ...]
plan_cases(versions, experiment) -> (tuple[ForecastCase, ...], list[dict])
select_actuals(versions, task) -> dict[str, ObservationVersion]
select_development_history(*, dataset, issued_at, snapshot_id=None, tag=None,
                           store="data/versioned", experiment=None)
    -> (ExperimentSpec, tuple[ObservationVersion, ...], ForecastCase, dict)
```

`load` reads version-1/2 JSON explicitly; unknown constructor fields/versions or
invalid settings raise. `to_dict` emits version 2 with expanded defaults, UTC
boundary timestamps and benchmark/requirement names as lists. `from_dict` restores
them; version 1 receives a default empty `EvidencePlan`.
No function fetches data, executes a model, or automatically starts final evaluation.

The parsing/planning functions support inspection without executing a run.
`read_task_data` validates the ordered daily CSV and release table;
`ObservationVersion` holds `timestamp`, `value`, and `available_at`.
`read_observations` reads the default immediate, unrevised task as plain
`Observation` records; local-level diagnostics use it. Pass its
validated records to `plan_cases`. A `ForecastCase` holds `origin_index`,
`issued_at`, `target_time`, and selected `versions`; `.history` returns immutable
model observations and `.history_sha256` hashes timestamp/value/release triples.
`select_history_as_of` selects latest available releases from a validated,
ordered release table. It requires aware times; periods must be no later than
`period_end` (default issuance), and releases no later than issuance. A period
cutoff later than issuance is rejected. The shared helper serves backtests,
predictive checks and prior comparisons.

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
| `Forecast` | `mean`, `quantiles=()`, `highest_density_regions=()`, `distribution=None` |
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
NormalMean.from_configuration(configuration) -> NormalMean
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
*, identity, required_checks=()) -> bool` validates/persists records and returns
success for model-required and caller-named checks; the runner supplies
`experiment.evidence.required_fit_checks` automatically. Direct library callers choose
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
NormalMeanVariance.from_configuration(configuration) -> NormalMeanVariance
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

## Learned local-level observation variance

Import from `model_pipelining.local_level_variance`:

```text
QuadratureSettings(*, minimum_nodes=32, maximum_nodes=1024,
                   relative_tolerance=1e-6, tail_tolerance=1e-8,
                   maximum_expansions=9)
LocalLevelVariance(*, initial_mean=0, initial_variance=100,
                   evolution_variance=0.1, prior_shape=3, prior_scale=2,
                   quadrature=QuadratureSettings())
LocalLevelVariance.posterior(request) -> VariancePosterior
LocalLevelVariance.prior() -> VariancePosterior
LocalLevelVariance.predictive_distribution(request) -> NormalMixture
LocalLevelVariance.fit(request) -> FittedForecast
LocalLevelVariance.forecast(request) -> Forecast
LocalLevelVariance.from_configuration(configuration) -> LocalLevelVariance
conditional_filter(model, history, variances, *, paths=False) -> tuple
initial_posterior_components(posterior) -> (initial_means, initial_variances)
```

The initial/evolution variances are nonnegative fixed squared-unit values,
independent of the unknown observation variance. Prior shape/scale are positive
and finite; the initial mean is finite. Histories must be nonempty, aware,
strictly ordered, finite and no later than issuance; targets must be after
issuance. Transitions use elapsed UTC days, including gaps and fractional
projection times. The initial prior is one day before the first selected value.

`VariancePosterior` retains model/history, variance nodes/weights, conditional
level means/variances, log support/evidence and precision metadata. Properties
give integrated level mean/variance and nullable expected observation variance.
`.cdf(value)`/`.ppf(probability)` evaluate observation-variance probabilities
using additional quadrature; `.predictive(elapsed_days=1)` returns the finite
numerical mixture. That finite representation has finite moments even when the
source-model predictive variance is undefined; the fit metadata preserves nulls
for undefined source moments. Numerical quadrature tails/refinement must pass.
`NumericalIntegrationError` carries failed evidence. Direct `fit` retains
failed candidates; `forecast` rejects failed required checks.

`NormalMixture(weights, means, variances)` lives in `mixtures.py`. It exposes
`.mean`, `.variance`, `.pdf(x)`, `.cdf(x)`, `.ppf(p)`, and
`.highest_density(level, tolerance=1e-7) -> (HighestDensityRegion, dict)`.
PDF/CDF accept scalars or arrays. Weights normalize; zero weights are discarded.
HPD regions retain disconnected segments and require agreement after grid
refinement. This is explicit numerical evidence, not an arbitrary-mixture
completeness proof. See [inference, tolerances and limits](../../docs/local-level-variance.md).

## Prior and posterior predictive checks

Import these functions and records from `model_pipelining.predictive_checks`:

```text
simulate_prior_predictive(model, *, observation_count, replications=1000, seed=0)
    -> PredictiveReplications
simulate_posterior_predictive(model, request, *, replications=1000, seed=0)
    -> PredictiveReplications
dataset_statistics(values) -> dict[str, dict]
run_prior_predictive_checks(*, model, observation_count, output_directory,
                            replications=1000, seed=0) -> Path
run_predictive_checks(*, dataset, model, issued_at, output_directory,
                      snapshot_id=None, tag=None, store="data/versioned",
                      experiment=None, replications=1000, seed=0) -> Path
plot_predictive_checks(check_directory) -> tuple[Path, ...]
```

Supply a concrete `NormalMeanVariance` with `mcmc=None`; these are exact joint
draws, not MCMC. Replication count and prior-only observation count are positive
integers; seed is an integer in `[0, 2**32)`. Posterior dataset length equals the
request history length. One shared sampled mean/variance generates every value
in a dataset; replications use independent PCG64/SeedSequence streams.

`ReplicatedDataset` fields are `replication`, `mean`, `observation_variance`, and
`values: tuple[float, ...]`. `PredictiveReplications` fields are `kind`, `seed`,
`requested_replications`, `observation_count`, `distribution: dict`,
`datasets: tuple[ReplicatedDataset, ...]`, and `failures: tuple[dict, ...]`.
Inspect failures even when the simulation function returns normally. Direct
helpers do not persist. `dataset_statistics` returns each fixed statistic as
`{"value": float | None, "unavailable_reason": str | None}`; invalid numeric
calculations raise, while mathematically undefined statistics retain reasons.

Persisted prior-only checks require no input. Observed checks need one snapshot
selector and an aware explicit `issued_at`; they select latest as-of releases
and enforce minimum history. Only development-role `ExperimentSpec`s are
allowed, and issuance must precede any declared final boundary. Unused forecast
scheduling/scoring settings are documented in [the workflow](../../docs/predictive-checks.md).

The workflows save simulation/statistic evidence and pending review. Numerical
failures are retained, later independent replications/phases continue, and
required failures fail computation. Undefined quantities and large observed
discrepancies do not. I/O failures propagate and may leave running output.
Existing directories/charts are rejected. Library plotting is explicit;
`scripts/check_normal_mean_variance.py` invokes it automatically. Plotting reads
saved files only. Backtests do not call this workflow. See [statistics, scope,
commands and replay](../../docs/predictive-checks.md).

These functions also accept a concrete `LocalLevelVariance`. Posterior variance
nodes are sampled by their checked numerical weights, then a joint initial
state is drawn using conditional Gaussian smoothing. Replicated states use
fresh process innovations; fitted state paths are not copied. `ReplicatedDataset`
adds `initial_level: float | None=None` and `latent_levels: tuple=()`; its `mean`
is nullable and empty for this model. Gaps use selected elapsed time. Artifacts
record numerical integration separately from independent-replication MCSE.
Select this mode with `--model local-level-variance` in the existing check CLI.

## Prior influence comparisons

Import from `model_pipelining.prior_influence`:

```text
run_prior_influence(*, dataset, reference, scenarios, issued_at, output_directory,
                    snapshot_id=None, tag=None, store="data/versioned",
                    experiment=None, interval_level=0.95) -> Path
plot_prior_influence(comparison_directory) -> tuple[Path, Path]
```

Supply concrete `NormalMeanVariance` models with `mcmc=None`, one reference and
a nonempty dictionary of named alternatives. Names must be nonblank and cannot
be `reference`. `interval_level` must be a representable probability strictly
between 0 and 1. Input/development cutoff requirements match predictive checks.
Minimum history is required at the final cutoff; the path starts with the
empty-history prior and the first available observation. Daily updates use
period+issuance delay and only then-available releases. A differing final
as-of time is appended. Revisions trigger a refit without rewriting past rows.

All configurations use the same selected histories. Saved paths contain exact
posterior parameters, mean/variance/predictive summaries and intervals,
conditional prior-location coefficients, scale components, status/errors and
signed reference differences. Undefined moments/differences are explicit nulls.
Mean/predictive Student-t intervals are HPD; variance parameter intervals are
equal-tail. Per-update calculation failures retain evidence and later cases
continue. Missing references never invent differences. Large influence is not
a failure, and no model/setting is selected automatically.

Library plotting is explicit; `scripts/compare_priors.py` calculates then plots
automatically. Charts read saved files only. Output directories/charts must be
unused; I/O failures propagate. See [quantities, timing, example configs and
replay](../../docs/prior-influence.md).

Prior comparisons also accept matching `LocalLevelVariance` instances. They
must share evolution variance and quadrature settings. Initial mean/variance
and observation-variance prior shape/scale may change. State/variance bounds
are equal-tail credible intervals; predictive bounds preserve mixture HPD
segments as JSON/counts, with total segment width. Inapplicable static fields
(conjugate parameters/degrees of freedom/location coefficient/scale components)
remain null. `integration_checks.json` saves numerical evidence per update.
The comparison CLI selects this with `--model local-level-variance`. See
[dynamic model details](../../docs/local-level-variance.md).

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

## Evidence providers and explicit experiment evidence

`NormalMeanVariance.evidence_provider()` (exact mode) and
`LocalLevelVariance.evidence_provider()` implement `evidence.EvidenceProvider`.
`EvidenceMetadata` declares the supported workflows, source files, simulation
parameter fields, numerical evidence and presentation. Providers implement
`simulate(*, observation_count, replications, seed, request=None)`,
`summarize(case, level)` and `validate_scenarios(models)`; shared workflows do
not dispatch on model classes. The existing records cover mean/variance/state
summaries, rather than arbitrary parameter types. `ReplicatedDataset` and
`PredictiveReplications` now live in `evidence.py` and remain importable from
`predictive_checks.py` for existing callers.

```text
get_evidence_provider(model, workflow) -> EvidenceProvider
assess_checks(checks, *, required_names=()) -> (bool, tuple[dict, ...])
save_fit_evidence(result, directory, *, identity, required_checks=()) -> bool
run_experiment_evidence(*, dataset, model, issued_at, experiment, output_directory,
                        snapshot_id=None, tag=None, store="data/versioned", scenarios=None) -> Path
plot_experiment_evidence(evidence_directory) -> tuple[Path, ...]
```

The last two functions are in `evidence_workflows`; the check/persistence helpers
are in `inference`. `run_experiment_evidence` requires nonempty
`experiment.evidence.required_workflows` and uses its replication/seed/interval
settings. It pins input selection once and checks child identity/settings and
mandatory checks. Missing capability/scenarios or failed checks are retained as
failed required outcomes; I/O propagates. Library plotting is explicit and reads
saved metadata only. `scripts/run_evidence.py` performs both calls automatically.
Standalone workflows keep their explicit options and do not certify the entire
plan. Per-origin fit names are enforced by `run_backtest`, while separate workflow
requirements remain pending in forecast manifests. See [the evidence guide](../../docs/evidence.md)
for all `EvidencePlan` fields, statuses, compatibility and examples.

## Weekly harmonic regression

Import `WeeklyHarmonicRegression` from `model_pipelining.harmonic_regression`.

```text
WeeklyHarmonicRegression()
WeeklyHarmonicRegression.forecast(request) -> float
WeeklyHarmonicRegression.from_configuration({"period_days": 7, "harmonics": 1})
```

Each forecast refits an intercept plus one seven-day sine/cosine pair by least
squares on the eligible history. The phase starts at the first eligible
observation, and the fitted curve is evaluated at the target time. It returns a
point forecast only, with no quantiles or highest-density regions. A history
with fewer than three distinct weekly phases raises `ValueError`, and the
runner records that case as failed. The period and harmonic count are fixed.
`scripts/run_harmonic_example.py` runs it against the training mean on a
generated weekly series; see the [reading guide](../../docs/reading-guide.md#run-the-weekly-harmonic-example).

## Rebuilding adapters from saved settings

Every repository adapter (`NormalMean`, `NormalMeanVariance`, `LocalLevel`,
`LocalLevelVariance`, `LocalLevelJointVariance`, `WeeklyHarmonicRegression`) provides
`from_configuration(configuration)`. It accepts the JSON form of the adapter's
`configuration` property, such as `manifest["model"]["configuration"]`, and
rebuilds nested settings (`MCMCSettings`, `QuadratureSettings`, tuples). The
constructor's validation still applies.

```text
adapters.MODEL_TYPES -> dict[str, type]            # keyed by each adapter's name
adapters.EVIDENCE_MODEL_NAMES -> tuple[str, ...]   # adapters with evidence providers
adapters.build_model(name, configuration) -> adapter
adapters.build_scenarios(reference, overrides) -> dict[str, adapter]
```

`build_scenarios` merges each named override dictionary into the reference
configuration and restores it, so unchanged settings, including MCMC and
quadrature settings, are retained. Scripts that take `--model` and `--config`
use this catalog. The runner and evidence workflows never consult it; they
accept any object that satisfies the `Forecaster` or `EvidenceProvider`
contract. Register a new adapter in `adapters.py` only to construct it by name
from scripts.

## Shared names and artifact writers

Each name list has one definition:

```text
models.BENCHMARK_MODELS -> {"training-mean": TrainingMean, "last-value": LastValue}
experiments.EVIDENCE_WORKFLOWS -> ("predictive_checks", "prior_influence", "sequential_checks")
evidence.PROVIDER_WORKFLOWS -> EVIDENCE_WORKFLOWS + ("calibration",)
evidence_workflows.WORKFLOW_MANIFESTS -> {workflow: "<workflow>.json"}
evidence_workflows.REQUIRED_WORKFLOW_CHECKS -> {workflow: required check names}
predictive_checks.tail_summary(count, total) -> dict
```

`EvaluationPlan.benchmarks`, `EvidencePlan.required_workflows` and provider
metadata are validated against these tables, and the runner builds benchmark
models from `BENCHMARK_MODELS`. Invalid benchmark entries in a saved
experiment raise `ValueError` with the accepted names.

```text
artifacts.write_json(path, value, *, overwrite=False) -> None
artifacts.write_csv(path, fields, rows) -> None
```

Both write UTF-8. `write_json` saves indented, finite JSON with a trailing
newline. It serializes before opening the file, so a rejected value leaves no
file behind. New files are created exclusively; workflows pass
`overwrite=True` only to replace their own manifest when they finish.
`write_csv` creates a new file with the given header order. Run and evidence
manifests list `artifacts.py` among their source hashes.

## Temperature regression and NOAA inputs

`TemperatureRegression(name)` accepts `annual-only`, `lag-only`, and
`annual-plus-lag`. It fits a point forecast from the eligible contiguous daily
history and requires the target to be one day after its last observation.
`TemperatureRegression.from_configuration(configuration)` rebuilds the
variant saved in a runner manifest. The fixed annual period is 365.2425 days,
the phase origin is 2023-01-01, and the lag is one day.

```python
from datetime import date
from pathlib import Path

from model_pipelining.noaa import daily_temperatures
from model_pipelining.temperature_regression import TemperatureRegression

source = Path("noaa-extrema.csv").read_bytes()
minimum_rows = daily_temperatures(
    source, station="USW00094728", start=date(2023, 1, 1), end=date(2025, 12, 31), element="TMIN"
)
model = TemperatureRegression("annual-plus-lag")
restored = TemperatureRegression.from_configuration(model.configuration)
```

`daily_temperatures` accepts `TMIN` or `TMAX` and returns date-sorted
`(timestamp, value)` pairs for the inclusive calendar. It checks station,
completeness, duplicates, finite values and the selected element's quality
flags. Supply metric data with attributes, then write and register normalized
rows explicitly if the backtest needs a snapshot. The UTC labels support
simulated availability.

In [temperature_evaluation.py](temperature_evaluation.py),
`read_evaluated_predictions(path, dates, values, *, benchmark=False)`
returns a NumPy array ordered by the declared dates after checking status,
dates, actuals, and finite predictions. Set `benchmark=True` to require a
last-value row for each date. `require_matched_origins(reference, counterpart,
dates)` checks saved model and persistence origins across both run
directories. `temperature_error_metrics(dates, values, predictions)` returns
a dictionary of point-error summaries by model. Supply actuals by date,
including the day preceding each evaluated date, and prediction arrays in
the supplied date order.

The [range-probability study](../../docs/noaa-range-probability.md) uses these
helpers to match saved scalar forecasts and computes event probabilities in
the study script.

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

## Sequential predictive checks

`sequential.run_sequential_checks(*, dataset, model, issued_at, output_directory, snapshot_id=None, tag=None, store="data/versioned", experiment=None) -> Path` freezes provider-supplied observation laws before scoring. `plot_sequential_checks(directory) -> tuple[Path, ...]` reads saved results. Providers implement `predictive(request) -> dict`; `EvidencePlan` accepts `sequential_checks`. See [selection, moment and failure rules](../../docs/sequential-checks.md).

## Simulation calibration

`calibration.run_calibration(*, model, observation_count, output_directory, trials=200, seed=0, interval_level=.95) -> Path` uses provider `calibration_draw(count, generator)`, `calibration_fit(request, truth, level)` and `calibration_parameters`. `get_evidence_provider(model, "calibration")` raises `ValueError` unless `calibration_parameters` is a nonempty tuple of unique nonblank names. `plot_calibration(directory) -> Path` reads saved records. The CDF-based workflow runs explicitly outside historical experiments. See [semantics and failure rules](../../docs/calibration.md).

## Joint local-level variances

`local_level_joint_variance.LocalLevelJointVariance` has Gaussian initial-state and inverse-gamma V settings plus `evolution_variances=(0,.01,.1,1)` and optional positive `evolution_weights` (uniform by default). It supplies `prior`, `posterior`, `fit`, `forecast`, `predictive_distribution`, `from_configuration` and sequential evidence. `JointVariancePosterior` integrates conditional scalar fits and exposes V CDF/PPF, W mass/quantiles, state moments, predictive mixtures and V/W covariance. `plot_joint_variances(run_directory)` reads saved fits. `dynamics_study.run_dynamics_study` compares caller-supplied W settings on controlled matched fixtures; plotting is explicit in Python. See [the full contract](../../docs/joint-variance.md).

## Joint generative evidence

`JointLevelEvidence` now implements `predictive_checks`, `prior_influence` and `sequential_checks`. `ReplicatedDataset.evolution_variance` is optional for compatibility and required by joint provider metadata. `StateMixture` supplies CDF/generalized quantiles for Gaussian states with genuine zero-variance atoms. Prior summaries add categorical W mass/quantiles/covariance; saved presentation metadata enables a third prior chart. See [field meanings and commands](../../docs/joint-evidence.md).

## Discrete and mixed calibration

Joint observation-variance quantiles verify their conditional-quantile bracket
signs and widen a failing side to checked integration support before solving the
marginal CDF. This handles nearly dominant categorical components without
relaxing numerical acceptance; see [details](../../docs/joint-variance.md).

The joint provider now supports `calibration`. It excludes fixed parameters and returns left/inclusive CDF bounds and actual closed-interval mass. `calibration_transform(item, *, seed, trial, parameter, interval_level)` randomizes only genuine jumps using independent parameter-labelled streams. Shared summaries retain randomized counts/mass and all failure records. Continuous CDF values and dataset seeds are unchanged. See [the contract and limits](../../docs/mixed-calibration.md).

## Saved experiment review

Import `build_experiment_review` from `model_pipelining.experiment_review`:

```python
build_experiment_review(
    forecast_directory="runs/forecast", output_directory="runs/review",
    evidence_directory="runs/evidence",
    calibration_directories=("runs/calibration",), require_calibration=True,
)  # returns the absolute output Path
```

Sources must already exist. A finished version-3 forecast anchors identity. The
function rechecks saved requirements/receipts and writes Markdown/JSON plus a
pending review; it performs no fitting or source mutation. See [the report
contract](../../docs/experiment-review.md) for failure statuses and scope.

## Full distribution losses

`Forecast(mean, quantiles=(), highest_density_regions=(), distribution=None)` now
accepts a serialized observation law. A model declaring
`supports_distribution=True` must supply it at every successful origin. The runner
validates agreement with mean/quantiles/region mass, freezes it before scoring and
requires complete CRPS/log-loss evaluation. Legacy capability omission means
unsupported. Current Bayesian fits supply their own law without a second fit.

`distribution_scoring.score_distribution(record, actual)` returns finite `crps`
and `log_score` losses from Student-t or Gaussian-mixture records, with explicit
failures for unsupported moments/precision. `compare_runs` adds complete paired
loss summaries and a chart when both runs declare distributions; review reports
also display these losses. See [the full contract](../../docs/distribution-scoring.md).

## Bounded innovation reference API

Import `InnovationMixtureReference`, `ReferenceSettings`,
`ReferencePrecisionError` and `run_innovation_reference` from
`model_pipelining.innovation_reference`:

```python
from model_pipelining.innovation_reference import (
    InnovationMixtureReference, ReferenceSettings, run_innovation_reference,
)

result = run_innovation_reference(
    (0., .1, -.1, 3.8, 4.1, 3.9),
    model=InnovationMixtureReference(small_variance=.01, large_variance=4., jump_probability=.05),
    settings=ReferenceSettings(),
    predictive_points=(-2., 0., 2., 4., 6.),
    state_points=(0.,), variance_points=(.5, 1., 2.),
)
```

Model/settings are optional and default to those records' defaults. Inputs are
finite daily values, zero to eight observations; query grids are finite,
strictly increasing tuples/lists of up to 64 points. V queries must be positive.
`prior_shape > 1`; initial/small variances may be zero. The returned dictionary
contains both numerical results, every positive-prior sequence, moment/CDF/density
comparisons, precision/tail evidence and a required agreement check. Invalid
inputs raise; unresolved computation raises `ReferencePrecisionError` with
`.evidence`. No `Forecaster`, fit-evidence provider or forecast serializer is added.

`enumerate_sequences(model, observation_count)` returns integer labels of shape
`(sequence_count, observation_count)` and their log prior probabilities. Probability
zero/one retains only the genuinely possible past sequence; coincident variances
retain all labels at an interior probability. See [the model, defaults, CLI and
feasibility limits](../../docs/innovation-reference.md).

## Innovation particle API

Import `ParticleSettings`, `ParticlePrecisionError`, and
`run_innovation_particles` from `model_pipelining.innovation_particles`.
`run_innovation_particles(values, *, model=None, settings=None,
predictive_points=(-2.,0.,2.,4.,6.), state_points=(0.,), variance_points=(.5,1.,2.))`
accepts zero to 512 daily finite values and the same model/query types as the
bounded oracle. It returns a JSON-compatible dictionary with posterior probes,
moments/dependence, independent-run MCSE, all attempted particle refinements,
raw replicate summaries and ESS/ancestry diagnostics. Short histories also report
all estimated path probabilities. No long-history path enumeration occurs.

Settings default to 512..4,096 particles, 16 independent replicates, seed 0,
resampling fraction .5, scaled tolerance .05, confidence .95 and minimum
replicate-evidence fraction .5. Counts double and require at least two levels.
Stochastic fits require `prior_shape > 2`. The prior and Gaussian limits
automatically use checked native scalar quadrature, retaining shape > 1 and
`mcse: null`. Invalid inputs raise `ValueError`; unresolved numerical or empirical
precision raises `ParticlePrecisionError` with complete `.evidence`.

Acceptance concerns empirical independent-run diagnostics at the requested probes.
It certifies neither missing posterior mass nor rare-path coverage; an unvisited
path can have estimated probability and MCSE both zero. The frozen study retains
such oracle failures. No forecasting/evidence interface is added. See
[the full method, example and study](../../docs/innovation-particles.md).

## Conservative innovation probability API

`bound_innovation_probabilities_lattice(values, *, model=None, settings=None,
predictive_points=(-2., 0., 2., 4., 6.), state_points=(0.,), variance_points=(.5,
1., 2.))` accepts the same `ProbabilityBoundSettings` as
`bound_innovation_probabilities`. It automatically builds the proposal, draws
and returns intervals, grid and tail diagnostics, and descriptive moment and
density estimates. Callers inspect `status` or explicitly invoke the study
script for a frozen comparison. See [derivation, limits and study
usage](../../docs/innovation-lattice.md).

Import `ProbabilityBoundSettings` and `bound_innovation_probabilities` from
`model_pipelining.innovation_probability_bounds`. The function accepts the same
`values`, fixed `model`, and predictive/state/V query grids as the particle API;
its separate `settings` specifies one fixed sample count, seed, confidence and
tolerance. It automatically draws independent bounded importance samples and
returns simultaneous probability intervals, normalizer bounds, sufficient
statistics, proposal-visit and underflow counts. No observed visits still yield
positive uncertainty. A nonpositive normalizer lower bound or overly wide
intervals returns `status: unresolved`; the caller must inspect it. Invalid or
nonfinite calculations raise `ValueError`. There is no automatic retry or change
to `run_innovation_particles`, and these intervals are not replacement MCSEs.
They do not bound densities, unbounded/joint moments or all points of a CDF.
See [signature, derivation, confidence scope and frozen results](../../docs/innovation-probability-bounds.md).

## Repository verification workflow

The explicit [development review study](../../docs/model-review-study.md) composes
the existing library workflows through `scripts/review_model_study.py`; it adds
no model capability or library runner stage. Its saved-only aggregation validates
complete matching cohorts before deriving PIT, window summaries and paired losses.

[GitHub Actions](../../docs/continuous-integration.md) runs the locked environment,
complete tests and `scripts/check_documentation.py --walkthrough`. The checker
parses examples by default; the explicit flag also executes the forecasting guide
in temporary storage. CI adds no library/runtime configuration or model approval
rule. Large scientific evidence exercises remain explicit calls.

## Saved decision layer

[decision_value.py](decision_value.py) supplies pure cost-loss calculations:

```text
decide(probability, alpha) -> bool
expense(acted, event, alpha) -> float
evaluate_decisions(cases, *, alpha, base_rate) -> dict
value_curve(cases, *, alphas, base_rate) -> list[dict]
asymmetric_quantile_level(under_cost, over_cost) -> float
asymmetric_expense(action, actual, *, under_cost, over_cost) -> float
evaluate_asymmetric_decisions(cases, *, under_cost, over_cost, baseline_action) -> dict
summarize_expenses(expenses) -> dict
brier_decomposition(cases, *, base_rate, bins=10) -> dict
binary_murphy_curve(cases, *, alphas) -> list[dict]
quantile_murphy_curve(cases, *, quantile_level, thresholds) -> list[dict]
mean_murphy_curve(cases, *, thresholds) -> list[dict]
murphy_dominance(first_curve, second_curve, *, tolerance=1e-12) -> dict
```

[decisions.py](decisions.py) separates forecast-time decision issuance from
post-outcome scoring without calling or refitting any model adapter:

```text
DecisionSpec(..., policy="binary_cost_loss" | "continuous_asymmetric", ...)
distribution_event_probability(record, *, threshold, comparison=">=") -> float
distribution_quantile(record, probability) -> float
event_indicator(value, *, threshold, comparison=">=") -> int
issue_decisions_from_run(*, run_directory, spec, output_directory) -> Path
issue_decisions_from_probabilities(*, source_directory, method, spec, output_directory, ...) -> Path
load_outcomes(outcomes_source) -> tuple[dict[str, dict], dict]
score_decisions(*, issued_directory, outcomes_source, output_directory, sensitivity_alphas=None) -> Path
plot_decision_scores(scored_directory) -> Path
compare_decision_scores(first_scored, second_scored, *, output_directory) -> Path
```

Issuance reads only forecast-time artifacts (`manifest.json`, `predictions.csv`,
`predictive_distributions.json`, or forecast-time columns of a saved probability
CSV) and writes immutable `decision_spec.json`, `recommendations.csv`, and
`decision_manifest.json` files. `score_decisions` verifies issued digests,
links realized outcomes in a separate directory without modifying issued files,
and reports robust expense concentration (`median_expense`, `p90_expense`,
`p95_expense`, `max_expense`, `top_5pct_expense_share`). `compare_decision_scores`
verifies identical specs, targets, and outcomes and saves paired expense
differences with win/loss/tie rates, action agreement, worst-5% difference share,
and naive plus Bartlett HAC standard errors. Library callers invoke
`plot_decision_scores` explicitly; `scripts/run_decisions.py score` draws it
automatically unless `--no-plot` is supplied. See [the decision guide](../../docs/decision-value.md).

## Robust evaluation summaries and probabilistic baselines

[robust_scoring.py](robust_scoring.py) provides deterministic, stdlib-only
summaries shared by `reporting.py`, `experiment_review.py`, and `decisions.py`:

```text
linear_quantile(ordered, probability) -> float
newey_west_se(differences) -> tuple[float | None, float | None, int]
summarize_loss_series(losses, *, nonnegative=True) -> dict
summarize_point_errors(rows) -> dict
summarize_paired_differences(differences) -> dict
```

`compare_runs` automatically populates `point_comparison` (including mean Murphy
dominance) and paired CRPS / log-score stability diagnostics (win rates,
worst-5% difference concentration, and Bartlett HAC SEs), writes a readable
`comparison.md` alongside `comparison.json`, and plots a cumulative CRPS
difference panel in `distribution_scores.png`. `compare_decision_scores`
similarly writes `decision_comparison.md`, `decision_comparison.json`, and a
3-panel `decision_comparison.png` with a Murphy diagram. `build_experiment_review`
surfaces robust point, distribution, and uncertainty summaries automatically and
accepts optional `decision_directories=()` and `require_decisions=False` to
verify and summarize saved decision evaluations alongside forecast, evidence,
and calibration runs. See [the evaluation and decision tools guide](../../docs/evaluation-and-decisions.md).

[probabilistic_baselines.py](probabilistic_baselines.py) supplies four
zero-tuning baseline adapters registered in [adapters.py](adapters.py) and
`PROBABILISTIC_BENCHMARKS`:

```text
GaussianClimatology(*, window=None)           # Student-t, df=n-1, HPD+quantiles+distribution
ResidualPersistence(*, window=None)           # Student-t, df=m matched lead diffs, HPD+quantiles+distribution
EmpiricalClimatology(*, window=None)          # empirical_samples, quantiles+distribution (no HPD/log_score)
EmpiricalResidualPersistence(*, window=None)  # empirical_samples of y_last + e_i, quantiles+distribution
```

Invoke them through `run_backtest` or explicitly via
`scripts/run_probabilistic_baseline.py --model <name>`. `distribution_scoring.py`
also exports `empirical_cdf_bounds(record, x)` and `empirical_quantile(record, probability)`
for validated `empirical_samples` laws (exact $O(k \log k)$ CRPS, `log_score=None`).
