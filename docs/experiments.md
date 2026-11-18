# Reusable forecasting experiments

The learned-observation-variance `LocalLevelVariance` adapter accepts selected
gapped histories and delayed issuance, scaling fixed state variation by elapsed
UTC days. It supports future horizons through its marginal mixture forecast.
The runner's daily calendar/release/final rules are unchanged. The original
fixed-variance `LocalLevel` still has its narrower daily next-observation timing
contract. Both evidence workflows accept `--model local-level-variance` with
the same development-only selection and final-boundary guard.

[Prior influence comparisons](prior-influence.md) also reuse development specs
and the shared `select_history_as_of` release selector. Their paths evaluate
each daily period plus the task's issuance delay, then the explicit as-of
cutoff. They include early observations before minimum history so prior effects
are visible; minimum history applies at the final cutoff. Actual vintage and
benchmarks do not score these comparisons. No model setting is selected
automatically, and backtests do not start comparisons.

The separate [predictive-check workflow](predictive-checks.md) can reuse this
specification for columns, calendar/release rules, minimum history and the final
boundary. It accepts development role only and an explicit as-of time before
`final_start`. It replicates the selected fitted history rather than scheduling
forecast cases; horizon, issuance delay, scoring vintage and benchmarks do not
produce check scores. Backtests do not invoke predictive checks automatically.

An experiment declares what a forecast means independently of the model and
any particular application. The same specification can run a point baseline,
the fixed-variance local-level model, or the parameter-learning `NormalMean` and
`NormalMeanVariance` adapters.
The preserved dataset/ID and model/settings are selected separately for each run.

The version-2 specification has three immutable records (version 1 remains readable):

| Record | Declares |
|---|---|
| `ForecastTask` | Target/source descriptions, columns, units, calendar/gap rule, horizon, minimum available history, issuance timing, release/revision rules, and scoring vintage |
| `EvaluationPlan` | Development or final role, optional final-period boundaries, and required point benchmarks |
| `EvidencePlan` | Required separate workflows, per-origin fit check names and evidence replication/seed/interval settings |

`ExperimentSpec` combines them and supports `to_dict()`, `from_dict(value)`, and
`load(path)`. Expanded settings are saved in every new run, including defaults.
This is task/evaluation infrastructure; it does not select a real application,
acquire data, choose priors, or choose an inference backend.

## Target, calendar, and issuance

The target is the numeric `target_column` at a future `timestamp_column` value,
in declared original units. Descriptions record caller intent, not verified
source provenance. The snapshot preserves the bytes and any acquisition context.

The implemented calendar is `daily_utc`: unique period timestamps must be ordered
and exactly 24 hours apart after UTC normalization. The phase need not be
midnight. `missing_policy="reject"` rejects missing periods or numeric values;
there is no sorting, deduplication, imputation, holiday calendar, or silent repair.
Timezone-aware timestamps are required.

An origin is one period timestamp. Issue at that timestamp plus
`issuance_delay_seconds` (integer 0–86399). Predict the period `horizon_days`
later (positive integer, one fixed horizon per run). Issuance does not imply that
the current period's value is already known. The minimum training count refers
to available distinct periods, not rows or releases.

The shared runner supports these coordinates independently of adapter support.
`TrainingMean` and `LastValue` use the supplied eligible values. The current
`LocalLevel` adapter still requires contiguous daily history, issuance exactly
at its last observation, and a target one day later. Delayed/gapped histories,
issuance offsets, or longer horizons can fail its requests visibly; the runner
does not silently substitute a different timing convention.
`NormalMean` and `NormalMeanVariance` accept any strictly ordered eligible past history and future target:
its exchangeable constant-mean assumption permits delayed/gapped histories and
longer horizons. Its optional sampler audit follows each immutable request;
see [parameter uncertainty](parameter-uncertainty.md).

## Availability and revisions

With `availability_column=None`, values are assumed available at their period
timestamps. With a named availability column, it records when each release or
revision became available. This is a caller-supplied historical claim, distinct
from when a file was downloaded or registered. A release cannot precede its
period timestamp.

Use `revision_policy="reject"` for one row per period. Use `"latest_available"`
with an explicit availability column for multiple vintages. Rows must be grouped
by increasing period timestamp; releases within a period must have strictly
increasing availability timestamps. Tied or unordered releases are rejected.
No source access or vintage reconstruction happens automatically.

At each origin, the runner includes only periods no later than that origin
and releases available **at or before issuance**. For each period it selects
the latest such release. For example:

| Period | Value | Available at |
|---|---:|---|
| January 1 | 1 | January 1, 00:00 UTC |
| January 1 | 101 | January 4, 00:00 UTC |
| January 2 | 3 | January 2, 12:00 UTC |

At January 2, 00:00, only the original January 1 value is eligible. At January 2,
12:00, the January 2 release is also eligible. The January 1 revision enters
only at or after January 4. An unreported internal period can produce a gap in
eligible history even though the complete period calendar is valid; adapters
must support such a request or fail explicitly.

Scoring uses a separate declared `actual_policy`: `"first_release"` or
`"latest_in_snapshot"`. The latter means latest in the selected preserved file,
not latest now at the source. This choice does not change training-vintage
selection. Actuals and their release times are joined only after all primary
and benchmark forecast files are written/closed.

## Development and final periods

Without final boundaries, only development evaluation is allowed. To reserve
a period, supply both aware `final_start` and `final_end`. Boundaries apply to
**target timestamps**:

- Development targets are strictly before `final_start`.
- Final targets are in `[final_start, final_end)`; this role requires an explicit
  caller invocation. Targets at/after `final_end` are excluded from both roles.
- Final evaluation is sequential: an earlier final observation may become
  training data for a later final forecast once available. It is never supplied
  to its own or an earlier forecast. Model/settings selection must precede final use.

`case_schedule.csv` records every candidate target and whether it is selected,
outside the requested role, or lacks the minimum available history. Zero selected
cases raises before creating a run directory. Selected cases receive the same
ordinary completion/failure accounting as other runs.

These boundaries enforce request/case selection; they do not certify that humans
have kept the final data untouched. The runner validates the complete input,
and trusted Python adapters can independently access files. Synthetic final-role
examples verify infrastructure only. They are not a new untouched scientific test
of already-inspected development fixtures. There is no tuning system or automatic
final run, approval, or release gate.

## Benchmarks on matching cases

`benchmarks=("training-mean", "last-value")` declares required point forecasts:
average the eligible history, or persist its latest value. The default is no
additional benchmarks, preserving the earlier example's execution choices.

Every declared benchmark receives exactly the primary model's selected history,
issuance, and target. Its predictions are frozen before scoring against the same
actual vintage. `scores.json.benchmarks` reports completion/failure counts and
point scores per benchmark, without choosing a winner. No uncertainty is
invented for these point-only models. Partial model/benchmark scores disclose
their own case counts and should not be compared as complete matching cohorts.
Any declared benchmark failure fails the overall run, even if primary point
scoring completed. Charts include saved benchmark lines without recalculation.

## Run the example

Prepare the sample snapshot/tag as in [the walkthrough](forecasting.md#prepare-the-input-explicitly).
The [example specification](../configs/daily_forecast_experiment.json) uses the
same 200 observations, reserves June 9 through July 18, and declares both point
benchmarks. Use unused output directories:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --experiment configs/daily_forecast_experiment.json \
  --output-directory runs/tutorial-experiment-development-001

PYTHONPATH=src .venv/bin/python scripts/run_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --experiment configs/daily_forecast_experiment.json --evaluation-role final \
  --output-directory runs/tutorial-experiment-final-001
```

Expect 130 development cases and 40 final cases with 30 minimum observations.
All three run scripts accept these flags. `--evaluation-role` overrides only the role;
it requires `--experiment`. Use the same permanent `--snapshot-id ID` instead
of a tag for an input shared across separate invocations. Final evaluation is a
separate caller action; the first command does not start it automatically.

The library equivalent is:

```python
from dataclasses import replace
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.backtest import run_backtest
from model_pipelining.local_level import LocalLevel

experiment = ExperimentSpec.load("configs/daily_forecast_experiment.json")
development_run = run_backtest(
    dataset="synthetic-mean-shift", tag="baseline-input",
    experiment=experiment, model=LocalLevel(),
    output_directory="runs/tutorial-library-experiment-development-001",
)
final_experiment = replace(
    experiment, evaluation=replace(experiment.evaluation, role="final"),
)
```

Constructing `final_experiment` does not execute it. To execute, make another
explicit `run_backtest(experiment=final_experiment, ...)` call. Library reporting
remains separate. Without an explicit experiment, existing calls still use the
immediate/unrevised daily development task and 30 minimum observations. The
legacy `minimum_history=` argument remains available in that mode. With an
experiment, choose minimum history in `experiment.task` and omit that argument;
conflicting interfaces raise before output creation.

## Caller-owned evidence requirements

`ExperimentSpec(evidence=EvidencePlan(...))` declares required separate
`predictive_checks`/`prior_influence` workflows and optional named per-origin fit
checks. Defaults require none and preserve earlier calls. Models still enforce
their own required numerical checks. `run_backtest` additionally enforces the
caller's requested fit names, saving missing/failed requirements before rejecting
forecasts; no fit capability cannot bypass a requested check. Benchmarks retain
their point-only role.

Invoke `run_experiment_evidence` or `scripts/run_evidence.py` explicitly to run
required separate development workflows on one pinned input/history. Their
replication count/seed and comparison interval come from the plan. Missing
capabilities/scenarios and mandatory checks fail the evidence run, while later
workflows continue. Forecasting records those separate requirements as pending
and does not execute them or certify human approval. The evidence runner does
not score forecasts or rewrite a saved forecast run. The [evidence guide](evidence.md)
defines fields, examples, status scope and artifact receipts.

## Evidence and compatibility

New runs use manifest format version 3; experiment specs now emit version 2.
Version-1 specs load with default empty evidence requirements. The manifest
saves expanded task/evaluation/evidence choices and a canonical specification
hash. `predictions.csv` adds training end and a hash of selected timestamp/value/
release triples; scoring adds `actual_available_at`. These identify vintage
selection without passing release tables or future outcomes to the adapter.
Requested benchmarks add `benchmark_predictions.csv` and
`benchmark_evaluation.csv`; their scores live in `scores.json.benchmarks`.

Reports still read version-1/2 artifacts. Comparisons normalize explicit version-1/2
experiment specifications, then require matching task/evaluation/evidence settings,
case coordinates/history fingerprints, inputs, and actual vintages.
Legacy runs without experiment/history metadata can compare with matching
legacy runs; comparisons to newer runs may reject insufficient or differing
metadata rather than infer missing task claims. No old artifacts are backfilled.

Replay a new run with its permanent input, `ExperimentSpec.from_dict(manifest["experiment"])`,
saved model configuration, interval levels, and matching source/environment.
Numerical artifacts repeat; timestamps differ. The source/hash records identify
settings and code, not authenticate or restore them. Input data, run evidence,
and matching locked environment must be retained explicitly.
