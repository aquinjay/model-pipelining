# Shared forecast uncertainty and highest-density regions

This page explains what the saved uncertainty means, then gives the detailed
adapter/storage contract. Start with [the walkthrough](forecasting.md) for a run
and [the model guide](local-level.md) for Bayesian calculations. The shared output
supports a mean, observation quantiles, and highest-density prediction regions.
Highest-density regions are the preferred chart display when supplied; the
average model remains point-only.

## What the uncertainty describes

Every shared quantile and region refers to the **future observed target value**
at the request's target time, in its original units. For the Bayesian local-level
adapter this is the posterior predictive distribution. It is distinct from the
posterior distribution of the hidden current level and from uncertainty about a
model parameter. Those would require separate diagnostic outputs.

The local-level model supplies a normal predictive distribution with:

```text
mean = updated level mean
variance = updated level variance + daily evolution variance + observation variance
```

Including only the updated level variance would understate observation forecast
uncertainty. All results remain conditional on the fixed initial prior and
variance settings; uncertainty about the variance parameters is not integrated out. No settings are
tuned to obtain particular coverage or score values.

## Read a quantile or region

A **quantile** is a value associated with a probability. A model's 0.9 quantile
is its 90th percentile: at least 90% of its predictive probability lies at or
below it. For continuous laws such as the current Gaussian model, an 80% central
equal-tail interval uses the 0.1 and 0.9 quantiles and leaves 10% in each tail.
Discrete laws can have tied quantiles and more than the nominal mass inside.

A **highest-density region** instead collects the most probable values by density
until the requested mass is reached. A hypothetical forecast can have mean 10
and a 95% region from 6 to 14: that region contains 95% of the model's predictive
probability, conditional on its assumptions. It need not be centered on the mean,
and a multimodal law can require separate segments.

Two different summaries must stay distinct:

| Summary | Question answered |
|---|---|
| Case completion (`expected_count`, `scored_count`, `failed_count`) | How many requested forecast cases were successfully evaluated? |
| Empirical region coverage (`coverage`) | Among those evaluated cases, what fraction of actual observations fell inside the saved region? |

A completed computation does not establish that predictive probabilities are
well calibrated. A model's requested 95% mass and its empirical coverage over a
finite development series are different quantities. The toolkit records both;
it does not require an empirical rate near 95% to accept the implementation.

## Highest posterior density versus equal tails

A highest-density region can contain several disconnected segments for a
multimodal law. Shared outputs preserve those
segments; the gaps are neither counted as covered nor filled on a chart.
For Bayesian forecasts, this is a highest **posterior predictive** density region.
See [Hyndman's highest-density-region paper](https://robjhyndman.com/papers/hdrgraph.pdf).

For a symmetric unimodal normal distribution, highest-density and central
equal-tail bounds coincide. `LocalLevel` computes its regions from that known
normal law. Shared code still records the methods separately: other adapters
must compute their own highest-density regions, not relabel central quantiles.
Highest-density regions cannot generally be recovered from a few quantiles.
They are defined on the saved original target scale.

## Shared records and capabilities

The immutable records live in [models.py](../src/model_pipelining/models.py):

| Record / field | Meaning |
|---|---|
| `ForecastRequest.quantile_levels` | Exact quantile probabilities required for this issuance |
| `ForecastRequest.highest_density_levels` | Probability masses required for highest-density regions |
| `Forecast.mean` | Predictive mean of the future observation |
| `Forecast.quantiles` | Tuple of `Quantile(probability, value)` records |
| `Forecast.highest_density_regions` | Tuple of `HighestDensityRegion(level, segments)` records |
| `Interval(lower, upper)` | One closed segment of a highest-density region |

Adapters declare boolean `supports_quantiles` and `supports_highest_density`
flags. The runner requests every declared uncertainty kind at caller-selected
`interval_levels`, default `(0.8, 0.95)`. Equal-tail quantiles for those levels
are `(0.025, 0.1, 0.9, 0.975)`. The request still contains only eligible history,
issuance, target time, and requested outputs; no future actual is supplied.

`LocalLevel` declares both kinds and returns `Forecast`. **Its direct return
type has changed from a float: use `result.mean` for the point prediction.**
`predictive_distribution(request)` additionally exposes its `NormalDist` for
direct inspection. Direct requests default to no requested quantiles/regions;
callers constructing requests themselves must set the desired levels. The shared runner does not depend on that Gaussian type.

`TrainingMean` declares neither uncertainty kind and continues returning a float.
Older float-returning adapters without capability flags are treated as point-only.
New adapters may also return `Forecast(mean=...)` with empty uncertainty tuples.
`Forecaster` is the shared protocol; `PointForecaster` remains an import alias.
The pipeline never estimates missing uncertainty for an adapter.

## Run and inspect

After [preparing the preserved input](forecasting.md#prepare-the-input-explicitly):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_level.py \
  --dataset synthetic-mean-shift \
  --tag baseline-input \
  --interval-levels 0.8 0.95 \
  --output-directory runs/local-level-uncertainty-example-001
```

The script automatically requests/saves both supported uncertainty kinds,
evaluates them after freezing forecasts, saves its separate prior simulation,
and draws the widest saved highest-density region. No extra dependency is added.
Use an unused output directory. The baseline script accepts `--interval-levels`
as evaluation configuration, but its point-only model supplies no uncertainty.

Library calls are explicit:

```python
from model_pipelining.backtest import run_backtest
from model_pipelining.local_level import LocalLevel
from model_pipelining.reporting import plot_run

run = run_backtest(
    dataset="synthetic-mean-shift", tag="baseline-input", model=LocalLevel(),
    interval_levels=(0.8, 0.95), output_directory="runs/uncertainty-library-example-001",
)
plot_run(run)  # Defaults to the widest saved highest-density region.
```

`plot_run` optionally accepts `interval_level=0.8` and
`interval_method="highest_density"` or `"equal_tail"`. Its default `"auto"`
prefers highest-density regions, then equal-tail intervals if available. A
point-only run gets no band. Explicitly requesting an unavailable method/level
raises; an existing `forecast.png` is never overwritten. To draw another view,
use a separate copy of the saved run without an existing chart. Using `run` from
the library example above:

```python
import shutil
from pathlib import Path

alternate = Path("runs/uncertainty-library-80-example-001")
shutil.copytree(run, alternate, ignore=shutil.ignore_patterns("forecast.png"))
plot_run(alternate, interval_level=0.8, interval_method="highest_density")
```

This copies saved evidence; it does not rerun a model. Reports read
frozen values and never call the model, re-estimate a density, or use input data.

## Saved evidence and compatibility

New run manifests use `format_version=2`; the existing point CSV columns remain
unchanged. The manifest records capabilities, interval/quantile requests,
`future_observation` semantics, primary display method, and scoring conventions.

| Artifact | Contents |
|---|---|
| `predictions.csv` | Existing frozen point predictions and case/failure details |
| `quantiles.csv` | Frozen target/probability/value records and failures, only when quantiles are declared |
| `highest_density.csv` | Frozen target/level/segment-index/lower/upper records and failures, only when highest-density regions are declared |
| `interval_evaluation.csv` | Equal-tail bounds, actual, coverage, width, interval score, and endpoint pinball losses |
| `highest_density_evaluation.csv` | Actual, region coverage, total segment width, segment count, and failures |
| `scores.json` | Existing point metrics plus a separate `uncertainty` summary and expected/scored/failed case counts |

Point, quantile, and highest-density forecast files are written and closed
**before** actuals are joined for scoring. Failed forecasts retain placeholder
rows for every required probability/region level instead of silently dropping
cases. No actual targets appear in frozen uncertainty files.

CSV target times are UTC ISO strings. Probability/mass fields are numbers in
`(0, 1)`. `segment_index` is zero-based within each target/level; segments are
ordered by their lower bound. A failed region gets one placeholder row for its
required level with blank segment/bound fields. Missing numeric CSV values are
blank; evaluated membership is written as `True`/`False`. JSON uses `null` for
unavailable metrics.

The `uncertainty` summary has one expected/scored/failed count per **forecast
case**, not per quantile, region level, or segment. Its `highest_density_regions`
list holds HPD summaries; `intervals` holds equal-tail summaries; `quantiles`
holds per-probability pinball losses. Partial uncertainty evaluation has status
`failed`, even if some cases were scored. Optional/inapplicable uncertainty
checks have `required=false` and `status="not_applicable"`.

Point-only uncertainty has status `not_applicable`, empty metric lists, and no
uncertainty files. Old version-1 runs still render and compare as saved point
forecasts; they are never backfilled with intervals. To reproduce new runs,
restore the saved model configuration, permanent input reference, minimum
history, and `evaluation.interval_levels` with matching code/environment.

## Validation and evaluation

Declared outputs are required at every origin. Shared validation rejects a
missing/extra/duplicate probability or region level, nonfinite values, crossing
quantiles, reversed/overlapping/unordered segments, and non-nested highest-density
sets across requested masses. Tied quantiles and zero-width segments are allowed.
The mean need not lie within a region, especially for skewed or multimodal laws.

This validates the **structure** of highest-density outputs. Their probability
mass and density optimality remain the adapter's mathematical responsibility.
The Gaussian adapter is checked against known posterior/predictive moments,
normal quantiles, and the equality of its two interval constructions. No generic
highest-density solver for arbitrary densities or samples is claimed.

Evaluation keeps the methods distinct:

- Highest-density regions: coverage (membership in any segment), total set width, and segment count. Gaps contribute neither width nor coverage.
- Equal-tail intervals: coverage, width, and the conventional Winkler interval score.
- Quantiles: unscaled pinball loss, recorded per requested probability.

The interval score is width plus `2 / (1-level)` times an outside-bound error.
Pinball loss is `p*(actual-q)` above the quantile and `(1-p)*(q-actual)` below it;
this omits the factor of two used by some references. The equal-tail score is
**not applied to HPD regions**, because its optimal bounds target equal-tail
quantiles. See [Forecasting: Principles and Practice, Section 5.9](https://otexts.com/fpp3/toolbox.html#evaluating-distributional-forecast-accuracy).
Log scores and CRPS are not inferred from these finite summaries.

Uncertainty scoring accepts a case only after all requested methods/levels
succeed, keeping their scored cohorts aligned. Numerical failure is retained
and fails a required check. Point scoring has its own case-completion counts: a run may have
all point scores but failed uncertainty evaluation. Its overall manifest status
is then `failed`. Read the manifest and required checks, not just point
`scores.complete`. There are no automatic coverage thresholds, model rankings,
or approval decisions; human review remains pending.

`compare_runs` still requires matching permanent inputs, task, training cases,
target dates, actuals, and complete runs. It compares uncertainty only where
**both** saved runs supply the same kind and level. A point-only baseline has
no uncertainty comparison; individual available bands can still be displayed.
