# Comparing prior influence on learning and prediction

[Joint variance evidence](joint-evidence.md) also supports categorical evolution-variance priors and preserves W/V/state dependence. The sections below retain their stated original-model scope.

Model evidence providers now own prior/parameter/predictive summaries and validate
which assumptions may vary. Shared code retains history selection, differences,
persistence and saved-only reporting. An [experiment evidence plan](evidence.md)
can require and explicitly execute comparisons alongside predictive checks;
standalone calls remain available and forecasting does not start them automatically.

The workflow also supports `--model local-level-variance`. Initial-state and
observation-variance priors may change, with evolution variance and numerical
settings held fixed. State/variance intervals are equal-tail; predictive regions
are numerically resolved mixture HPD sets. The rest of this page details the
constant-mean mode; see [the dynamic guide](local-level-variance.md) for fields,
precision checks and commands. No scalar prior coefficient or conjugate scale
decomposition is claimed for the dynamic model.

This workflow compares caller-chosen normal/inverse-gamma priors for
`NormalMeanVariance`, the model with an unknown **constant** mean and unknown
**constant** observation variance. Every prior is evaluated on identical
histories, from before any observations through an explicit development cutoff.

The purpose is to see how assumptions affect learning and prediction. Strong
or persistent influence is evidence to understand, never a failure condition
or a quantity to minimize. There is no tuning, ranking, preferred prior, score
optimization, or automatic model approval.

## What is compared

One reference prior and at least one named alternative supply:

| Setting | Interpretation |
|---|---|
| `prior_mean` | Conditional prior location for the mean, in target units |
| `prior_mean_precision` | `k0`; conditional prior mean variance is `v/k0` |
| `prior_shape` | `a0`; inverse-gamma shape controlling variance-prior form |
| `prior_scale` | `b0`; inverse-gamma scale, in squared target units |

The [model guide](normal-mean-variance.md) defines the exact update. The path
records posterior parameters, the mean-parameter location and HPD interval,
observation-variance median/mean and equal-tail credible interval, Student-t
predictive mean/variance/scale/HPD bounds/width, and explicit moment statuses.

The coefficient `k0/(k0+n)` describes how prior location affects the learned
location with other prior settings held fixed. It is **not total prior
influence**. Changing mean location or precision can also change learned
variance through the prior/data disagreement term:

```text
b = b0 + sum((y[i]-average(y))^2)/2
       + (k0*n/(k0+n))*(average(y)-prior_mean)^2/2
```

All three scale components are saved. Shape/scale changes alter variance
uncertainty and Student-t tails; they can also alter marginal uncertainty about
the mean because this is a joint prior. Groups of changed settings identify
which inputs were changed, not an attribution of all resulting effects.
See [Murphy's conjugate Gaussian analysis](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf),
Sections 5–6, for the mathematical reference.

## Intervals, moments and differences

The caller selects one `interval_level` (default 0.95). Mean-parameter and
observation-predictive Student-t intervals are HPD because these laws are
symmetric and unimodal. The skewed inverse-gamma variance parameter uses an
**equal-tail credible interval**, not an HPD interval or prediction interval.
The plots and saved scope retain that distinction.

Proper heavy-tailed priors can have finite quantiles without finite means or
variances. The path uses locations and the variance median so these cases can
still be compared. Undefined moments are empty CSV values/JSON nulls with
explicit statuses; no finite substitute is invented. For example, before any
observations, `prior_shape<=0.5` has no marginal mean, and `prior_shape<=1` has
no expected observation variance. Numerically unrepresentable existing moments
or requested bounds are computational failures, distinct from undefined moments.

Each update saves signed differences from the reference for mean location,
variance median/mean, predictive variance, predictive bounds and predictive
width. A difference is null when a needed moment or reference calculation is
unavailable. `comparison_status` distinguishes unavailable configurations,
unavailable references, and numerical difference failures. Summary maxima
use only defined matched differences, including the empty-history prior.
They are descriptions, not acceptance thresholds.

## Run explicitly

Prepare a snapshot/tag using [the walkthrough](forecasting.md#prepare-the-input-explicitly),
then use a new output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_priors.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59Z \
  --experiment configs/daily_forecast_experiment.json \
  --config configs/normal_mean_variance_priors.json \
  --interval-level 0.95 \
  --output-directory runs/tutorial-prior-influence-001
```

[The example configuration](../configs/normal_mean_variance_priors.json)
contains `reference` settings and named `scenarios` with overrides. Every
scenario inherits unchanged reference settings. Expanded configurations are
saved for replay. Illustrative alternatives change location, mean precision,
variance concentration, expected variance, or tail shape. The concentrated
variance prior uses shape 30 and scale 29, preserving expected variance 1 while
changing its uncertainty. These are examples, not a search grid or recommendations
for a particular application. Scenario names are nonblank and cannot be
`reference`; concrete model instances with `mcmc=None` are required.

The CLI automatically calculates/saves comparisons and draws two charts. It
does not acquire/register data, move tags, select priors, run MCMC, compute
held-out scores, invoke predictive-check simulations, or approve a model.
Library reporting is a separate explicit call. Backtests and predictive checks
do not invoke prior comparisons automatically.

## Histories and information timing

The input snapshot is verified and pinned. Optional `ExperimentSpec` settings
provide columns, calendar/revision rules, daily issuance delay, minimum history
and a final boundary. Only development role is allowed, and the explicit aware
as-of time must precede any declared `final_start`. Default input rules use
immediate/unrevised `timestamp`/`value` and minimum history 30.

Update zero is the prior with empty history. At each period's timestamp plus
the task's issuance delay, the workflow selects the latest release available
by that issuance for periods no later than the period being processed. It
starts with the first nonempty available history; earlier empty histories are
recorded as waiting in the schedule. Minimum history applies at the final
cutoff, allowing the early path to reveal prior influence from the first datum.

The explicit final as-of time is appended if it differs from the last daily
issuance. This can include delayed releases received later in the day. Repeated
histories may appear when no new information arrives; revisions can change
values without increasing history count. Paths refit from the eligible history
at each update. Later revisions never rewrite earlier updates; these are not
prefixes of a retrospectively revised end-of-period file.

All configurations receive the same histories, coordinates and history hashes.
Future/final values are excluded from every model request and comparison.
The whole file is still parsed/validated, and selection cannot establish that
humans have never viewed final data. Forecast horizon only provides future
request coordinates; actual vintage and benchmarks do not score the comparison.
The shared release selector also serves backtests and predictive checks.

## Saved evidence and charts

Prior-influence format 1 is separate from predictive-check format 1, fit format
1 and backtest format 3. Files remain local and are ignored by Git.

| File | Contents |
|---|---|
| `prior_influence.json` | Kind/version/status, full configurations, interval level, pinned input/development context, semantics, environment/source fingerprints and update counts |
| `learning_schedule.csv` | Daily issuance/period boundaries, available counts, waiting entries and explicit final cutoff |
| `selected_histories.csv` | Selected period/release/value records for each nonempty update, common to all configurations |
| `learning_paths.csv` | Prior and every update for every configuration: coordinates/history identity, calculation status/errors, posterior/predictive quantities, scale components and reference differences |
| `configuration_summaries.json` | Expanded settings, changed setting groups, completion/failure counts, final row and maxima over available signed differences |
| `checks.json` | History-selection and required calculation completion checks |
| `review.json` | Pending interpretation and limitations; no automatic approval |
| `prior_learning.png` | Location/mean HPD, variance median/equal-tail bounds, predictive HPD width, conditional prior-location coefficient, predictive HPD and degrees of freedom |
| `prior_differences.png` | Signed mean-location, variance-median and predictive-width differences from reference |

Plots use update index on the horizontal axis, with zero denoting the prior.
Dates and available observation counts are retained in the CSV. Variance,
predictive width, and prior coefficient use logarithmic vertical axes on the
learning chart; differences use symmetric-log axes with a linear region within
0.01 of zero, retaining signed original units. This display setting is not an
acceptance threshold. Shaded mean
credible regions and observation prediction regions answer different questions.

Only saved files are read when plotting. No model calculation, input retrieval,
random draw or refit occurs. Both output paths must be unused; existing files
and symlinks are rejected before plotting. Source evidence is not rewritten.

Numerical failures are retained per update. Later updates and other priors
continue. A failed reference leaves valid alternative quantities visible with
unavailable differences. Required calculation failures make overall status
`failed`; influence magnitude does not. Undefined moments are not failures.
Plots show gaps and name failed configurations. Invalid settings/input raise
before output creation. I/O failures propagate and can leave a `running`
directory; optional plot failure does not revise saved computation status.

## Library and replay

```python
from dataclasses import replace
from datetime import datetime
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.prior_influence import run_prior_influence, plot_prior_influence

reference = NormalMeanVariance()
comparison = run_prior_influence(
    dataset="synthetic-mean-shift", tag="baseline-input", reference=reference,
    scenarios={
        "stronger-mean": replace(reference, prior_mean_precision=30),
        "concentrated-variance": replace(reference, prior_shape=30, prior_scale=29),
    },
    issued_at=datetime.fromisoformat("2020-06-08T23:59:59+00:00"),
    output_directory="runs/library-prior-influence-001",
)
plot_prior_influence(comparison)
```

This library example uses default input rules. Pass `experiment=` to enforce a
declared final boundary or release schema. Replay restores the permanent input,
expanded configurations, saved experiment, interval level and cutoff into a
new directory with matching source/environment. No seeds or MCMC are needed.
Numerical evidence repeats; execution timestamps differ. Retain the input
store, whole comparison directory and matching code/dependencies.

Prior predictive plausibility, posterior predictive criticism, influence and
held-out forecast evaluation answer different questions. Use [predictive checks](predictive-checks.md)
to inspect the datasets implied by each prior/model, and [forecast evaluation](experiments.md)
when the question is held-out performance. No cross-workflow automatic selection
or release policy is introduced here.
