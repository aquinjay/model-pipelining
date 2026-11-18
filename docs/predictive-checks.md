# Prior and posterior predictive checks

This workflow checks the datasets implied by `NormalMeanVariance`, our model
with an unknown **constant** mean and unknown **constant** observation variance.
It is separate from historical forecast scoring and from the MCMC audit.
[Prior influence comparisons](prior-influence.md) now provide a separate exact
workflow for comparing explicit prior settings on common historical histories.

Prior checks ask what observations our initial beliefs imply. Posterior checks
ask which features of the selected observed history the fitted model can
reproduce. For each replication, draw **one shared mean and variance**, then
simulate a whole dataset of the selected length. Generating a new parameter
pair or independent marginal Student-t value at every position would describe
a different joint model.

## Generation and interpretation

The model uses the [normal/inverse-gamma prior and exact posterior](normal-mean-variance.md).
For prior replications, use the caller's prior parameters. For posterior
replications, first update them using only the selected history. In either case:

```text
G ~ Gamma(shape=a, scale=1)
v = b/G
mu | v ~ Normal(m, v/k)
y_rep[i] | mu,v ~ independent Normal(mu, v), for all i in one dataset
```

All replications are independent joint draws, with no MCMC, warmup, convergence
threshold, or fit per replicated observation. The exact prior/posterior law is
available, so Markov chains would be unnecessary for these checks. A configured
`mcmc` setting is rejected explicitly; pass a `NormalMeanVariance` with
`mcmc=None`. Forecast APIs and their optional sampler audit are unchanged.

This follows [prior/posterior predictive checking](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html):
simulate datasets and compare relevant features. Posterior replications reuse
the fitted history, so discrepancies are **in-sample criticism**. They are not
held-out forecast scores, calibrated hypothesis-test p-values, or automatic
model approval. Plausibility depends on target units and application context.

## Run a prior check before observing data

No input file, snapshot, or tag is needed:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_normal_mean_variance.py \
  --prior-only --observation-count 160 --replications 1000 --seed 0 \
  --prior-mean 0 --prior-mean-precision 0.01 --prior-shape 3 --prior-scale 2 \
  --output-directory runs/tutorial-prior-checks-001
```

This saves parameter draws, complete replicated datasets, their statistics,
computation checks, and a pending human review. The CLI automatically draws
`prior_predictive.png`. Choose the dataset length and priors explicitly;
the command does not choose or tune a prior from data. No observed-statistic
comparisons are invented in prior-only mode.

## Check a selected development history

Prepare the snapshot/tag as in [the walkthrough](forecasting.md#prepare-the-input-explicitly).
For the 200-day synthetic series, this selects the first 160 observations and
respects the example experiment's final boundary:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_normal_mean_variance.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59Z \
  --experiment configs/daily_forecast_experiment.json \
  --replications 1000 --seed 0 \
  --output-directory runs/tutorial-predictive-checks-001
```

The sample experiment reserves targets starting 2020-06-09. `--as-of` is an
explicit timezone-aware selection time. The workflow validates the input using
the experiment's columns, calendar, and revision rules, then selects the latest
release for each period whose timestamp and availability are both no later
than that time. Minimum available history is enforced. Both prior and posterior
replications use the selected history's length.

Only development-role specifications are allowed, and `--as-of` must precede
any declared `final_start`. Final observations and later revisions do not enter
the fitted history, discrepancies or random seeds. The complete input file is
still parsed/validated. This is a selection guarantee, not proof that humans
have never viewed final data. Without an experiment, default immediate/unrevised
`timestamp`/`value` data and minimum history 30 apply; no final boundary is
invented. For a different development subset, supply its explicit as-of time.

Forecast horizon, issuance delay, scoring vintage and benchmarks do not schedule
these replications or produce forecast scores. The as-of time controls history
selection directly. The complete experiment is retained for provenance; its
horizon only supplies an unused future coordinate to the posterior request.

The CLI automatically saves both phases and draws `prior_predictive.png` and
`posterior_predictive.png`. Simulation and plotting are separate library calls.
The backtest and its run scripts do not invoke these checks automatically.
Acquisition, registration, tags, model settings, selection time, experiment
boundaries, invocation, and human interpretation remain caller actions.

## Statistics and finite-simulation precision

Every completed replicated dataset and the selected observed history receive
the same fixed statistics:

| Feature | Statistics |
|---|---|
| Location and spread | Mean and sample SD (`n-1` denominator) |
| Distribution and extremes | Minimum, maximum, 5th percentile, median, 95th percentile |
| Tail shape relative to scale | Maximum absolute distance from the sample mean, divided by sample SD |
| Serial dependence | Pearson correlation of successive values |
| Change in level | Second-half mean minus first-half mean; an odd middle observation joins the second half |
| Accumulated departure | Maximum absolute cumulative sum of centered values, divided by `sample_sd * sqrt(n)` |

Quantiles use linear interpolation. Undefined quantities are null with explicit
reasons: for example, a single observation has no sample SD, and constant lagged
sequences have no lag correlation. This is distinct from a failed numerical
calculation. Both are visible in saved evidence.

For each statistic, summaries retain the observed value, valid/missing
replication counts, replicated 2.5%/50%/97.5% quantiles, and separate inclusive
fractions `replicated >= observed` and `replicated <= observed`. Ties are counted
and the inclusive fractions need not sum to one. Missing values never become
zero and do not enter denominators. Missing counts include failed replications
and undefined or failed statistics; inspect the reasons and phase failures.

Fractions include the independent-replication Bernoulli MCSE and Wilson 95%
interval. A zero count has estimated fraction/plug-in MCSE zero but a positive
upper interval bound: it is not proof of zero probability. These intervals
describe simulation uncertainty for the fraction conditional on this fitted
history, not posterior uncertainty about a parameter. No discrepancy threshold
changes execution status or selects a model.

## Saved artifacts, plots and failure behavior

Predictive-check format 1 is separate from backtest/run format 3 and fit format
1. Use a new output directory; existing directories/charts are rejected.

| File | Contents |
|---|---|
| `predictive_checks.json` | Kind/version, computation status, full model settings, seed derivation, replication count/length, environment and source fingerprints; observed mode adds permanent input, experiment, as-of/history hash and selected release records |
| `observations.csv` | Selected history with period/availability/value; header only in prior-only mode |
| `observed_statistics.json` | Observed values/undefined reasons, or explicit prior-only inapplicability |
| `prior/parameter_draws.csv`, `posterior/parameter_draws.csv` | One shared mean/variance per replication; failed indices/errors retained |
| `prior/replicated.csv`, `posterior/replicated.csv` | Every generated observation with replication/index coordinates |
| `prior/statistics.csv`, `posterior/statistics.csv` | Dataset-level statistics and undefined reasons |
| `prior/summary.json`, `posterior/summary.json` | Phase completion/errors, generating joint distribution, missing counts, replicated quantiles and observed comparisons |
| `checks.json` | Required computation checks; observed-statistic check is inapplicable for prior-only mode |
| `review.json` | Pending human interpretation, questions and limitations |
| `prior_predictive.png`, `posterior_predictive.png` | Explicit library plot call or automatic CLI reporting |

Prior-only mode has no posterior folder or chart. Charts show the first up to
20 replicated paths and their empirical CDFs, with observed data where supplied,
plus discrepancy histograms. They are representative simulations, not
simultaneous prediction bands. Plotting reads only saved files and never fits,
retrieves inputs, or generates new draws.

Numerical simulation failures retain their replication index/error and later
independent replications continue. A failed posterior calculation retains prior
evidence. Statistic failures retain the generated dataset and their error.
Any required computational failure makes overall status `failed`; a large
model discrepancy does not. Partial failed results can be plotted. Persistence
errors propagate and can leave a `running` directory. An optional chart failure
does not rewrite computation status. No crash recovery is supplied.

## Library and replay

```python
from datetime import datetime
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.predictive_checks import (
    run_prior_predictive_checks, run_predictive_checks, plot_predictive_checks,
)

model = NormalMeanVariance()
prior = run_prior_predictive_checks(
    model=model, observation_count=160, replications=1000, seed=0,
    output_directory="runs/library-prior-checks-001",
)
plot_predictive_checks(prior)

checks = run_predictive_checks(
    dataset="synthetic-mean-shift", tag="baseline-input", model=model,
    issued_at=datetime.fromisoformat("2020-06-08T23:59:59+00:00"),
    output_directory="runs/library-predictive-checks-001",
)
plot_predictive_checks(checks)
```

The second library example uses default input rules; pass `experiment=` to
enforce a declared final boundary or release schema. Direct simulation helpers
`simulate_prior_predictive(model, observation_count=..., replications=..., seed=...)`
and `simulate_posterior_predictive(model, request, replications=..., seed=...)`
return `PredictiveReplications` without persistence. `dataset_statistics(values)`
returns named values with unavailable reasons.

Each replication has a separate PCG64 generator initialized by
`SeedSequence([seed, phase_number, replication_index])`, using 0 for prior and
1 for posterior. Failures do not shift later streams; increasing replication
count preserves the prefix. Prior draws depend only on model settings, seed and
chosen length; changing observed values does not change them. Python/NumPy
global RNG states are untouched. These stream rules and library versions are
saved for replay; there are no interacting chains or MCMC diagnostics.

Replay explicitly restores the saved model settings, seed/count, and (for
observed mode) permanent input, `ExperimentSpec.from_dict(...)` and as-of time,
then invokes the workflow in a new directory with matching source/environment.
Numerical files repeat; execution timestamps differ. Preserve the input store,
whole check folder and matching code/dependencies. Generated data, checks and
PNGs stay local and are ignored by Git.

[Prior influence comparisons](prior-influence.md) are now implemented separately.
Other models' full-data predictive checks,
formal simulation-based calibration, and application-specific review policies
remain future work. These checks do not automatically approve a prior or model.
