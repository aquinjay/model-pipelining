# The local-level Bayesian model

This page explains the implemented model and its assumptions. Follow
[the forecasting walkthrough](forecasting.md) to run it; use
[the API reference](../src/model_pipelining/README.md) for signatures and
[the uncertainty guide](uncertainty.md) for the shared output contract.

## What the model assumes

An observed value is an underlying **level** plus noise. The level is hidden:
we observe a value such as a daily measurement, rather than the level itself.
The level can move between days. Observation noise can also move a measurement
away from that level.

The model starts with a probability distribution for the initial level. It then
updates that distribution as each past observation arrives. This updated belief
is the **posterior**. Tomorrow's forecast combines uncertainty about today's
level, another day's level movement, and noise in tomorrow's measurement.

“Bayesian” here means inference about that hidden level, conditional on the
chosen noise variances and initial prior. The code does not estimate the noise
variances or place a posterior distribution over them. It uses exact Gaussian
(normal-distribution) updates, with no sampling algorithm.

This model was selected as the first Bayesian adapter because its small,
checkable calculation can exercise the shared infrastructure. Suitability for a
real application still needs that application's question, timing, assumptions,
and evidence. Better scores or a faster response to a change do not verify the
Bayesian calculation.

## Settings and timing

`LocalLevel` accepts keyword arguments only:

| Setting | Default | Interpretation |
|---|---|---|
| `initial_mean` | `0.0` | Prior mean level before the first daily transition |
| `initial_variance` | `100.0` | Prior uncertainty about that level; standard deviation 10 in input units |
| `observation_variance` | `1.0` | Noise variance around each daily level; standard deviation 1 in input units |
| `evolution_variance` | `0.1` | Variance of the level's movement per day |

Variances are in squared input units. They are not standard deviations.
Observation variance must be positive; initial/evolution variance may be zero.
All settings must be finite numbers, excluding booleans/strings. These defaults
are illustrative assumptions, fixed throughout a run. They are not tuned or
read from the synthetic generator's hidden settings.

Each forecast refilters the supplied past history from this same initial prior.
No training state is retained between requests. Only forward updates are used;
there is no smoothing with later observations. The adapter requires daily UTC
spacing and a target one day after the final observation.

The initial prior is **before the first observation's daily transition**. Thus
the first observation's prior variance is
`initial_variance + evolution_variance + observation_variance`. Filtering and
prior simulation use the same timing convention.

## Calculations

Let `m` and `C` be the preceding posterior level mean and variance, `W` the daily
evolution variance, `V` the observation variance, and `y` the next observed value.
All Gaussian distributions below use mean/variance notation:

```text
initial level ~ Normal(initial_mean, initial_variance)
next level   = previous level + Normal(0, W)
observation  = current level  + Normal(0, V)
```

Noise terms are independent. For one forward update:

```text
R = C + W                 predicted level variance
Q = R + V                 predicted observation variance
K = R / Q                 weight placed on the new observation
m_new = (1-K)*m + K*y      posterior level mean
C_new = V*K               posterior level variance
```

Tomorrow's **observation** distribution is normal with mean `m_new` and variance
`C_new + W + V`. Using only `C_new` would instead describe uncertainty about the
updated hidden level. The adapter uses the larger observation variance for its
shared forecast quantiles and highest posterior predictive density regions.
For this symmetric unimodal normal law, HPD and equal-tail bounds coincide.
The shared pipeline keeps those methods separate for other distributions.

### A small worked check

Use `initial_mean=0`, `initial_variance=1`, and `V=W=1`, with observations 2 and 4.
These values are for checking the arithmetic, not the default configuration:

| Stage | Level mean | Level variance | Observation predictive variance before seeing that value |
|---|---:|---:|---:|
| After observing 2 | `4/3` | `2/3` | `3` |
| After observing 4 | `3` | `5/8` | `8/3` |
| Forecast the next observation | `3` | — | `5/8 + 1 + 1 = 21/8` |

[The numerical tests](../tests/test_local_level.py) check the filtering result
against both this hand calculation and independent batch Gaussian conditioning.
[The uncertainty tests](../tests/test_uncertainty.py) check the next-observation
variance, quantiles, HPD probability mass, and boundary densities.

The model specification is also described in the
[statsmodels documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html).
The scalar filtering equations are the Gaussian-random-walk example in Section
4.3 of [Särkkä's Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf#page=79).
The explicit initial-prior timing is a project convention.

## Run it

After [preparing an input](forecasting.md#prepare-the-input-explicitly), run from
the repository root, using an unused output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --initial-mean 0 --initial-variance 100 \
  --observation-variance 1 --evolution-variance 0.1 \
  --interval-levels 0.8 0.95 \
  --output-directory runs/model-guide-local-level-001
```

Use `--snapshot-id ID` instead of `--tag` for a permanent selection. `--store`
chooses another snapshot store; `--minimum-history` changes the initial training
count. The script automatically performs these actions:

1. Validate the four model settings and simulate one 200-step prior trajectory,
   using `--prior-seed` (default 0), before fitting.
2. Call the shared backtest, which validates the input, requests the declared
   mean/quantile/HPD outputs, freezes them, evaluates them, and saves checks.
3. Save `prior_predictive.csv` and `prior_predictive.json` with the simulated
   levels/values, seed, count, Python version, settings, and pending prior review.
4. Draw `forecast.png` from saved results, preferring the widest HPD region, and
   print scores. A failed backtest produces exit status 1.

The simulated trajectory is generated before fitting but saved after the runner
creates the output directory. Invalid input leaves no run/prior files. Extra
script/reporting I/O failures propagate; the manifest describes execution of the
backtest, not completion of every optional script artifact.

A prior simulation shows consequences of assumptions **before conditioning on
data**. It is separate from held-out forecasts and is not a passed plausibility
check. Reviewing whether the prior is sensible remains a human action.

## Library interface and diagnostics

Construct `LocalLevel(...)` and pass it as `model=` to `run_backtest`, as in
[the full walkthrough](forecasting.md#run-the-bayesian-model). Its settings are
saved automatically in the run manifest. Library callers invoke reporting and
prior simulation separately; the script above orchestrates those extra calls.

| Method | Returns / meaning |
|---|---|
| `forecast(request)` | `Forecast` with `.mean`, requested `.quantiles`, and requested `.highest_density_regions` for the future observation |
| `predictive_distribution(request)` | The same conditional observation law as a standard-library `NormalDist` |
| `filter_history(history)` | Immutable `FilterStep` records: `timestamp`, `observation_mean`, `observation_variance`, `innovation`, `level_mean`, `level_variance` |
| `simulate_prior(*, count, seed=0)` | Immutable `PriorDraw(level, value)` records, using a local random generator |

An **innovation** is the observed value minus its pre-observation predicted mean.
The filter trace is a training diagnostic; it is not a replacement for the
runner's frozen historical forecasts and is not saved automatically. The
adapter's current `forecast` return type is a record, not a float: use `.mean`.

Reproduction uses `LocalLevel(**manifest["model"]["configuration"])`, the saved
permanent input, minimum history, and `evaluation.interval_levels`, with matching
code/environment and a new run directory. No automatic model restoration exists.

## Compare saved runs

The walkthrough's [comparison step](forecasting.md#compare-saved-results) requires
matching inputs/cases and completed runs. Reporting reads saved artifacts only.
It does not refit, tune, or select a model. This is secondary evidence about model
behavior, separate from checking the implementation's calculations.

The illustrative defaults were exercised on both 200-day sample series, using
170 forecast dates (days 31–200):

| Series | Model | Mean absolute error (MAE) | Root mean squared error (RMSE) |
|---|---|---:|---:|
| Mean shift | Training mean | 2.4197 | 2.9462 |
| Mean shift | Local level | 0.9018 | 1.1744 |
| Constant mean | Training mean | 0.7683 | 0.9638 |
| Constant mean | Local level | 0.8412 | 1.0413 |

These are development results for two known scenarios and fixed settings. They
illustrate that neither model is universally preferable. A later setting change
motivated by these results is a development revision, not an untouched test.

## Remaining work

Tests verify arithmetic, isolation, replay, and some seeded simulation behavior.
They do not establish prior plausibility or suitability for a real application.
Prior sensitivity, fitted-model criticism, variance-parameter uncertainty, and
automatic specialized diagnostic requirements remain open. There is no sampling algorithm requiring sampler diagnostics in this analytic filter.
