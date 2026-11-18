# Learning observation and evolution variance together

`LocalLevelJointVariance` learns observation variance V and an unknown constant
state-evolution variance W. W has an explicit categorical prior on caller-chosen
ordered support points, including zero when desired. V has an independent
inverse-gamma prior. The initial state has an independent Gaussian prior.

This is a discrete model for W. It does not approximate an unspecified continuous
prior or tune its support from scores. The default support is `(0, .01, .1, 1)`
with equal prior probabilities. The support, prior probabilities, initial-state
settings and inverse-gamma settings are all saved and caller-controlled.

Conditional on W, the existing [scalar learned-variance model](local-level-variance.md)
integrates V with checked log-variance quadrature and integrates states by Kalman
filtering. For each support value w_j the full marginal likelihood Z_j gives:

```text
posterior_probability(W=w_j | y) ∝ prior_probability(W=w_j) * Z_j
p(V, state | y) = sum_j p(W=w_j | y) * p(V, state | y, W=w_j)
```

Every component must satisfy its required precision checks. Numerical marginal
likelihoods and tail estimates remain approximations with saved refinement
records. All support points are enumerated; none are silently selected as the
single fitted setting. W and V can be dependent after observing data, despite
independent priors. Saved covariance and support probabilities make that
confounding inspectable. A single support point reproduces the scalar model.

The observation forecast is a Gaussian mixture over both W and V. Means,
quantiles and HPD sets integrate both sources of uncertainty, including between-
component variation. HPD sets retain gaps and require the shared numerical
resolution checks. A general forecast is not Student-t. State increments use
actual elapsed UTC days; the first transition is one day.

## Controlled investigation

`dynamics_study.run_dynamics_study` creates stable, gradual and abrupt synthetic
processes with identical Gaussian noise, preserves their input files, then
compares explicit fixed-W settings using matched sequential development cases.
The default 120 observations split into 100 development and 20 excluded final
observations, with minimum history 20: 80 scored cases per setting/series.

```bash
PYTHONPATH=src .venv/bin/python scripts/study_dynamics.py \
  --evolution-variances 0 .01 .1 1 \
  --output-directory runs/dynamics-study-001
```

The command automatically preserves its generated inputs, runs sequential checks,
saves `study.json`, `summary.json`, pending review and child evidence, and draws
`dynamics.png`. Library plotting is explicit. No application model is selected.

The initial exercise is local in `runs/dynamics-study-initial/`. Development
mean squared errors across W=0/.01/.1/1 were:

| Fixture | 0 | .01 | .1 | 1 |
|---|---:|---:|---:|---:|
| Stable | .578 | .583 | .645 | .814 |
| Gradual | 2.741 | .717 | .659 | .817 |
| Abrupt | 7.641 | 2.465 | 1.180 | 1.146 |

These controlled results motivate retaining uncertainty over dynamics. They do
not establish that learning one global W explains abrupt transitions. Posterior
mass near the largest support point motivates examining the prior/support or
transition assumption; it is not an automatic grid-expansion rule.

## Run the joint model

Prepare an input snapshot and tag as in [the walkthrough](forecasting.md), then:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_joint_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --experiment configs/daily_forecast_experiment.json \
  --config configs/local_level_joint_priors.json \
  --interval-levels .8 .95 \
  --output-directory runs/joint-local-level-001
```

The caller chooses input/model/experiment and a new directory. The CLI runs the
shared backtest and draws `forecast.png` and `joint_variances.png` automatically.
The latter reads saved per-origin W probabilities, expected V/W and their
posterior covariance; it never fits or retrieves input. Library callers invoke
`reporting.plot_run` and `plot_joint_variances` explicitly.

`LocalLevelJointVariance.from_configuration` restores JSON lists and numerical
settings. `prior()` and `posterior(request)` return `JointVariancePosterior` with
component posteriors, W probabilities, V CDF/quantiles, state moments, predictive
mixtures and variance covariance. `evolution_cdf(value, inclusive=...)` preserves
atoms; `evolution_ppf` returns an actual support point. No interpolation creates
an unsupported W value. `fit` saves full component evidence and labels every
quadrature node with its W. Failed precision remains required evidence.

The joint model currently supplies the `sequential_checks` provider capability.
Generative predictive checks, prior comparisons and calibration must be extended
explicitly before they can be required for this new model; unsupported evidence
fails visibly. Existing models retain their established capabilities.

## Verification and limits

Thirteen new tests include independent adaptive integration of a batch Gaussian
likelihood for W probabilities, predictive CDF and V/W covariance; the single-W
limit; observation/state degeneracy; V CDF/quantile inversion; precision failure
persistence; replay; own-target isolation; matched/shared-noise studies; saved-
only plotting and CLI behavior. The suite now contains 323 tests. The independent
reference exposed a near-zero symmetric-mixture median bracket affected by CDF
roundoff; verified endpoint probabilities now handle that case without loosening
bulk or tail root tolerances.

Support choice and categorical prior influence remain modeling assumptions.
One learned global W is not time-varying volatility or a change-point model.
Numerical correctness and development scores do not approve an application.

The joint exercise completed 80 forecasts and 80 sequential checks per fixture
under `runs/joint-dynamics-validation-_3bcrkga/`, including predictive HPD regions
and saved variance-learning charts. Stable-series final W probabilities were
about .775/.224/.000764/negligible; gradual-series probabilities were concentrated
on .01/.1 at about .102/.898. These are development findings, not approval.
All 323 tests passed.
