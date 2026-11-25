# Full predictive distribution scoring

Historical backtests now evaluate model-supplied observation laws with CRPS and
negative log density. Both are losses: lower is better. They complement mean
errors, quantile loss, interval coverage and predictive criticism. A successful
computation or a lower development loss does not approve a model or select it.

## What the scores measure

CRPS integrates the squared distance between the predictive CDF and the observed
step function. It evaluates the whole distribution and has the target's units.
Equivalently, for independent predictive draws X and X', it is
E|X-y| minus half E|X-X'|. Production uses deterministic closed forms, without
simulation or reducing the distribution to its mean/variance.

Negative log density is `-log(p(y))` using natural logarithms and the density in
original target units. It penalizes assigning little density to what occurred.
A density may exceed one, so this loss can be negative. Changing units multiplies
CRPS by the scale factor and adds its log to log loss. Compare only identical
observations, transformations and units. No arbitrary density floor caps tail
penalties. See [proper scoring sources](references.md).

Student-t calculations use its actual degrees of freedom, location and scale;
scale is not its standard deviation. The interface requires df > 1 and a finite
first moment, including cases with undefined variance. Gaussian-mixture scoring
uses every representable positive component and log-sum-exp for density. Gaussian
forecasts are encoded as one-component mixtures. Mixture CRPS sums component-pair
absolute-distance expectations in blocks: memory is bounded, while runtime is
quadratic in component count. Quadrature/HPD checks still assess model integration
separately; a finite loss does not prove quadrature accuracy.

## Automatic behavior and model contract

`Forecast` has an optional `distribution` dictionary. Models explicitly declare
`supports_distribution=True` to require it on each successful origin. Supported
records are:

```python
student = {"family": "student_t", "df": 8, "location": 2, "scale": 1.5}
mixture = {"family": "gaussian_mixture", "weights": [0.7, 0.3],
           "means": [-2, 4], "variances": [0.16, 2]}
empirical = {"family": "empirical_samples", "values": [1.0, 2.5, 4.0],
             "weights": [0.25, 0.5, 0.25]}
```

Discrete `empirical_samples` laws compute exact $O(k \log k)$ nonnegative
step-integral CRPS (`int (F(x) - 1{x >= y})^2 dx`) and leave `log_score` as
`None` because discrete atoms have no continuous Lebesgue density. Nonparametric
baselines (`EmpiricalClimatology`, `EmpiricalResidualPersistence`) use this
family with `supports_highest_density=False` so no continuous density or HPD
region is fabricated. Comparisons between continuous and empirical runs still
compute paired CRPS differences, worst-5% concentration, and Bartlett HAC SEs
while marking log-score comparison not applicable.

Current Bayesian adapters (`LocalLevel`, `NormalMean`, `NormalMeanVariance`,
`LocalLevelVariance`, `LocalLevelJointVariance`) serialize the law from the same
fit that supplied their mean/quantiles/regions. The runner validates parameters,
mean agreement, quantile CDF values and region mass. Region mass agreement does
not independently establish highest-density optimality. Point-only adapters
require no distribution; legacy missing capability flags mean unsupported.
No law is inferred from a few saved interval endpoints.

All origin laws and failure placeholders are written to
`predictive_distributions.json` before actual selection or scoring. The runner
reads those saved laws for `distribution_evaluation.csv`, whose origin/history,
actual release vintage, statuses and individual losses are retained.
`scores.json.distribution` saves method/version, counts and mean losses.
Both full-law validation and complete distribution scoring are required checks
for declaring such a run completed. A scoring failure leaves the frozen forecast
and other scores intact, fails the run, and preserves later cases.

Nonfinite inputs, unsupported families, missing laws, undefined first moments,
unrepresentable weights/losses and unresolved CRPS subtraction fail explicitly.
There is no Gaussian substitution, weight pruning, density floor or invented
finite value. A relative roundoff screen (128 ulps of the subtracted terms versus
1e-7 times the result) rejects severe cancellation; it is a numerical screen,
not a proof of all floating-point errors or model approximation precision.

## Saved-only comparison and review

Call the existing `compare_runs(first, second, output_directory=...)` explicitly.
It still requires complete matching permanent inputs, experiment/evaluation role,
histories, cases and scoring actual vintages. If both runs declare full laws, it
also checks saved distribution/score case alignment, required checks, finite
losses, method/units and aggregates recomputed from per-case scores. It does not
refit or authenticate historical files.

`comparison.json.point_comparison` summarizes each run's saved errors (signed
mean/median bias, MdAE, MAE, p95/max absolute error, RMSE, and worst-5% absolute
and squared error concentration) alongside paired absolute-error and squared-error
differences with win/loss/tie rates, worst-5% absolute difference share,
deterministic naive plus Bartlett HAC standard errors (`robust_scoring.py`),
and an Ehm et al. (2016) mean Murphy dominance check (`mean_murphy_curve`).
`compare_runs` also writes a human-readable `comparison.md` alongside
`comparison.json` and the charts. See
[the evaluation and decision tools guide](evaluation-and-decisions.md) for how
to read each metric.

`comparison.json.distribution_comparison` retains both mean losses and per-case
first-minus-second differences, and adds per-run CRPS/log-score tail summaries
plus paired CRPS and log-score stability diagnostics (median difference, win
rates, worst-5% absolute difference share, and Bartlett HAC SEs). Negative
differences favor the first forecast on that loss. `distribution_scores.png`
plots per-case CRPS, negative log density, and cumulative CRPS difference over
the same dates so regime shifts and single-day outliers stand out visually.
These remain descriptive paired results; no significance threshold or automatic
winner is introduced. If either run lacks full laws, distribution comparison is
explicitly not applicable while point and common uncertainty comparisons remain
available. Older saved runs are not backfilled or rewritten. Failed/partial
declared distribution scores cannot silently enter a full-cohort comparison.

[Experiment review](experiment-review.md) also checks the saved distribution
cohort and displays mean losses when present. The library scoring API is
`score_distribution(record, actual) -> {"crps": float, "log_score": float}` in
`model_pipelining.distribution_scoring`; it requires neither a model nor source
snapshot. Existing sequential-check records use the same family parameter keys
and can be scored explicitly, but sequential checks do not start new comparisons.

## Verification

Twenty-one new tests check independent adaptive CDF integrals, direct density
references, the normal limit, separated mixture modes, transformations, extreme
tails without density underflow, precision failures and all five adapters. Workflow
tests cover pre-actual freezing, own-target/final isolation, explicit release
vintages, exact replay, missing/failed scores, point/legacy compatibility,
saved-only comparison/review, consistent cohorts and unchanged source files.

A local end-to-end exercise under `runs/distribution-scoring-validation-89wgplk4/`
completed 130 matched development cases each for the constant-mean Student-t,
fixed-evolution learned-noise model and categorical joint-variance model. Their
mean CRPS values were 1.64837, 0.66828 and 0.66253; mean negative log densities were
2.73546, 1.64919 and 1.67372 respectively. Both saved comparisons produced paired
loss charts and the joint forecast assembled into a review report. These are
known mean-shift development data: the two losses emphasize different aspects
of prediction, and these observations neither select settings automatically nor
establish performance on an untouched application. Generated artifacts are local
and ignored by Git.
