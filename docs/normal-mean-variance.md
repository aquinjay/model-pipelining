# Learning a constant mean and observation variance

`NormalMeanVariance` learns both parameters of an exchangeable Gaussian model.
It integrates them exactly into a Student-t forecast of the next observation.
An optional independent-chain MCMC audit checks the joint posterior against
that reference. Forecasts use exact integration in both modes.

The existing [NormalMean](parameter-uncertainty.md) still uses known observation
variance. [LocalLevel](local-level.md) still uses fixed observation/evolution
variances. This model has a constant mean, no evolving state, and no dependence
on forecast horizon beyond the task's information selection.

## Prior, posterior, and prediction

Use variance `v`, not standard deviation, in the inverse-gamma law:

```text
v ~ InvGamma(a0, b0)          density proportional to v^(-a0-1) exp(-b0/v)
mu | v ~ Normal(m0, v/k0)     k0 is dimensionless mean precision
y[i] | mu,v ~ Normal(mu, v)   conditionally independent observations

k = k0 + n
m = (k0*m0 + n*average(y)) / k
a = a0 + n/2
b = b0 + sum((y[i]-average(y))^2)/2 + (k0*n/k)*(average(y)-m0)^2/2

v | history ~ InvGamma(a,b)
mu | v,history ~ Normal(m,v/k)
mu | history ~ StudentT(df=2*a, location=m, scale=sqrt(b/(a*k)))
y_future | history ~ StudentT(df=2*a, location=m, scale=sqrt((b/a)*(1+1/k)))
```

The predictive law integrates both parameters, rather than substituting their
posterior means into a Gaussian forecast. The Student-t scale is **not** its
standard deviation. When `a>1`, predictive variance is
`b/(a-1) * (1+1/k)`: expected observation variance plus variance of the mean.
Quantiles and HPD intervals use the Student-t law directly. Symmetry and
unimodality make its HPD intervals equal to its central equal-tail intervals.
These model-specific calculations follow [Murphy's conjugate Gaussian analysis](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf),
Sections 6 and 5.5; SciPy supplies [Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
and [inverse-gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html)
distribution functions.

| Setting | Default | Meaning |
|---|---:|---|
| `prior_mean` | 0 | Conditional prior mean in target units |
| `prior_mean_precision` | 0.01 | `k0`; conditional mean variance is `v/k0` |
| `prior_shape` | 3 | `a0`; positive, dimensionless inverse-gamma shape |
| `prior_scale` | 2 | `b0`; positive inverse-gamma scale in squared target units |

The defaults imply prior expected observation variance 1, not known variance 1.
They are illustrative caller choices. The prior mean's posterior coefficient
is `k0/(k0+n)`. Changing mean precision can also change the learned variance
through disagreement between the prior and observations. Strong influence is
saved descriptively and does not fail computation.

All prior settings must be finite; shape, scale, and mean precision must be
positive. Histories must be nonempty, finite, strictly ordered, timezone aware,
and no later than issuance; the target must be later than issuance. Gapped
eligible histories and arbitrary future horizons are allowed by the model;
the runner separately applies daily calendar and release/revision rules.
Unrepresentable posterior parameters, prior-mean weights, scales, or requested
bounds fail visibly.

## Heavy tails and moments

Proper positive priors are accepted even when moments do not exist. With at
least one observation, the predictive mean exists (`2*a>1`). If `a<=1`, posterior
mean-parameter variance, expected observation variance, and predictive variance
are undefined. Their API properties and saved metadata are `None`/JSON `null`,
with an explicit `moment_status`; finite Student-t quantiles and HPD intervals
can still be returned. A mathematically existing moment that overflows or
underflows the numeric representation fails rather than becoming null.

Mean MCSE audits of `mu`/future draws require `a>1`. Audits of expected `v` and
the centered second moments of `mu`/future draws require `a>2`, because the
variances of those audited quantities must exist. The sampler records omitted
moment diagnostics and reasons. Precision, log variance, conditional standardized
mean, and bounded predictive probabilities remain required. Omitting an undefined
moment never turns a missing or failed applicable diagnostic into success.

## Run explicitly

Prepare the input snapshot/tag using [the walkthrough](forecasting.md#prepare-the-input-explicitly).
Use new output directories:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_normal_mean.py \
  --learn-variance --dataset synthetic-mean-shift --tag baseline-input \
  --prior-mean 0 --prior-mean-precision 0.01 --prior-shape 3 --prior-scale 2 \
  --output-directory runs/tutorial-learn-variance-exact-001

PYTHONPATH=src .venv/bin/python scripts/run_normal_mean.py \
  --learn-variance --dataset synthetic-mean-shift --tag baseline-input \
  --mcmc-config configs/normal_mean_variance_mcmc.json --plot-fit-origin 29 \
  --output-directory runs/tutorial-learn-variance-audit-001
```

`--learn-variance` selects the new adapter. Without it, the CLI preserves the
known-variance `NormalMean` behavior. Learned-variance prior flags require this
switch; `--prior-variance`/`--observation-variance` conflict with it. Invalid
combinations fail before output creation. Both adapters accept `--experiment`,
`--evaluation-role`, and `--interval-levels` with the existing semantics.

The supplied audit config requests 4 chains, 1500 warmup and 6000 retained draws
per chain, proposal scale 1.7, and unchanged computation thresholds. It saves
30,000 rows per fit; a 170-fit run is substantially larger than an exact run.
Start by declaring a short development period in an experiment specification.
`--mcmc` uses the shared `MCMCSettings` defaults instead; those defaults are not
guaranteed to pass every joint fit. No settings are tuned automatically.

The CLI automatically runs the backtest and draws `forecast.png`. The trace
flag explicitly requests one saved fit chart. The runner automatically writes
posterior evidence, any requested sampler draws, fit checks and forecast files
before actual selection/scoring. A failed required audit rejects the case and
fails the run while retaining evidence and continuing later origins. Direct
`fit` returns that failed evidence; direct `forecast` raises. Input acquisition,
registration, tags, priors, audit selection, reporting from the library, final
evaluation invocation, and human interpretation remain caller actions.

## Library and replay

```python
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.sampling import MCMCSettings
from model_pipelining.backtest import run_backtest
from model_pipelining.reporting import plot_run, plot_fit

model = NormalMeanVariance()  # add mcmc=MCMCSettings(...) to request an audit
run = run_backtest(dataset="synthetic-mean-shift", tag="baseline-input",
                   model=model, output_directory="runs/library-learn-variance-001")
plot_run(run)  # explicit library reporting
# plot_fit(run, origin_index=29)  # requires a fit saved with an MCMC audit
```

`posterior(request)` returns `NormalInverseGammaPosterior` with fields `mean`,
`mean_precision`, `shape`, `scale`, and `prior_mean_weight`. Properties expose
`degrees_of_freedom`, `mean_scale_squared`, `predictive_scale_squared`,
`observation_variance_mean`, `mean_variance`, and `predictive_variance`.
`predictive_distribution(request)` returns a frozen SciPy `t` law with `.pdf`,
`.cdf`, `.ppf`, `.mean()`, and `.var()`. SciPy's `.var()` can return infinity when
the variance is undefined; use the posterior properties for nullable metadata.
`fit`/`forecast` use the existing fitted-forecast and forecast records.

Replay rebuilds the model, including any nested sampler settings, from the
saved configuration:

```python
model = NormalMeanVariance.from_configuration(manifest["model"]["configuration"])
```

Restore the permanent input, `ExperimentSpec.from_dict(manifest["experiment"])`,
interval levels, and matching code/environment, then run into a new directory.
Replay is a caller action; there is no automatic replay or fit reuse. Seeds use only eligible history,
issuance, target, base seed and chain index. Reordering fits or changing future
values cannot change earlier chains. Global Python/NumPy RNG states are untouched.

## Joint sampler and evidence

The backend remains independent one-walker emcee Gaussian Metropolis chains;
ArviZ receives independent chain/draw arrays. Each chain samples two coordinates:

```text
log_center = log(b) - digamma(a)
log_sd = sqrt(trigamma(a))
v = exp(log_center + log_sd*w)
mu = m + sqrt(v/k)*z
log Jacobian = 1.5*log(v) - 0.5*log(k) + log(log_sd)
```

The target evaluates prior times sufficient-statistic likelihood and includes
this nonconstant Jacobian. Exact posterior centering/scaling and conditional
standardization assist this reference audit; it is not evidence of performance
on difficult posteriors. Starts span both transformed coordinates. There is no
adaptation, thinning, cache, or warm start. Warmup is saved but excluded from
diagnostics; predictive draws have a separate saved seed.

Required audits use the shared [computation criteria](parameter-uncertainty.md#computation-criteria-and-failures):
rank R-hat, bulk/tail ESS, MCSE/SD, and exact-reference agreement. Quantities
include precision `1/v`, log variance, `z`, `z^2`, available parameter/predictive
moments, the predictive CDF of paired future draws, and conditional Gaussian
probabilities at requested exact predictive quantiles and inside exact HPD
intervals. These probabilities average to the exact requested mass; conditional
interval endpoints are never averaged. MCSE measures computation precision,
not predictive spread. HMC diagnostics are explicitly inapplicable.

The existing fit format 1/run format 3 are retained. Metadata adds the joint
posterior, Student-t law, nullable moments, omitted-moment reasons, transform,
versions, seeds, and saved reference distributions. Draw rows add
`observation_variance`, `log_variance`, `standardized_log_variance`, and
`log_jacobian` alongside the existing mean/future/chain/warmup/acceptance fields.
`plot_fit` reads only saved evidence and draws mean, variance, and observation
traces/densities against exact Student-t/inverse-gamma references. Legacy normal
fits remain readable. Partial failures retain draws and permit saved-only plots;
persistence errors can leave a `running` directory as before.

## Verification and limits

[Tests](../tests/README.md) check hand calculations, raw prior-times-likelihood
quadrature, joint-density ratios, predictive mixture integration, HPD mass and
boundary densities, strong prior influence, undefined moments, the transform
Jacobian, joint sampler/reference agreement, stuck/partial failures, RNG/seed
isolation, frozen fits/replay, release/final selection, CLI modes and saved-only
reporting. The model's constant-mean assumption is false for the mean-shift
fixture. Passing tests or audits verifies computation, not that assumption,
prior plausibility, formal simulation-based calibration, or application approval.

[Learned local-level observation variance](local-level-variance.md) is now
implemented using checked scalar integration with fixed evolution variance.
[The joint model](joint-variance.md) also learns a categorical evolution variance;
MCMC-only predictive/HPD approximation remains future work. No shared check-plan
registry or new dependency is added by that extension.

## Prior and posterior predictive checking

[The separate checking workflow](predictive-checks.md) now simulates whole
datasets from this model's exact prior and posterior. It offers data-free prior
checks and as-of development-history posterior replications, with saved mean/
variance draws, datasets, discrepancy statistics, finite-simulation precision,
and charts. There is no MCMC in that workflow. Large discrepancies remain
human review evidence rather than computational failures. Invoke it explicitly;
backtests do not run it automatically.

[Prior influence comparisons](prior-influence.md) now follow the same model
through identical historical as-of histories under explicit caller-chosen
location/precision and shape/scale settings. They save exact learning and
prediction paths, scale components and signed differences from a reference,
starting before any data. Strong influence has no failure threshold. The
comparison CLI draws saved-only charts; backtests do not invoke it automatically.
