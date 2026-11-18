# A changing level with learned observation variance

The model now supplies an evidence provider to the shared predictive/prior
workflows. Its numerical inference, joint initial-state draws and fresh-process
replication semantics are unchanged. [Experiment evidence requirements](evidence.md)
can explicitly request these workflows and name its per-origin precision checks.

`LocalLevelVariance` learns one unknown observation variance while filtering a
level that changes over time. Evolution variance remains an explicit fixed
setting. The initial-state prior and evolution variance are independent of the
observation variance:

```text
V ~ InvGamma(prior_shape, prior_scale)
level[0] ~ Normal(initial_mean, initial_variance)
level[t] | level[t-1] ~ Normal(level[t-1], elapsed_days * evolution_variance)
y[t] | level[t], V ~ Normal(level[t], V)
```

The initial prior is one day before the first selected observation, matching
the original local-level convention. Later transitions use actual elapsed UTC
days, including gaps. Observation variance is constant within each modeled
history; its posterior updates as more observations become available. The
changing quantity is the hidden level, not a time-varying noise parameter.

The original [LocalLevel](local-level.md) retains fixed variances and analytic
Gaussian forecasts. [NormalMeanVariance](normal-mean-variance.md) retains a
constant mean and conjugate Student-t forecasts. This model's observation
forecast is generally a **Gaussian mixture**, not a Student-t distribution.
Setting evolution variance to zero leaves an independent Gaussian mean prior;
it does not turn that prior into the conditional normal/inverse-gamma prior
used by `NormalMeanVariance`. A known initial level with zero evolution does
have a conjugate variance posterior, and is checked as a limiting case.

## Inference and numerical acceptance

For each candidate `V`, the Kalman filter integrates out the state sequence and
evaluates the marginal data likelihood. Conditional last-state and forecast
distributions are Gaussian. The posterior of `V` is proportional to its prior
times that likelihood. Because the state prior and evolution variance have
fixed absolute scales, the variance posterior is generally nonconjugate.

Only one scalar parameter needs integration. The implementation uses
Gauss-Legendre quadrature over `log(V)`, including the change-of-variable factor,
with stable log weights. It expands finite support and doubles quadrature
order. Two successive refinements must agree on normalization, variance/state
summaries and predictive CDF probabilities. Existing mathematical moments that
cannot be represented numerically fail visibly.

The omitted-tail calculation uses the pointwise likelihood bound
`L(V) <= (2*pi*V)^(-n/2)`. Multiplying the inverse-gamma prior by this bound
gives a constant times an inverse-gamma density with shape `prior_shape+n/2`
and scale `prior_scale`. Its two tails bound unnormalized omitted probability;
the normalized estimate uses the quadrature evidence. A corresponding
first-moment tail bound is checked when expected observation variance exists.
These numerical checks are saved evidence, not a mathematical proof of error
for arbitrary inputs. Unresolved refinement/support fails the requested fit.

| `QuadratureSettings` field | Default |
|---|---:|
| `minimum_nodes` | 32 |
| `maximum_nodes` | 1024 |
| `relative_tolerance` | `1e-6` |
| `tail_tolerance` | `1e-8` |
| `maximum_expansions` | 9 |

Normalization uses absolute log-evidence change (a relative-evidence check).
Refinement uses a relative comparison with an absolute scale floor for moments;
CDF comparisons use absolute probability error. Support begins around the
prior variance mode and expands geometrically within the representable log
range `[-700,700]`. The prior center must lie within `(-690,690)`; more extreme
scales are rejected. Exhausted precision budgets are failures, not permission
to publish an unchecked forecast. Callers may explicitly select different
settings with [the example config](../configs/local_level_quadrature.json).
Heavy-tailed priors can require more nodes than the defaults; for example, the
shape-0.2 undefined-prior-mean test explicitly allows 2048 nodes. Moment
undefinedness alone never causes a failure, while unresolved numerical precision
does. A wide tail-component bracket never loosens the quantile solver's absolute
tolerance near the bulk.

No MCMC, adaptation, fitted-state cache, or new dependency is introduced. The
small cached Gauss-Legendre rules contain only fixed mathematical nodes and
weights; model fits retain no mutable history. Filtering and smoothing follow
the [Gaussian state-space reference](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf);
SciPy provides [quadrature rules](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html).

## Marginal prediction and HPD regions

For a target `h` elapsed days after the last eligible observation:

```text
p(y_future | history) = integral Normal(m[V], C[V]+h*W+V) p(V | history) dV
```

Numerical posterior weights combine the conditional densities/CDFs. Quantiles
solve the marginal CDF. Conditional interval endpoints are never averaged.
Forecast variance includes expected conditional variance and variation of
conditional means. State uncertainty and observation uncertainty remain distinct.

Mixture HPD regions are numerical density superlevel sets. Component centers,
component-scale knots and a central grid locate crossings; roots are refined,
and mixture CDF differences evaluate segment probability. A doubled resolution
must agree on segment count/boundaries, mass and boundary density. Disconnected
segments are preserved, including their gaps. A saved small weight budget can
omit component-scale knots for tiny components; all components still participate
in PDF/CDF evaluation and contribute their centers. The search's excluded tail
probability and resolution evidence are recorded. This is a numerical resolution
test, not a completeness proof for arbitrary pathological mixtures; unresolved
cases fail. There is no automatic equal-tail-to-HPD relabeling.

Requested quantile tails and HPD exclusions must be compatible with integration
tail precision. Increase precision explicitly for more extreme requests. Failed
output checks retain the candidate mean and fit evidence but reject the forecast.

For `a = prior_shape+n/2`, expected `V` and observation-predictive variance are
undefined when `a<=1`. Saved moments are null with explicit status, although
finite quantiles/HPD regions may exist. Filtered level moments remain finite
for representable inputs because their conditional variances and means are
bounded. The finite quadrature mixture itself has finite numerical moments;
use the source model's nullable saved moments when its true variance is infinite.

## Run forecasts

Prepare the preserved input using [the walkthrough](forecasting.md#prepare-the-input-explicitly),
then use a new output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_level.py \
  --learn-observation-variance \
  --dataset synthetic-mean-shift --tag baseline-input \
  --initial-mean 0 --initial-variance 100 --evolution-variance 0.1 \
  --prior-shape 3 --prior-scale 2 \
  --quadrature-config configs/local_level_quadrature.json \
  --output-directory runs/tutorial-local-variance-001
```

`--learn-observation-variance` selects the new adapter; omit it to retain the
fixed-variance model. `--observation-variance` conflicts with learned mode.
Variance-prior/numerical flags require learned mode. Existing experiment,
role, input, interval and output arguments retain their semantics. The CLI
automatically saves a single prior trajectory, runs historical forecasts and
draws `forecast.png`. That trajectory is for inspection, not automatic prior
approval; use the full predictive-check workflow for replicated datasets.

The learned adapter accepts ordered aware eligible histories, delayed issuance,
gaps and positive future horizons. State variation scales with elapsed UTC days,
including fractional days when issuance/target timing requires them. The runner
separately enforces its daily task/release rules. The old fixed-variance adapter's
narrower timing contract is unchanged.

## Predictive checks and prior comparisons

Use [the existing predictive-check workflow](predictive-checks.md) explicitly:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_normal_mean_variance.py \
  --model local-level-variance \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59Z \
  --experiment configs/daily_forecast_experiment.json --replications 1000 \
  --output-directory runs/tutorial-local-variance-checks-001

PYTHONPATH=src .venv/bin/python scripts/compare_priors.py \
  --model local-level-variance \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59Z \
  --experiment configs/daily_forecast_experiment.json \
  --config configs/local_level_variance_priors.json \
  --output-directory runs/tutorial-local-variance-priors-001
```

Prior-only checks also support `--model local-level-variance --prior-only
--observation-count N`. Initial-state and evolution flags replace constant-mean
location/precision flags. Both models share inverse-gamma shape/scale flags.
The explicit quadrature config applies only to the learned local-level model.

Each prior replication samples `V`, an initial level, fresh state innovations
and observations. Each posterior replication samples a variance node using
checked quadrature weights, then an initial level from its conditional Gaussian
posterior. That initial posterior uses smoothing **within the selected history**.
It then simulates a new state trajectory and new observation noise, preserving
variance/initial-state dependence while resimulating all process innovations.
It does not copy the fitted state path and add observation noise. Histories with
gaps use those elapsed intervals. These are full-data model replications, not
held-out forecasts; their Monte Carlo precision is separate from quadrature error.

Prior comparisons hold evolution variance and numerical settings fixed. They
compare initial-state location/variance and observation-variance shape/scale on
identical as-of histories, including revisions and an explicit cutoff. State
and variance intervals are equal-tail credible intervals; predictive HPD sets
retain every segment. There is no conjugate scale decomposition, scalar
prior-location coefficient, or Student-t degrees of freedom for this model;
those fields are null rather than invented. Saved integration checks explain
numerical acceptance at every update. Strong influence never fails computation.

All existing release/final-boundary guards apply. Backtests do not invoke checks
or comparisons automatically. CLIs calculate then plot; library plotting remains
explicit. Human interpretation stays pending.

## API, saved evidence and replay

```python
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings

model = LocalLevelVariance(
    initial_mean=0, initial_variance=100, evolution_variance=0.1,
    prior_shape=3, prior_scale=2, quadrature=QuadratureSettings(),
)
# model.forecast(request) -> Forecast; model.fit(request) -> FittedForecast
# model.posterior(request) -> VariancePosterior with passed integration checks
# model.predictive_distribution(request) -> NormalMixture
```

`VariancePosterior` exposes variance nodes/weights, conditional filtered level
moments, log bounds/evidence and saved precision checks. Properties give nullable
expected observation variance and integrated level moments. `.cdf`/`.ppf` evaluate
variance probabilities using additional quadrature on the resolved support;
`.predictive(elapsed_days)` returns the finite numerical Gaussian mixture.
`.prior()` integrates the variance prior before any observations. Unresolved
posterior/prior integration raises `NumericalIntegrationError` carrying evidence.
`fit` instead retains candidate evidence and required failures; `forecast` raises
on those failures. Direct methods do not persist artifacts.

The existing runner's optional fit dispatch saves format-1 evidence before
accepting/scoring a forecast. Metadata includes quadrature settings, versions,
support/refinement/tail checks, weighted nodes/conditional moments, elapsed
projection time, HPD resolution checks and output errors. Quadrature nodes are
saved as metadata, not mislabeled MCMC draws. Existing run/fit formats remain
unchanged. Fit traces for MCMC are not applicable to this inference mode.

Predictive-check format 1 gains the kind `local_level_variance_predictive_checks`.
Parameter rows include `initial_level`; the constant `mean` column is empty.
`latent_levels.csv` retains every fresh state trajectory. Posterior phase
metadata records quadrature evidence; failures retain that evidence while prior
simulation results survive. Prior-comparison kind
`local_level_variance_prior_influence` adds `integration_checks.json` and
explicit predictive segment JSON/count columns. Old artifact kinds remain
readable by saved-only plotting.

Restore nested configuration with
`LocalLevelVariance.from_configuration(manifest["model"]["configuration"])`.
Then restore the permanent input, experiment/role, interval requests and matching
source/environment into a new directory. Numerical forecasts and evidence replay;
execution timestamps differ. Simulation retains independent per-replication
PCG64 streams and leaves global RNG states untouched. Only fixed quadrature
rules are cached. Generated runs/plots remain local and ignored by Git.

## Verification and scope

Tests compare fixed-variance filters with an independently constructed batch
Gaussian covariance and likelihood, integrate the scalar posterior using
independent adaptive quadrature, and check the known-level conjugate limit.
They independently verify initial-state smoothing, forecast probabilities,
multimodal HPD sets, fresh-process replication moments, release/final isolation,
numerical failure persistence, CLI compatibility and saved-only reporting.

Evolution variance is still fixed. These checks do not establish that a random
walk, Gaussian measurement noise, prior choices or a particular fixed evolution
variance suit an application. Sample-based calibration of MCMC-only fits and fully
learned evolution/observation variances remain future work. No automatic model
ranking, application choice, deployment or approval rule is added.

The local exercise under `runs/local-level-variance-20261117-bs3pk_xh/` completed
130 development forecasts on each 200-row fixture, excluding its final 40
values. All 143 numerical/fit files per series replayed exactly with matching
configuration/environment. Each series also exercised 1000 prior and posterior
fresh-path replications and five prior settings over 162 learning points.
These known synthetic fixtures are development evidence; the selected model
and fixed evolution variance still require interpretation.

[Continuous posterior-CDF calibration](calibration.md) is now implemented for the mean/variance and learned local-level models as a separate model-level workflow.
