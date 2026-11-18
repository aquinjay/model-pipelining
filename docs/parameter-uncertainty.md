# Parameter uncertainty and an exact-reference MCMC audit

Implemented: `NormalMean` learns an unknown constant mean, integrates its
posterior into observation forecasts, and optionally audits each eligible fit
with independent Metropolis chains. **Forecasts use exact integration in both
modes**; they are not estimates obtained from those chains. The separate
`LocalLevel` model continues to condition on fixed variance settings.

## Model and exact calculation

```text
mu ~ Normal(m0, P0)
y[i] | mu ~ independent Normal(mu, V)

C = 1 / (1/P0 + n/V)
m = C * (m0/P0 + sum(y)/V)
mu | history ~ Normal(m, C)
y_future | history ~ Normal(m, V + C)
```

`prior_mean=0`, `prior_variance=100`, and `observation_variance=1` are illustrative
caller settings. Both variances must be positive, finite squared target units.
Observation variance is known; the mean is learned afresh from each request.
The implementation uses scaled variance weights to avoid overflowing precisions;
unrepresentable or nonfinite calculations fail visibly.

The prior-mean coefficient is `V / (V + n*P0)`, conditional on both variances.
Changing prior precision also changes posterior/predictive variance. Strong
influence can complete successfully; there is no criterion that it should be small.

Prediction integrates `p(y_future | mu) p(mu | history)` over mu. Using only
`Normal(m, V)` would omit parameter uncertainty. Returned observation quantiles
and HPD regions instead use `V+C`. For this symmetric unimodal normal law,
HPD regions are central intervals with equal endpoint densities. This is a
model-specific fact. Averaging conditional interval endpoints is not integration;
see [Stan posterior prediction](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html).

The adapter accepts strictly ordered aware observations no later than issuance
and a future target. Its exchangeable assumption supports gapped eligible
histories, delayed issuance, and longer horizons. It has no evolving level or
horizon-dependent variance. The runner independently enforces daily calendar
and release/vintage rules. The mean-shift fixture violates the constant-mean
assumption; correct computation does not establish suitability.

## Run exact forecasts or request an audit

Set up the [locked environment](../README.md#development-environment) and prepare
the sample snapshot/tag using [the walkthrough](forecasting.md#prepare-the-input-explicitly).
Choose unused output directories:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_normal_mean.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --output-directory runs/tutorial-normal-mean-exact-001

PYTHONPATH=src .venv/bin/python scripts/run_normal_mean.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --mcmc-config configs/normal_mean_mcmc.json --plot-fit-origin 29 \
  --output-directory runs/tutorial-normal-mean-audit-001
```

The 200-row example produces 170 fits. The default audit stores 12,000 rows per
fit, so it needs more time and disk space. Start with a caller-declared short
development period when inspecting sampler mechanics. No fits are cached or
warmed from other origins. `--mcmc` selects defaults without a file; it is
mutually exclusive with `--mcmc-config`. The config selects sampler settings;
`--prior-mean`, `--prior-variance`, and `--observation-variance` configure the model.

The script automatically runs the experiment and draws `forecast.png`.
`--plot-fit-origin` explicitly requests one saved trace; 29 is the zero-based
origin index after 30 observations. It requires an audit and a selected origin.
Failed computational runs return exit code 1. Optional chart failures occur
after the backtest and do not change its saved status.

All three run scripts accept `--experiment FILE`, `--evaluation-role final`,
and `--interval-levels`. With an experiment, omit `--minimum-history` and choose
it in the specification. Final evaluation is a separate explicit invocation;
role selection does not certify untouched data. Acquisition, registration, tags,
configuration choice, and human interpretation remain caller actions.

For restricted home directories, configure writable dependency caches first:

```bash
export XDG_CACHE_HOME=/tmp/model-pipelining-cache
export MPLCONFIGDIR=/tmp/model-pipelining-matplotlib
```

Ordinary writable-home setups need no extra step. ArviZ 0.23.4 writes a daily
migration-warning marker in its user cache on import; matplotlib also uses a
cache. The implementation does not change environment variables or global
random states to work around these dependencies.

## Library and replay

```python
from model_pipelining.backtest import run_backtest
from model_pipelining.normal_mean import NormalMean
from model_pipelining.sampling import MCMCSettings
from model_pipelining.reporting import plot_fit, plot_run

model = NormalMean(prior_mean=0, prior_variance=100, observation_variance=1,
                   mcmc=MCMCSettings(seed=0))  # omit mcmc for exact-only fits
run = run_backtest(
    dataset="synthetic-mean-shift", tag="baseline-input", model=model,
    output_directory="runs/tutorial-normal-mean-library-001",
)
plot_run(run)                    # explicit library reporting
plot_fit(run, origin_index=29)   # explicit saved-only sampler report
```

`posterior(request)` returns `MeanPosterior(mean, variance, prior_mean_weight,
observation_variance)` with a `predictive_variance` property.
`predictive_distribution(request)` returns the exact `NormalDist`.
`fit(request)` returns `FittedForecast(forecast, evidence)` even when required
audit criteria fail, retaining the exact candidate for inspection.
`forecast(request)` rejects failed audits with `ValueError`. Direct methods do
not persist files. The runner calls `fit` and saves evidence for adapters
with `supports_fit_evidence=True`, rejecting failed candidates.

Reconstruct nested settings from a saved manifest explicitly:

```python
configuration = dict(manifest["model"]["configuration"])
sampler_configuration = configuration.pop("mcmc")
model = NormalMean(**configuration, mcmc=(
    MCMCSettings(**sampler_configuration) if sampler_configuration is not None else None
))
```

Restore the permanent input, `ExperimentSpec.from_dict(manifest["experiment"])`,
interval levels, and matching code/environment. Invoke `run_backtest` in a new
directory. Numerical files and fit evidence replay; run timestamps differ.
No automatic restoration is supplied.

## Sampler and reproducibility

The backend is emcee 3.1.6 using its documented [Gaussian Metropolis
move](https://emcee.readthedocs.io/en/stable/user/moves/#metropolis-hastings-moves).
Each chain is a separate one-walker sampler with an independent proposal,
not an interacting ensemble. Coupled walkers are never presented as independent
chains to diagnostics. No home-grown sampler or compiler backend is introduced.

The sampled coordinate is `z=(mu-exact_mean)/exact_sd`. The target evaluates
prior and sufficient-statistic likelihood, dropping parameter-independent
constants; the transform has a constant Jacobian. Exact centering/scaling is
deliberate assistance for this calibration exercise, not evidence of performance
on difficult posteriors. Starts are evenly spaced from -4 to +4 exact posterior
SDs; proposal SD is 2.4. There is no adaptation, tuning, fit reuse, or thinning.
Warmup is saved and excluded from all summaries and diagnostics.

Each chain's seed derives from the base seed, eligible UTC timestamp/value
history, issuance, target, and chain index. It excludes the whole-snapshot ID
and ineligible values: changing future data cannot change an earlier chain.
A separate saved predictive seed generates a paired `future_value ~ Normal(mu,V)`
for every row. Chain/draw dimensions are retained for diagnostics. Python and
NumPy global random states remain unchanged.

## Computation criteria and failures

ArviZ 0.23.4 supplies [rank R-hat](https://python.arviz.org/en/v0.23.0/api/generated/arviz.rhat.html),
[bulk/tail ESS](https://python.arviz.org/en/v0.23.0/api/generated/arviz.ess.html),
and [mean MCSE](https://python.arviz.org/en/v0.23.0/api/generated/arviz.mcse.html).
The dependency declaration keeps this pre-1.0 API; lockfile and pip export pin
exercised versions. Migrating to the modular API is a separate compatibility
change. NumPy and SciPy are explicit direct dependencies.

| Criterion | Default |
|---|---:|
| Independent chains | 4 |
| Warmup / retained draws per chain | 1000 / 2000 |
| Maximum rank-normalized split/folded R-hat | 1.01 |
| Minimum bulk ESS / tail ESS | 400 / 400 |
| Maximum mean MCSE / quantity SD | 0.1 |
| Maximum absolute reference error | `6 * mean_MCSE + 64 * ulp(reference)` |

Thresholds are explicit caller configuration, saved per fit. Six MCSEs provide
a finite-simulation comparison tolerance, not a deterministic guarantee or model
approval. The ULP term handles representation error. No prior-influence or
empirical-coverage threshold is introduced. Defaults need not pass every fit.

Diagnosed quantities are mu (also the conditional predictive mean),
`(mu-exact_mean)^2`, future-observation draws and their centered second moment,
conditional observation CDF values at each requested exact predictive quantile,
and conditional probabilities inside each requested exact HPD region.
The latter average conditional **probabilities**, with references equal to the
requested quantile probability or HPD mass. They do not estimate interval endpoints
or density thresholds. This Rao-Blackwellized audit checks predictive integration;
forecasts never use these estimates. Moment quantities have analytic SD references;
probability integrands use their empirical retained-draw SD for the MCSE ratio.

Missing/nonfinite diagnostics, zero MCSEs, and constant chains fail explicitly.
All required criteria must pass. Acceptance fractions remain backend evidence.
HMC divergences, depth, and energy diagnostics are explicitly inapplicable for
Metropolis. [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html)
explains posterior spread versus finite-computation precision.

Chain failures retain completed/partial rows, record the error, and continue
other chains. Failed required checks retain evidence, fail that forecast case,
and let later cases continue. Exact forecasts do not bypass requested audit
failures. Any required case failure fails the run; human review stays pending.
Persistence errors propagate, leaving a possibly partial `running` directory.
Invalid model/settings raise before run creation. No recovery is implemented.

## Saved evidence and remaining work

Version-3 manifests gain an optional fit capability and fingerprint the inference
helper and used sampler source. Existing version-1/2/3 reports remain readable.

| File | Contents |
|---|---|
| `fit_index.csv` | Every selected case, history hash, relative fit directory or missing-evidence error, and accepted/failed status |
| `fits/NNNNNN/fit.json` | Fit format version 1; runner-owned input/history/issuance/target identity, posterior/predictive moments, configuration, required checks, diagnostics/tolerances, transforms, starts, seeds, backend versions, failures |
| `fits/NNNNNN/draws.csv` | Audit only: chain, within-phase draw index, warmup flag, mu, standardized mu, paired future value, log target, backend acceptance indicator |
| `fits/NNNNNN/trace.png` | Explicit `plot_fit` or CLI flag: warmup separation, parameter/predictive traces and retained densities against exact references |

Exact-only fits save posterior evidence and an inapplicable MCMC check, with no
invented chains. Fit evidence and forecasts are written before actual-vintage
selection/scoring. Origin indices refer to the complete period calendar, not
positions in a filtered cohort. Generated files are ignored by Git; preserve
whole runs, inputs, and matching environment for replay. Files remain writable.

[Tests](../tests/README.md) independently integrate raw prior times likelihood,
check predictive CDFs and HPD mass/density, and exercise strong prior influence,
sampler target ratios/agreement, bad mixing, biased samples, missing diagnostics,
partial backend failures, future isolation, seeds, freezing, replay, final/release
selection, saved-only charts, and persistence failures.

This supplies one concrete model and optional fit dispatch. Unknown observation/
evolution variances, MCMC-only forecasts/HPD estimation for intractable laws,
a general sampler/check-plan registry, adaptation/caching, full-data criticism,
and formal simulation-based calibration remain future work. Computation does
not approve priors, the constant-mean assumption, or an application.
