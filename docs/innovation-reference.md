# Reference for occasional large state innovations

The bounded prototype is implemented and independently checked. It enumerates
every positive-prior innovation sequence for **at most eight daily observations**,
integrates one shared unknown observation variance, and compares batch Gaussian
conditioning/adaptive integration with Kalman smoothing/refined quadrature.
The seven recorded cases agree within 4.78e-11 in the stated scaled comparison;
all three Gaussian limiting cases agree with the existing scalar model within
3.90e-12. This establishes a short-history numerical reference for investigating
the [transition question](model-review-study.md#bounded-next-modeling-question).

It is a reference calculation, not a forecasting adapter or longer-history
approximation. No transition hyperparameter is learned. Model adequacy,
fresh-path predictive criticism and simulation calibration remain separate work.

## Model and timing

Before the first observation, the initial state has distribution
`theta[-1] ~ Normal(initial_mean, initial_variance)`. Independently,
`V ~ InverseGamma(prior_shape, prior_scale)`. For each daily transition:

```text
z[t] ~ Bernoulli(jump_probability), independently across t
theta[t] = theta[t-1] + innovation[t]
innovation[t] | z[t] ~ Normal(0, small_variance or large_variance)
y[t] | theta[t], V ~ Normal(theta[t], V)
```

`z[t]=1` selects the large variance. The two configured quantities are variances,
not standard deviations. The first innovation occurs before `y[0]`. The forecast
adds one fresh daily transition and observation after the supplied history.
Unlike [one global categorical W](joint-variance.md), past labels may differ
within a trajectory. The next label's probability remains the fixed caller
probability even when past posterior jump probabilities are large.

The reference accepts numerical arrays only: immediately available, equally
spaced daily observations, without release/vintage selection or final-period
claims. It learns V jointly with allocations and states. Initial variance may
be zero; small variance may also be zero. Real known-state probability atoms
are preserved. Positive observation variance keeps the next-observation law
continuous. The prototype requires `prior_shape > 1`, ensuring a finite prior
expected observation variance; this is a deliberate reference-scope restriction.

## Two independent calculations

For a past label sequence z, let `w[k]` be its selected innovation variance.
The batch calculation constructs the observation-state covariance

```text
B[i,j] = initial_variance + sum(w[k] for k <= min(i,j))
Cov(y | z,V) = B + V * Identity
```

It uses dense Cholesky likelihoods and Gaussian conditioning of initial/final
states. Error-loading (Joseph) covariance formulas avoid subtracting almost
equal positive variances. Adaptive integration covers the entire positive V
axis with `V=(prior_scale/prior_shape)*t/(1-t)`, `0<t<1`; an evidence-scaled second
pass keeps absolute integration error meaningful. No Kalman recursion or
finite log-V support is used in this calculation.

The other calculation runs scalar Kalman filters for every sequence and V node,
then backward smoothing for the initial state and its covariance with the final
state. It integrates in log V using Gauss-Legendre rules, doubling node counts
until two successive refinements agree. Support expands separately. The bound
`L(V) <= (2*pi*V)^(-n/2)` supplies inverse-gamma bounds on omitted probability
and the first V moment, also after summing sequence priors.

Defaults require refinement change <=1e-7, omitted probability <=1e-9 and
omitted V moment <=`1e-9*max(1,E[V|y])`. Adaptive relative/absolute integration
tolerances are 1e-9/1e-10. Adaptive error estimates must fit a separate budget;
integration failures retain diagnostics. Finite precision and refinement agreement
are numerical evidence, not a proof for every pathological input.

Both calculations retain the joint allocation/V/state posterior. For example,
the full next-observation law sums both possible *future* innovation variances
inside each past-sequence/V integral. It never substitutes one fitted variance,
one Gaussian with matched moments or averaged conditional interval endpoints.
The prototype returns CDF/density evaluations at caller-specified points, not
HPD regions, predictive samples, a serialized `Forecast` or log-density scores.
Ordinary floating-point density arithmetic can underflow in remote tails;
there is no density floor or claim of a tail log-loss implementation.

Comparison covers log evidence, every sequence mass, per-transition jump
probabilities, initial/final state moments and covariance, V mean/CDF,
`Cov(V,sum(z))`, next-observation moments, CDF/density and strict/inclusive state
CDFs. Means use a documented natural scale, second moments its square, CDF/mass
errors absolute probability, and density error multiplied by that scale.
Log-evidence error is absolute. The maximum scaled difference must be <=5e-6.
The natural scale is
`s=max(1, sqrt(initial_variance+n*large_variance), sqrt(prior_scale/(prior_shape-1)), max(abs(y-initial_mean)))`;
the last term is zero for empty history. The adaptive max-norm estimate describes
the scaled integrated vector, not a rigorous bound on every normalized posterior
quantity. The independent normalized-output comparison is a separate check.
Rare sequence masses have an absolute error comparison, not a claim of relative
precision in arbitrarily small probabilities. All positive-prior sequences stay
present; unrepresentable whole-sequence mass fails instead of silently pruning it.

## Run explicitly

From the repository root after [setup](../README.md#development-environment):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_innovation_reference.py \
  --config configs/innovation_reference.json \
  --output-directory runs/innovation-reference-new
```

Choose an unused output directory. The command validates all inputs, freezes
expanded settings, query grids and numerical cases, runs both calculations and
the three Gaussian limits, then draws a chart from saved successful results.
Invalid input raises before output creation; unresolved computations save failed
case diagnostics, continue later cases and return exit status 1. No existing
output is overwritten. No data is fetched, registered or tagged, and no backtest
or evidence runner invokes this command automatically. The fixed default config
is `configs/innovation_reference.json` when `--config` is omitted; no directory
discovery or tuning occurs.

| File | Purpose |
|---|---|
| `design.json` | Expanded model/settings and all histories/query grids, frozen before calculations |
| `study.json` | Status, case files, input/config hash, source hashes and environment |
| `cases/<name>.json` | Both numerical results, all past labels/masses, convergence/tail evidence; failures retain error/evidence |
| `gaussian_limits.json` | Zero/one jump probability and coincident-variance comparisons to `LocalLevelVariance` |
| `summary.json` | Case sizes, sequence/node counts, agreement, V/state/jump summaries |
| `runtime.json` | Per-case timing, separated from reproducible numerical results |
| `feasibility.json` | Enumeration cap and exact exponential sequence counts |
| `checks.json`, `review.json` | Computational status and separate pending human interpretation |
| `reference.png` | Saved-only next-observation CDF/density and past jump probabilities |

Library use:

```python
from model_pipelining.innovation_reference import (
    InnovationMixtureReference, run_innovation_reference,
)

result = run_innovation_reference(
    (0., .1, -.1, 3.8, 4.1, 3.9),
    model=InnovationMixtureReference(
        small_variance=.01, large_variance=4., jump_probability=.05,
    ),
    predictive_points=(-2., 0., 2., 4., 6.),
)
assert result["status"] == "completed"
posterior_jump_probabilities = result["batch_adaptive"]["jump_probabilities"]
```

`ReferenceSettings` selects precision/resource limits. Query grids must be
strictly increasing, finite tuples/lists of at most 64 points (positive for V).
Empty history evaluates the prior with one empty past sequence. Histories longer
than eight are rejected before enumeration, including degenerate Gaussian cases;
use the existing scalar model for longer Gaussian histories. The model record
has no `forecast()`/evidence-provider interface and is not passed to `run_backtest`.

## Recorded exercise and interpretation

The complete local exercise is `runs/innovation-reference-verified-20261119/`.
Its seven numerical case files replay byte for byte against the earlier
development exercise with the same settings. Source/environment metadata,
wall-clock timings and plots retain their separate scopes. Curated summary and
figure exports below are deliberately tracked; ordinary commands save local
artifacts only, and updating curated exports remains a caller action.

The [retained numerical summary](artifacts/innovation-reference-summary.json)
contains sequence/node counts and posterior values for every case.
[The saved chart](figures/innovation-reference.png) overlays both methods' CDF
and density probes and shows the per-transition posterior jump probabilities.
Lines connect stored query points; they are a display guide, not a new density
interpolator or additional numerical fits.

| Case | Observations | Past sequences | Largest scaled discrepancy |
|---|---:|---:|---:|
| Prior | 0 | 1 | 1.50e-11 |
| Short | 3 | 8 | 1.08e-12 |
| Quiet | 6 | 64 | 7.23e-14 |
| Gradual | 6 | 64 | 3.07e-13 |
| Abrupt | 6 | 64 | 3.14e-11 |
| Dominant | 6 | 64 | 1.14e-13 |
| Cap-eight | 8 | 256 | 4.78e-11 |

With the illustrative fixed probability 0.05, quiet-history posterior jump
probabilities range roughly 0.014–0.024. In the abrupt case, the fourth
transition's probability is about 0.9913 and final-state mean 3.7788. Its V mean
is 0.5341. The larger-change example concentrates that transition's probability
near 0.9993 without discarding other paths. Future jump probability stays 0.05;
the model does not learn a persistent high-jump regime.

The gradual case's final-state mean is about 0.6422 while its last observed value
is 1.3. That short numerical example retains the earlier concern about directional
drift; these deliberately chosen histories do not establish comparative model
adequacy. The saved V/jump-count covariance demonstrates that allocation and
observation noise are not treated as posterior-independent.

## Feasibility and next work

At eight observations the reference retains 256 past sequences and, in the
recorded calculation, 65,536 joint sequence/V nodes. Evaluating the future law
also includes both next-label branches. Nondegenerate enumeration at 160
observations would require `2**160`, about 1.46e48 past sequences before V
integration. Short-case timings do not establish long-history feasibility.

The [particle prototype](innovation-particles.md) now implements a controlled
longer-history investigation against this oracle, with independent-run MCSE,
refinement and retained rare-path uncertainty failures. The oracle remains
unchanged. Its original requirements for that investigation follow: State which probabilities/components it retains, how omitted mass or
Monte Carlo error is measured, and how it compares on rare/dominant allocations,
noise confounding, state atoms and full predictive laws. Preserve the Gaussian
limits. Do not infer an adequate approximation merely from a small predictive
mean error. Only then extend the shared forecasting/evidence interfaces and run
fresh-path criticism, prior comparisons, calibration/recovery and the matched
independent-seed stable/gradual/abrupt studies specified in the earlier decision.
No long-history inference method or generic MCMC backend is selected here.

Fourteen tests cover independent conditioning, direct one-observation V integrals,
exact inverse-gamma/Student-t calculations, all Gaussian limits, label
nonidentifiability at equal scales, fresh future probability, joint dependence,
real state atoms, all 256 sequences, invalid/bounded inputs, precision failures,
deterministic replay/global-RNG isolation and CLI persistence/failure/replay.
