# Calibration with discrete variances and state probability atoms

The [joint model](joint-variance.md) now supports the data-free calibration
workflow. Observation variance is continuous, evolution variance is categorical,
and state distributions can mix Gaussian components and known-state atoms.
Known parameters are excluded: a single fixed W support point is not calibrated,
and a fixed initial state is not treated as an unknown parameter.

## Preserve and randomize real CDF jumps

For a continuous posterior, evaluating its CDF at a simulated true parameter
produces the usual calibration variable. For a discrete or mixed posterior the
inclusive CDF alone is generally not uniform. The workflow records both limits
and uses the randomized distributional transform:

```text
left  = P(parameter < simulated_truth | data)
right = P(parameter <= simulated_truth | data)
U     = left + Uniform(0,1) * (right-left)
```

This extends the probability-integral-transform idea with genuine probability
mass, following the randomization principle in
[Dunn and Smyth's randomized quantile residuals](https://gksmyth.github.io/smyth/pubs/residual.html).
Our application is posterior calibration, not an observed-response residual.
With no jump, the transform is exactly the original CDF and draws no random
number. No positive state variance is invented to hide an atom.

For example, when the initial state is known and W=0, the final state is also
known. A joint posterior over W=0 and W>0 therefore gives the final state an atom
plus Gaussian components. Its left/right state CDFs preserve that mass.
Conditional variances that should be positive but are unrepresentable fail
explicitly; they are not turned into model probability atoms.

Randomization uses independent parameter-labelled PCG64 streams:
`SeedSequence([seed,3,trial_index,first_64_bits_SHA256(parameter)])`, with a
little-endian label. Prior/data draws retain their existing `[seed,2,trial_index]`
streams. Extending trial count or changing parameter iteration order preserves
existing data and transformed values; global RNG states remain untouched.

## Credible interval mass and interpretation

`calibration.csv` saves the transformed `cdf`, raw `cdf_lower`/`cdf_upper`, a
`randomized` flag, and each closed interval's actual posterior `interval_mass`.
The posterior median and bounds remain actual support values for categorical W.
A nominal 95% equal-tail interval can contain 100% probability on a finite
support. Its repeated closed-interval coverage should be interpreted alongside
that mass, rather than assuming it must equal exactly 95%.

Summaries retain coverage Monte Carlo errors/Wilson intervals, add randomized
trial counts and mean interval mass, and use the transformed CDFs for uniform
reference plots. Uniform reference bands apply only when every trial computes;
failed fits can induce selection bias. Computation status does not automatically
accept or reject calibration. Marginal calibration alone can also pass for
prior-only inference, so the independent likelihood, numerical-reference,
parameter-recovery and joint-dependence tests remain necessary.

## Run explicitly

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_model.py \
  --model local-level-joint-variance \
  --config configs/local_level_joint_priors.json \
  --observation-count 12 --trials 200 --seed 812 \
  --output-directory runs/joint-calibration-example-001
```

The command uses the reference prior and automatically draws the saved chart.
For a known initial state, explicitly supply a configuration with
`initial_variance: 0`; include zero and positive W support to exercise mixed
state atoms. Library callers invoke plotting separately. Calibration is a
model/configuration-level operation, not an automatic per-origin backtest stage.
Earlier continuous calibration artifacts remain plottable without new fields.

Ten new tests check independent atomic-transform uniformity, continuous identity,
label/prefix stability, invalid bounds, categorical CDF mass, mixed-state jumps,
known-parameter exclusion, replay, numerical failures, seeded joint recovery and
CLI behavior. The suite has 343 tests. The independent batch Gaussian/adaptive
reference also now checks the observation-variance CDF directly.

## Initial findings retained for review

Two 200-trial exercises are local under
`runs/joint-calibration-validation-mggbvpmy/`: the four-point joint model (seed
812), and a known-initial-state model with W=0/.2 (seed 813). Both computed every
trial. The first exercise's V CDF distance .1192 exceeded the simultaneous
reference .1126. This finding remains in the record; it is not silently approved
or removed. Three low/middle/high posterior-CDF cases were checked separately
against batch Gaussian likelihoods and adaptive scalar integration, agreeing
within 2.3e-12.

For the mixed-state exercise, all 200 W CDFs and 144 final-state CDFs required
randomization. W closed-interval coverage was 1.000 with mean posterior interval
mass .9986. This illustrates why discrete interval mass must be explicit.
A separately seeded 1000-trial joint follow-up is retained in
`runs/joint-calibration-followup-1000-814/`; its results complement the original
finding rather than replacing it. Human interpretation remains pending.

The independent 1000-trial follow-up (seed 814) completed every fit. V/W/initial/
final CDF distances were .0237/.0159/.0163/.0423 against the simultaneous reference
.0504. Closed-interval coverage was .941/.997/.944/.951; W mean posterior interval
mass was .9961. These results support the calculations while leaving the original
finite-simulation finding visible. All 343 tests passed, and the updated independent
numerical reference tests passed separately.
