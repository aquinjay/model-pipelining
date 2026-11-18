# Simulation calibration and parameter recovery

This data-free workflow checks posterior inference over independently simulated
prior/data draws. It complements fixed numerical examples and predictive model
criticism. The generative model, rather than an external dataset, supplies truth.

The original continuous-model implementation uses the posterior-CDF form: draw a
parameter and dataset from the prior/model, fit the dataset, then evaluate each
unknown parameter's marginal posterior CDF at its simulated truth. Correct
continuous inference produces uniform CDF values over repeated prior/data trials.
This follows the calibration identity underlying [simulation-based calibration](https://arxiv.org/abs/1804.06788)
and [the Stan explanation](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html).
It evaluates exact/numerical CDFs directly; it does not manufacture sample ranks
or audit an MCMC sampler. Marginal uniformity does not establish every joint
posterior dependence or adequacy for external data.

`NormalMeanVariance` calibrates mean and observation variance.
`LocalLevelVariance` calibrates observation variance, unknown initial level and
unknown last level. Known fixed states are excluded explicitly. Each record also
saves a posterior median and equal-tail credible interval for parameter recovery.
Coverage averages over the prior; each fixed true parameter need not have the
same coverage. Weakly identified parameters can retain wide intervals.

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_model.py \
  --model local-level-variance \
  --config configs/local_level_variance_priors.json \
  --observation-count 20 --trials 200 --seed 711 \
  --output-directory runs/calibration-example-001
```

The caller selects configuration, sample size, trial count and seed. No snapshot,
tag or final evaluation data is used. The command saves calculations then draws
`calibration.png` automatically. Library callers invoke
`calibration.run_calibration(model=..., observation_count=..., output_directory=...,
trials=200, seed=0, interval_level=.95)` and `plot_calibration(directory)` separately.
This potentially expensive model/configuration-level exercise is not started by
a backtest or an experiment's historical evidence runner.

## Evidence and computation status

- `calibration.json`: settings, continuous unknown parameters, source/environment,
  seed derivation and computational status.
- `trials/NNNNNN.json`: simulated truth/data, frozen before fitting.
- `calibration.csv`: every trial/parameter's CDF, median, interval, truth or failure.
- `inference.json`: successful numerical evidence and failed precision details.
- `summary.json`: CDF means/distances, uniform reference bands, credible coverage
  with independent-trial Monte Carlo error/Wilson intervals, and recovery errors.
- `checks.json` and `review.json`: computation requirements and pending interpretation.

Each trial uses `PCG64(SeedSequence([seed,2,trial_index]))`. Extending trial count
preserves earlier draws and fits; global RNG states are untouched. Numerical
integration precision is separate from finite-trial simulation uncertainty.

The simultaneous 95% uniform-CDF reference uses the Dvoretzky–Kiefer–Wolfowitz
bound and a union bound over K marginal parameters:
`epsilon = sqrt(log(2*K/.05)/(2*N))`. These reference bands apply to the full
simulation only when every trial completes. Failed fits remain in counts and
can induce selection bias; summaries conditional on successful fits cannot
certify unconditional calibration. No uniformity threshold automatically approves
or rejects a model. `completed` means every requested calculation succeeded.
I/O propagates and existing destinations are never reused. Plots read saved
records without fitting or simulation.

## Verification and initial exercise

Ten tests cover independent conjugate CDF/interval references, the known-state
inverse-gamma limit, simulation covariance, replay/prefix/RNG behavior, biased-CDF
controls, omitted parameters, precision failures, both CLI modes and saved-only
plotting. This slice brought the suite to 310 tests; see [the test guide](../tests/README.md) for the current total.

The initial exercise used 200 trials of 20 observations for each model, seed 711,
with no failed trials. Artifacts are local in
`runs/calibration-validation-lkny0va7/`. Constant-model mean/variance 95% interval
coverage was .955/.970; dynamic observation variance/initial level/last level
coverage was .950/.955/.915. The last value merits interpretation with its saved
Monte Carlo uncertainty rather than an automatic pass. All marginal CDF distances
were below their simultaneous uniform reference bands. These are finite
simulation observations, not model approval or proof of joint correctness.

A separately seeded follow-up used 1000 dynamic trials (seed 712), retained under
`runs/calibration-followup-1000-712/`. All completed. Variance/initial/final-state
coverage was .939/.958/.954, with Wilson intervals containing .95. CDF distances
were .0253/.0271/.0318 against the simultaneous reference .0489. This follow-up
adds evidence; it does not replace the original 200-trial findings.

[Discrete/mixed calibration](mixed-calibration.md) now extends this workflow to joint variance learning. Additional fields preserve CDF jumps, randomization and actual closed-interval mass; continuous results retain their original meaning.

## Link calibration to historical review

[Saved experiment review](experiment-review.md) accepts explicit original and
follow-up calibration folders with the same expanded model settings. It retains
separate seeds, summaries and trial records; it does not pool trials, rerun them or
claim calibration at every historical sample size. Requiring calibration is an
explicit report option.
