# Independent verification of innovation moments and predictive density

A continuous-state Gibbs sampler agrees with the state-grid estimates on the
frozen 160-observation history `abrupt-42`. The comparison covers initial and
final state moments, covariance between observation variance and jump count,
and the next-observation density and CDF at the saved points. Two independent
Gibbs budgets and two new state-grid seeds passed the declared comparison.
The original state-grid probability intervals and original rare-path failures
remain part of the evidence.

The state-grid candidate takes about 16 seconds per fit on this history. At
that cost, 130 equally sized fits would take about 35 minutes on one thread.
Use the existing Gaussian local-level adapters for the first BDLM workflows.
Keep the rare-jump approximation as a research option and revisit integration
when a concrete task needs it and can accommodate its measured cost and
predictive output requirements.

## Run the verification

Use the completed [state-grid study](innovation-lattice.md#saved-evidence)
as the source. Invoke the check with a new output directory.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/check_innovation_moments.py \
  --source-directory runs/innovation-lattice-full-20261119 \
  --output-directory runs/innovation-moment-check-new
```

The command freezes the inputs, source files, budgets, and thresholds and saves
each case even when a check fails. It runs the six oracle controls, both
independent four-chain budgets, two full-budget state-grid fits, and a
deliberately undersized grid fit. Exit code 1 indicates an unresolved check.
Use a separate output directory and `--refine-control short-abrupt` to repeat
that control at the larger budget. This leaves the initial result available.
The study runs when invoked explicitly. CI exercises smaller independent tests.

## Independent calculation

The Gibbs chain alternates between the hidden path, observation variance, and
jump labels. The Gaussian conditional on variance and labels has a tridiagonal
precision matrix. Factor it as `Q = L D Lᵀ`, draw a perturbation with covariance
`Q`, and solve the precision system. This draws the continuous path without a
state grid, particle resampling, or the earlier conditional smoother. Given
the path, the observation variance has an inverse-gamma conditional and
positive-variance labels update by their Gaussian likelihoods for each
increment. For short histories mixing zero and positive transition variance,
draw all labels jointly from their marginal likelihood conditional on the
observation variance and then redraw the state. This block allows the chain
to enter and leave the known-state atoms. The exact blocked update is limited
to eight observations. The short controls check both atoms and the Gaussian
limits of zero or certain jumps and equal innovation variances.

Each draw contributes centered state moments, the centered product for
variance and jump count, and the two future Gaussian innovation branches.
The small and large future innovation branches keep their original probabilities
at each draw. Constant sampled indicators keep an unavailable MCSE marker.
That distinction matters when an event may have positive mass and remains
unvisited. Compare strict and inclusive CDFs in the oracle atom control.
Rank-normalized split R-hat and bulk effective sample sizes follow
[Vehtari et al.](https://arxiv.org/abs/1903.08008).

## Results and repeated-fit cost

The held history used four chains with 8,192 draws each and four new chains
with 16,384 draws each after warmup. A comparison passes when the absolute
difference plus three Gibbs MCSEs is within its frozen tolerance. Examples
are 0.05 for state means and variances, 0.08 for variance and jump-count
covariance, 0.010 for density, and 0.025 for CDF probes. Chain checks require
R-hat at most 1.01 and bulk ESS at least 400 for estimates with diagnostics.
The original `short-abrupt` control was too imprecise at 8,192 draws per chain.
A separate four-chain run at 65,536 draws per chain passed the same criteria.
Five other short controls passed at the original budget. This is an empirical
cross-check at the recorded probes. References on the long history still
include Monte Carlo error.

| Quantity | Refined Gibbs estimate | Absolute difference from original grid |
|---|---:|---:|
| Initial state mean | 9.95590 | 0.00057 |
| Final state mean | 15.27266 | 0.00188 |
| Observation variance mean | 0.84868 | 0.00090 |
| Covariance of variance and jump count | -0.00529 | 0.00040 |
| Predictive variance | 1.18262 | 0.00200 |
| Predictive density at 15 | 0.38051 | 0.00018 |

Across the three grid seeds the largest density difference from the refined
Gibbs result was 0.00027. On the same history the two measured full-budget grid
fits took 16.52 and 16.27 seconds. The two Gibbs budgets took 2.75 and 5.49
seconds before diagnostics and trace writing. The 32-draw grid control
returned `unresolved` and retained the result. These timings describe this
machine and fixture. The projection above assumes the same history length
and fit budget. It is not a measured end-to-end backtest.

The [curated summary](artifacts/innovation-moment-check-summary.json) links the
local run and its control refinement. The original main run exited with status
1 because of the short-control precision failure. The follow-up checks that
same frozen history and thresholds separately. The refined long run and the
control refinement both replayed their saved state, variance, and label draws
exactly. All 449 repository tests and the documentation walkthrough passed.
