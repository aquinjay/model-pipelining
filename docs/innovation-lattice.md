# Probability bounds from an observation-conditioned state grid

The follow-up uses an evenly spaced grid for the hidden level to propose plausible
observation variances and sequences of small or large innovations. The proposal
comes with a likelihood envelope, a bound on the error from the state grid, and a
bound on the contribution from omitted observation-variance tails. The computed
probability intervals include all three allowances. A path with zero sample
visits still has a positive upper endpoint.

The pilot used the original 262,144 draws per case and the original confidence
allocation across 17 cases. The widest interval fell from 0.9300 to 0.01026 on
`short-abrupt`, from 1 to 0.01030 on `short-dominant`, and from 1 to 0.03276 on
`abrupt-42`. All three met the existing criteria of a positive lower bound for
the normalizer and maximum interval width of 0.10. The two short reference
comparisons covered all 192 probability probes.

## Run the pilot or full study

Use the retained fixed-budget study or reproduce it from the parent particle
study. The parent provides the fixed model, histories, query grids, seed, draw
count, confidence allocation, and comparisons. See the [parent study
guide](innovation-probability-bounds.md#explicit-caller-actions).

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/study_innovation_lattice.py \
  --source-directory runs/innovation-probability-bounds-checked-20261119 \
  --output-directory runs/innovation-lattice-pilot-new
```

Choose a new output directory. The default runs the three pilot cases. Pass
`--all-cases` to run the full frozen design. The optional `--samples` changes the
draw count for a separate run. The command snapshots the inputs and sources
before drawing, records each case and its comparisons, and returns exit code 1
if a case is unresolved or fails an oracle comparison. Case failures are
retained in the output directory. These studies run when invoked explicitly.

For individual calls, import `bound_innovation_probabilities_lattice` from
`model_pipelining.innovation_lattice`. Its keyword arguments are `model`,
`settings`, `predictive_points`, `state_points`, and `variance_points`, with the
same defaults and settings record as the original bound. An empty history
uses the original prior-only probability calculation. The returned status
reflects the probability checks. Invalid inputs and unsupported resource
requirements raise `ValueError`. The current grid limits are 12,000 state
points, 1,000 observation-precision cells and an estimated 80 billion direct
convolution products per knot pass. Runtime also depends on the draw count.

## The envelope

Write `s = 1/V`, `a` and `b` for the inverse-gamma prior parameters, and `n`
for the history length. Combining `s**(n/2)` from the measurement likelihood
with the prior gives a gamma density `g(s)` with shape `a+n/2` and rate `b`,
times the constant `C` from the original probability bound. Conditional on a
latent state trajectory, the remaining observation factor is
`exp(-s * RSS / 2)`, where RSS is the sum of squared differences between the
trajectory and observations.

On each precision interval `[l,r]`, replace `g(s)` by its maximum over that
interval and `exp(-s * RSS / 2)` by the straight line connecting its endpoint
values. Convexity gives an upper bound for every trajectory. On a finite state
grid, the mass of this envelope is a sum of forward likelihoods at precision
knots. Each interval endpoint contributes its grid likelihood times half the
interval width and the gamma maximum. The sampler selects an interval and
endpoint by these masses, draws a precision from the corresponding triangular
density, then draws a grid trajectory and labels from the endpoint's forward
filter by backward sampling. The draw's weight is the actual gamma density
times observation factor divided by their envelope. It lies in `[0,1]` in
exact arithmetic. The grid transition uses the original probabilities for the
two positive-density Gaussian branches and identity moves for zero innovation
variance.

Each draw uses the exact conditional Gaussian smoother for its sampled `V`
and label path to integrate the initial and ending states and both future
innovation branches. The grid trajectory is auxiliary. In particular, the
strict and inclusive state CDFs use the true conditional Gaussian law and
preserve genuine atoms.

## State-grid and precision-tail allowances

For a fixed label path and `s`, the joint state density given the observations
is Gaussian on its free coordinates. A zero innovation variance identifies
neighboring coordinates and a known initial state fixes its coordinate. The
grid includes the known-state point. The code bounds the largest eigenvalue
`P` of the conditional precision matrix using the transition edge weights,
initial prior precision and observation precision. Positive-variance edges add
a weighted path Laplacian whose norm is bounded by `4/W_min`. The
observation precision per coordinate is at most `s`, or `n*s` if zero
innovations allow collapsed coordinates. The lattice spacing `h` is set using
this uniform bound.

For `d` free coordinates, Poisson summation gives a bound on the relative error
of summing the Gaussian density on the infinite grid. With
`u = exp(-2*pi**2/(h**2 * P))`, the absolute error is at most
`(1 + 2*u/(1-u))**d - 1`. This follows by taking absolute values of the
nonzero Fourier terms and bounding the one-dimensional squared-index series
by a geometric series. The identity also applies after shifting a Gaussian's
mean, which is used to control the finite-grid truncation. See [Micciancio and
Regev, Lemma 2.8](https://cims.nyu.edu/~regev/papers/average.pdf).

Conditional means lie in the hull of the observations and the initial prior
mean. Observed coordinates have conditional variance at most `V`, and an
unobserved initial coordinate has variance at most its prior variance.
Exponential tilting in each coordinate and a union bound over grid faces
control the removed grid mass. The resulting multiplicative bounds
`lattice_mass_ratio` apply separately to every label path and retained
precision, including unvisited paths.

Outside the retained precision range, the observation factor is at most one.
The omitted continuous evidence is bounded by `C` times a gamma tail bound
from exponential Markov inequalities. The range is chosen using a rough
single-path evidence scale as a hint. The actual tail bound is calculated
from the resulting limits and does not rely on the hint's accuracy.

The same simultaneous empirical Bernstein rule used for the prior proposal
bounds all weighted probability numerators and the normalizer under independent
sampling. The code divides these expectation bounds by the appropriate grid
mass-ratio endpoints and allows all omitted tail mass to fall inside or outside
each event. This gives intervals for the continuous model at the requested
probability probes. See [Maurer and Pontil, Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).

These guarantees assume exact arithmetic. The implementation uses direct
nonnegative convolution and log-domain normalization. An earlier FFT version
was rejected after a Gaussian-limit check found that roundoff could invent
mass for very unlikely transitions. Density and moment estimates are saved
for comparison, including state and variance/path dependence. They remain
descriptive because the probability envelope covers bounded event functions.
The study retains the original particle results as empirical comparisons for
long histories.

## Saved evidence

The direct-convolution pilot is saved in
`runs/innovation-lattice-pilot-direct-20261119/`. The exploratory FFT pilot is
retained in `runs/innovation-lattice-pilot-20261119/`. Its Gaussian-limit
regression prompted the direct-convolution rerun before expanding to all cases.

The full run at `runs/innovation-lattice-full-20261119/` passed all 17 cases at
the same budget. The widest interval on any short case was 0.01033. Across the
six 160-observation histories, the widest interval on each case ranged from
0.03268 to 0.03379, compared with 1.0 in the prior proposal. The largest scaled
point-estimate difference from the saved particle posterior across those long
histories was 0.01434. That comparison includes moments and density probes.

| Path | Oracle probability | Upper endpoint | Draws that visited |
|---|---:|---:|---:|
| Quiet `011100` | 5.027e-6 | 1.418e-4 | 2 |
| Quiet `101001` | 6.258e-6 | 1.300e-4 | 1 |
| Gradual `000111` | 2.175e-5 | 1.696e-4 | 5 |

Each of these three intervals starts at zero. The audit recalculated all 17
sets of intervals, verified 48 input hashes and 12 source hashes, and matched
all three pilot case files byte for byte. All 872 short-oracle probability
checks passed at the retained comparator tolerance. The only differences
without that tolerance were 14 oracle CDF values that rounded above 1 by at
most 4.45e-16. All 446 repository tests passed. See the [curated study
summary](artifacts/innovation-lattice-summary.json).

The subsequent continuous-state Gibbs audit checks moments and predictive
density on `abrupt-42` and records repeated-fit cost and failure handling.
See the [independent verification](innovation-moment-check.md).
