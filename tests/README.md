# Verification guide

This page maps implementation claims to tests. Run the suite from the repository
root after [environment setup](../README.md#development-environment):

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

There are currently **483 tests**. They use temporary fixtures/stores rather than
requiring the local example data. The harness is standard-library `unittest`.
NumPy supplies independent batch Gaussian calculations; SciPy supplies independent
numerical integration, and emcee/ArviZ exercise the optional sampler audit. In a
restricted home, use the [writable cache setup](../docs/parameter-uncertainty.md#run-exact-forecasts-or-request-an-audit)
before running sampler tests.

The [Python 3.12 workplace profile](../docs/python312-public-forecast.md) omits
`emcee`, `arviz` and `openpyxl`. Tests that need them are decorated with
`requires(...)` from [optional_dependencies.py](optional_dependencies.py) and are
skipped in that profile. Set `REQUIRE_OPTIONAL_TEST_DEPENDENCIES=1` to make a
missing package fail instead, as the research CI job does.

## Test map
| File | Count | Main evidence |
|---|---:|---|
| [test_ingest.py](test_ingest.py) | 17 | Exact copies, overwrite/failure behavior, receipts, CSV/XLSX equivalence, worksheets/headers, invalid options |
| [test_snapshots.py](test_snapshots.py) | 27 | Identity/reuse/revisions, separate histories, retrieval/tampering, failure publication, relocation, receipt validation, old records, XLSX preservation |
| [test_tags.py](test_tags.py) | 13 | Explicit movement, dataset scope, permanent IDs, event order, invalid targets, safe publication failure, relocation |
| [test_backtest.py](test_backtest.py) | 21 | Known predictions/scores, eligible immutable history, future isolation, tag pinning/replay, failures, scoring overflow, existing-output protection, reports without fitting |
| [test_local_level.py](test_local_level.py) | 19 | Independent posterior calculations, prior/state simulations, validation/limiting cases, no retained state, shared execution/replay, matched-case reports |
| [test_uncertainty.py](test_uncertainty.py) | 21 | Observation predictive variance, quantile/HPD semantics, known scores, disconnected sets, capability failures, freezing/replay, old-artifact compatibility |
| [test_diagnostics.py](test_diagnostics.py) | 12 | Independent initial-mean influence, strong prior completion, prior/noise distinction, sequential errors, isolation/replay, saved failures, explicit missing correlation, charts without models, CLI inheritance |
| [test_experiments.py](test_experiments.py) | 17 | Named targets, release/revision isolation, actual vintages, available-history counts, disjoint target roles, benchmark validation/freezing/failures, horizons, replay/history fingerprints, legacy version 2, both CLIs |
| [test_normal_mean.py](test_normal_mean.py) | 18 | Hand posterior, independent raw-density/predictive quadrature, HPD mass/density, strong prior influence, numerical extremes, fit identity/freezing/replay/failures, delayed releases/final roles, exact/audit CLI |
| [test_sampling.py](test_sampling.py) | 10 | Independent-chain posterior/predictive agreement, target-density ratios, warmup/acceptance, random-state isolation, seed/replay/order invariance, biased/stuck/missing diagnostics, partial failures, saved-only trace plotting |
| [test_normal_mean_variance.py](test_normal_mean_variance.py) | 20 | Joint posterior/raw-density quadrature, predictive mixture integration and HPD, undefined moments, nonconstant sampler Jacobian, joint/reference agreement, partial/stuck failures, isolation/replay, release/final roles, CLI and saved-only traces |
| [test_predictive_checks.py](test_predictive_checks.py) | 27 | Independent joint/predictive moments and cross-position covariance, statistics/tails, data-free priors, seed/prefix replay, partial failures, as-of releases/final isolation, discrepancy versus execution status, saved-only plots and both CLI modes |
| [test_prior_influence.py](test_prior_influence.py) | 19 | Hand learning/scale decomposition, counterfactual location coefficient, credible/predictive interval mass, variance concentration, persistent influence/undefined moments, common release histories, future/final isolation, reference/update failures, replay, CLI and saved-only charts |
| [test_local_level_variance.py](test_local_level_variance.py) | 18 | Independent batch Gaussian likelihood, adaptive scalar integration, known-level conjugate limit, initial-state smoothing, Gaussian-mixture quantiles/disconnected HPD, broad-tail root precision, failures, horizons, pipeline replay and both CLI modes |
| [test_dynamic_evidence.py](test_dynamic_evidence.py) | 11 | Random-walk covariance, joint initial-state/variance posterior replication with fresh innovations, gaps, replay/RNG isolation, numerical failure evidence, dynamic prior comparisons, undefined moments, final isolation and workflow CLIs |
| [test_evidence_interfaces.py](test_evidence_interfaces.py) | 22 | Provider extension without model dispatch, v1/v2 migration/comparison, caller-required fits/workflows, missing/optional evidence, pinned tags/history/settings, numerical failures, replay/final isolation, saved-only/legacy plots and both new CLI modes |
| [test_sequential.py](test_sequential.py) | 9 | Full predictive CDFs, freezing, release-aware histories, own-target/final isolation, replay, failures and CLI |
| [test_adapters.py](test_adapters.py) | 7 | JSON restoration of every adapter (including the weekly harmonic adapter) including MCMC/quadrature settings, catalog names, scenario overrides that keep unchanged settings, invalid restored settings and calibration parameter validation |
| [test_artifacts.py](test_artifacts.py) | 6 | JSON/CSV writer bytes unchanged, exclusive creation and explicit manifest replacement, no partial file for nonfinite JSON, single-definition benchmark/workflow tables and the immediate observation reader |
| [test_harmonic_regression.py](test_harmonic_regression.py) | 4 | Exact recovery of a noiseless weekly curve, rejection of unidentified phases, fixed configuration and the 72-case example command using the package adapter |
| [test_calibration.py](test_calibration.py) | 10 | Independent CDF references, simulation covariance, recovery, failed trials, seed/prefix isolation and CLI |
| [test_joint_variance.py](test_joint_variance.py) | 14 | Independent batch/adaptive joint posterior references, dominant-component V-quantile bracket repair, limits, uncertainty, confounding, controlled studies and CLI |
| [test_joint_evidence.py](test_joint_evidence.py) | 10 | Joint parameter dependence, fresh paths, atoms, prior mass, support restrictions and historical evidence |
| [test_mixed_calibration.py](test_mixed_calibration.py) | 10 | Randomized transforms, discrete/mixed CDF bounds, interval mass, replay and calibration failure/CLI behavior |
| [test_experiment_review.py](test_experiment_review.py) | 18 | Saved-only review, identities/receipts, scope, missing/failed evidence, version conflicts, fit requirements, complete trial records, paths/concurrency and CLI |
| [test_distribution_scoring.py](test_distribution_scoring.py) | 21 | Independent CDF-integral/density losses, full mixtures/tails/units, precision, all Bayesian adapters, freezing, release/final isolation, replay, partial failures and matched saved comparisons/review |
| [test_documentation_checks.py](test_documentation_checks.py) | 9 | Local links/anchors/references, ignored artifact discovery, Python/Bash syntax without execution, incomplete fences and CLI failure codes |
| [test_model_review_study.py](test_model_review_study.py) | 9 | Original fixture bytes/shared noise, saved-only full-CDF metrics, final exclusion, matched histories/vintages, rejected partial/duplicate HPD and score cohorts, fixed windows, continuation identity/source guards, JSON joint settings and retained failures/partial output |
| [test_innovation_reference.py](test_innovation_reference.py) | 14 | Independent batch/filter conditioning, direct V integrals, exact Student-t/IG and Gaussian limits, label independence, future probability, posterior dependence, atoms, all 256 paths, bounded/invalid input, precision failures, RNG isolation and CLI/replay persistence |
| [test_innovation_particles.py](test_innovation_particles.py) | 16 | Independent dense smoothing, hand optimal label/evidence update, unbiased systematic resampling, ancestry alignment, evidence-weighted joint moments and jackknife MCSE, oracle/limit/atom checks, refinement failures, long history without enumeration, seed/global-RNG isolation, CLI failure persistence and successful replay |
| [test_innovation_probability_bounds.py](test_innovation_probability_bounds.py) | 11 | Dense-likelihood/prior/proposal envelope identity, unseen-event positive uncertainty, vacuous-normalizer failure, retained quiet/gradual oracles, Gaussian limits and atoms, near-dominant components, fixed-seed/global-RNG isolation, bounded long histories, saved-statistic interval replay, validation and CLI input/source freezing/overwrite rejection |
| [test_innovation_lattice.py](test_innovation_lattice.py) | 4 | State-grid likelihoods versus dense Gaussian references, including unlikely observations in Gaussian limits, positive upper bounds for unseen events, exact state atoms, resource failures and replay |
| [test_innovation_gibbs_check.py](test_innovation_gibbs_check.py) | 3 | Banded likelihood and Gaussian state draws against dense calculations, zero-transition atom entry and exit, and independent-seed replay |
| [test_noaa_temperature.py](test_noaa_temperature.py) | 4 | Packaged and frozen calendar sorting, leap day, preserved units, element-specific quality flags, gaps, and duplicate dates |
| [test_noaa_seasonality.py](test_noaa_seasonality.py) | 4 | Known annual-plus-lag prediction, one-day horizon, saved case validation, configuration restoration, and parity with frozen forecasts and summaries |
| [test_noaa_later_period.py](test_noaa_later_period.py) | 2 | Frozen final date boundaries, exact history preservation, and rejection of changed inputs before reading evaluation data |
| [test_noaa_extrema.py](test_noaa_extrema.py) | 2 | Ordered-range scoring with a deliberately crossed pair and rejection of mismatched eligible dates |
| [test_noaa_range.py](test_noaa_range.py) | 4 | Finite residual pair counts, ties and negative ranges, paired Brier scoring, season selection, frozen archive reuse, replay and future actual isolation |

The [Python 3.12 profile](../docs/python312-public-forecast.md) passed 80 targeted
tests covering the Gaussian local-level adapter, backtests, experiment timing,
distribution scores, and the NOAA CSV validation. The public NOAA run is explicit
and uses a caller-acquired CSV. Unit tests use generated input. The
[seasonal and later-year checks](../docs/noaa-later-period.md) and
[two-series checks](../docs/noaa-temperature-pair.md) bring the verified
Python 3.12 subset to 86 tests.

The probability tests verify the finite-sample calculation and its assumptions;
they do not certify floating arithmetic or useful precision on long histories.
The full [17-case probability-bound follow-up](../docs/innovation-probability-bounds.md)
is an explicit caller action outside CI. Its original particle failures and
unresolved broad intervals stay retained separately from passing software tests.

## Data behavior checked in detail

Copy receipts preserve original source/destination, copy completion time, and
byte count. Passing a receipt to registration saves it automatically; omission
and older records return `None` without rewriting history. Tests include source
removal, store movement, differing acquisition/registration times, UTC handling,
wrong destinations/sizes/types, and naive timestamps.

Registration tests preserve revised old values as new snapshots, keep repeated
identical registrations as separate events, detect changed manifests/content,
and handle injected copy/publication failures. Tag tests retain fixed IDs after
a move, keep independent datasets/tags, and preserve saved event order despite
clock changes. These behaviors are detailed in [snapshots and tags](../docs/snapshots.md).

## Forecasting and Bayesian calculations

The baseline checks use known means and errors. Changing future actual values
must not change an earlier forecast. A model receives an immutable eligible
history and target coordinates, not future actuals. Failed origins remain saved
and later origins continue; missing/invalid output cannot silently become success.

The local-level filter is checked against a hand calculation, static normal
conjugacy, and full joint-Gaussian conditioning that does not repeat the scalar
recursion. Seeded simulations check prior moments and hidden-state errors relative
to posterior variances, with broad deterministic tolerances. Configuration and
permanent input references reproduce saved forecasts through the shared runner.
[The model guide](../docs/local-level.md#calculations) explains the arithmetic.

## Uncertainty and reporting

Tests independently check that next-observation variance is `C + W + V`, rather
than the posterior hidden-level variance alone. Normal quantiles/HPD regions are
checked against known bounds, mass, and boundary densities. Shared tests reject
incomplete/crossed/nonfinite quantiles and missing, overlapping, or non-nested
highest-density regions. They check membership and charts against disconnected
sets without treating their gaps as covered.

Known numerical examples verify unscaled pinball and equal-tail interval scores.
HPD evaluation checks separate coverage/total width/segment counts. A numeric
scoring failure can leave valid point scores while correctly failing overall
execution. Frozen forecasts remain unchanged by scoring/reporting; old point-only
artifacts still render without invented uncertainty.

Comparison tests require matching data/cases, reject failed runs and changed
actuals/tasks, preserve source evidence, and prohibit model calls from reporting.
[The uncertainty guide](../docs/uncertainty.md) defines the contract being checked.

## Influence and sequential criticism

The diagnostic tests check initial-mean weights against independent batch
Gaussian conditioning and counterfactual mean changes. A hand calculation
checks update gains, predictive variance, standardized innovations, and PIT.
Changing prior precision and the two noise settings exposes their different
effects on gains/uncertainty; the derivative is not treated as total prior
influence. A known fixed level retains full initial-mean influence and still
completes, with pending human review.

Other checks cover unchanged earlier paths after future-data changes, exact
numerical replay, permanent tag pinning, retained numerical failures and later
successful configurations, missing reference comparisons, unavailable lag
correlation, saved-only plotting, overwrite protection, and CLI scenario
inheritance. These tests verify the [documented evidence](../docs/diagnostics.md),
not that sensitivity or residual patterns meet an approval threshold.

## Reusable task and evaluation contracts

Experiment tests use delayed releases and later revisions as counterexamples to
naive full-file training. They check exact as-of histories, inclusion at the
issuance boundary, unchanged earlier forecasts after future revisions, and
separate first-release/latest scoring without changing predictions. Calendar
gaps, ambiguous releases, invalid boundaries, and conflicting settings raise
before output creation. Available history counts distinct eligible periods.

Target-based development/final cases are disjoint. Changing final-period values
does not change development forecast/evaluation artifacts; earlier final
observations can enter later final histories only sequentially. Known benchmark
scores use identical histories/targets, and all primary/benchmark forecasts are
saved before actual-vintage selection. A benchmark failure fails overall status
even with complete primary scores. Replay restores the saved experiment and
numerical artifacts, while reporting checks history identity and still reads
legacy version-2 artifacts. Both CLIs exercise explicit final selection.
See [experiment semantics](../docs/experiments.md).

## Joint mean and variance learning

The mean/variance tests integrate the raw prior times likelihood independently
of the conjugate update, compare joint-density ratios, and integrate the
Gaussian/inverse-gamma predictive mixture. They verify HPD mass/boundary density
and distinguish Student-t scale from variance. Joint sampler tests subtract
the saved nonconstant Jacobian before comparing target ratios against raw
densities. Exact-reference diagnostics, nullable heavy-tail moments, explicit
audit omissions, RNG/seed isolation, saved failures, frozen fit replay, delayed
releases/final roles, CLI conflicts and saved-only mean/variance traces are
exercised. These are computation checks, not approval of constant-mean behavior.

## Prior and posterior predictive checking

Predictive-check tests compare independent hand-derived prior/posterior moments,
Student-t probabilities and cross-position covariance. The latter rejects
incorrect simulation from independent marginal Student-t values. Hand statistics,
undefined quantities, missing denominators, tail precision, partial simulation/
statistic/posterior errors, prior independence from data, stream-prefix/RNG
isolation, tag pinning, release/final selection, replay, saved-only charts and
both CLI modes are exercised. A conspicuous level shift produces extreme saved
discrepancies while computation completes and human review stays pending.

## Prior influence comparisons

Prior comparison tests check updates and scale components by hand, compare
the location coefficient against an explicit counterfactual, independently
integrate interval masses, and distinguish variance equal-tail intervals from
Student-t HPD regions. They verify concentration changes with fixed prior
expected variance, strong persistent influence completing successfully,
undefined moments, early learning before minimum forecast history, period/
release cutoff separation, delayed revisions, shared predictive-check endpoints,
future/final isolation, order/replay/tag pinning, visible reference/update failures,
saved-only plots and CLI inheritance. These verify descriptive calculations;
they do not decide which prior or amount of influence is appropriate.

## Learned local-level variance

New tests construct independent batch Gaussian likelihoods and conditional
moments, then use adaptive scalar integration as a reference for the production
quadrature. A known constant-level limit has an independently checked
inverse-gamma variance posterior and Student-t forecast. Separate tests check
smoothed initial-state moments, Gaussian-mixture probability and disconnected
HPD mass, refinement/tail failures, gapped timing, replay and legacy CLI behavior.

Dynamic replication tests verify random-walk covariance and fresh zero-mean
state innovations after drawing the joint initial-state/variance posterior.
Reusing a fitted latent trajectory would fail those checks. Workflow tests
cover numerical failure persistence, exact endpoint selection, prior-comparison
semantics, undefined prior moments, source-data isolation, saved-only charts
and both dynamic CLI modes. These are numerical and behavioral checks, not
model approval or a proof of HPD resolution for all pathological mixtures.

## What passing tests does not decide

These checks verify stated calculations and local software behavior. They do
not approve prior plausibility, real-data availability, application usefulness,
formal simulation-based calibration, or release readiness. The seeded tests
are not a full calibration study. Better model scores, faster reactions, and
particular empirical coverage rates are not implementation acceptance thresholds.
A configurable check plan and automatic specialized diagnostic requirements
remain future work; human interpretation remains separate from execution status.
The optional unknown-mean audit does implement explicit saved computation
criteria for its sampler, including reference tolerances based on MCSE. Passing
them does not validate the constant-mean assumption or provide formal simulation-
based calibration. Exact centering/scaling deliberately assists this tractable
calibration exercise; it is not a difficult-posterior benchmark.

## Sequential evidence verification

`test_sequential.py` adds 9 tests for independent mixture/Student-t PIT, frozen distributions, own-target exclusion, replay, final isolation, numerical failures, saved-only plotting, both CLI models and experiment requirements.

## Calibration verification

`test_calibration.py` adds 10 tests for conjugate and known-state references, independent generator covariance, finite-trial CDF/coverage behavior, biased controls, missing parameters, numerical failures, replay/prefix/RNG behavior, CLI and saved-only plotting.

## Joint variance and dynamics verification

`test_joint_variance.py` adds 14 tests: independent batch-likelihood/adaptive quadrature references, W posterior weights, predictive CDF and confounding, scalar limit/HPD, known-state degeneracy, mixture root roundoff, dominant-component V-quantile brackets, numerical failures, configuration, replay/isolation, shared-noise matched studies, CLI and saved-only plots.

## Joint generative verification

`test_joint_evidence.py` adds 10 tests for joint dependence, conditional fresh innovations, prior frequencies/covariance, atomic states, unsupported parameter omissions, matched prior support, future/final isolation, numerical failures, saved-only plots and CLI experiment execution.

## Discrete/mixed calibration verification

`test_mixed_calibration.py` adds 10 tests for atomic-transform uniformity, continuous identity, labelled streams, invalid bounds, discrete/mixed posterior jumps, closed interval mass, known-parameter exclusion, replay, failures, joint calibration smoke checks and CLI behavior.

## Automatic repository checks

[GitHub Actions](../docs/continuous-integration.md) runs the full suite from the
locked environment, checks the interpreter/dependencies/pip export and verifies
documentation. Run the documentation check explicitly from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_documentation.py --walkthrough
```

Local paths/ATX anchors and Python/Bash syntax are checked automatically. The
flag also runs the forecasting guide's Python blocks in temporary storage,
verifying two 170-case runs and comparison/charts; other examples are parsed.
Large calibration studies stay outside CI and require explicit execution.
