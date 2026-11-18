# Verification guide

This page maps implementation claims to tests. Run the suite from the repository
root after [environment setup](../README.md#development-environment):

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

There are currently **174 tests**. They use temporary fixtures/stores rather than
requiring the local example data. The harness is standard-library `unittest`.
NumPy supplies independent batch Gaussian calculations; SciPy supplies independent
numerical integration, and emcee/ArviZ exercise the optional sampler audit. In a
restricted home, use the [writable cache setup](../docs/parameter-uncertainty.md#run-exact-forecasts-or-request-an-audit)
before running sampler tests.

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
| [test_experiments.py](test_experiments.py) | 16 | Named targets, release/revision isolation, actual vintages, available-history counts, disjoint target roles, benchmark freezing/failures, horizons, replay/history fingerprints, legacy version 2, both CLIs |
| [test_normal_mean.py](test_normal_mean.py) | 18 | Hand posterior, independent raw-density/predictive quadrature, HPD mass/density, strong prior influence, numerical extremes, fit identity/freezing/replay/failures, delayed releases/final roles, exact/audit CLI |
| [test_sampling.py](test_sampling.py) | 10 | Independent-chain posterior/predictive agreement, target-density ratios, warmup/acceptance, random-state isolation, seed/replay/order invariance, biased/stuck/missing diagnostics, partial failures, saved-only trace plotting |

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
