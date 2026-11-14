# Verification guide

This page maps implementation claims to tests. Run the suite from the repository
root after [environment setup](../README.md#development-environment):

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

There are currently **118 tests**. They use temporary fixtures/stores rather than
requiring the local example data. The harness is standard-library `unittest`.
NumPy, already installed through pandas, supplies an independent batch Gaussian
calculation for the model tests.

## Test map

| File | Count | Main evidence |
|---|---:|---|
| [test_ingest.py](test_ingest.py) | 17 | Exact copies, overwrite/failure behavior, receipts, CSV/XLSX equivalence, worksheets/headers, invalid options |
| [test_snapshots.py](test_snapshots.py) | 27 | Identity/reuse/revisions, separate histories, retrieval/tampering, failure publication, relocation, receipt validation, old records, XLSX preservation |
| [test_tags.py](test_tags.py) | 13 | Explicit movement, dataset scope, permanent IDs, event order, invalid targets, safe publication failure, relocation |
| [test_backtest.py](test_backtest.py) | 21 | Known predictions/scores, eligible immutable history, future isolation, tag pinning/replay, failures, scoring overflow, existing-output protection, reports without fitting |
| [test_local_level.py](test_local_level.py) | 19 | Independent posterior calculations, prior/state simulations, validation/limiting cases, no retained state, shared execution/replay, matched-case reports |
| [test_uncertainty.py](test_uncertainty.py) | 21 | Observation predictive variance, quantile/HPD semantics, known scores, disconnected sets, capability failures, freezing/replay, old-artifact compatibility |

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

## What passing tests does not decide

These checks verify stated calculations and local software behavior. They do
not approve prior plausibility, real-data availability, application usefulness,
formal simulation-based calibration, or release readiness. The seeded tests
are not a full calibration study. Better model scores, faster reactions, and
particular empirical coverage rates are not implementation acceptance thresholds.
A configurable check plan and automatic specialized diagnostic requirements
remain future work; human interpretation remains separate from execution status.
