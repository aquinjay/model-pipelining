# Workflow review — 2026-11-14

This is the step 1 review of the implementation at `497680f`, before adding
model diagnostics or choosing real data. The documented synthetic workflow
completed successfully in temporary storage, and all 118 existing tests passed.
No blocker was found in the exercised path. This is a computational review;
the runs' human review records remain pending.

Later additions are documented separately in [diagnostics](diagnostics.md) and
[reusable experiments](experiments.md). The scope/results below remain the
review checkpoint at `497680f`.

## Follow one forecast through the pipeline

The review regenerated both seeded 200-day series using CPython 3.14.10. The
following example uses the mean-shift series and the walkthrough's default
settings. The runner does not receive the generator's hidden settings as model
inputs.

| Stage | What happened | Who initiates it |
|---|---|---|
| Prepare | Generate the CSV, register a preserved snapshot, assign `baseline-input`, and resolve it once | Caller, through separate calls |
| Select | Verify the snapshot and select January 1–30 as the first training history | Runner |
| Request | Issue after the January 30 observation; request a forecast for January 31 | Runner |
| Compute | Training mean returns 10.0233; local level returns mean 9.9520 and requested observation uncertainty | Adapter |
| Freeze | Validate outputs and write/close point, quantile, and HPD forecast files | Runner |
| Evaluate | Join the January 31 actual, 10.5683, and calculate errors and applicable uncertainty summaries | Runner |
| Report/review | Draw saved results, compare matching runs, and interpret assumptions/evidence | Caller; scripts automate charting, human interpretation stays separate |

For the next forecast, January 31 becomes eligible past data and February 1 is
the target. Training histories expand from 30 to 199 observations, producing
170 forecasts through July 18. The complete CSV is available to the runner for
validation, but the model request contains only the eligible past. This is an
interface for trusted Python adapters, not a security sandbox.

The Bayesian settings were `initial_mean=0`, `initial_variance=100`,
`observation_variance=1`, and `evolution_variance=0.1`, fixed throughout each run.
The initial prior precedes the first daily transition. Each request refilters
the eligible history from that prior, with no smoothing or retained training
state. The model updates a hidden level; it does not estimate its variance
settings.

For the first forecast, the 95% highest posterior predictive density (HPD)
region is approximately **[7.6578, 12.2462]**. It describes the future observation,
including level uncertainty, one more day's movement, and observation noise.
It is conditional on the fixed settings. HPD and equal-tail bounds coincide
for this normal distribution; the shared pipeline preserves their distinct
meaning for other adapters.

## What the saved checks establish

| Saved check | What a pass means | What it leaves open |
|---|---|---|
| `input_schema_and_daily_calendar` | Required columns, finite numerical values, and ordered daily UTC observations are valid | Whether values were historically available or later revised |
| `eligible_history_and_target_alignment` | Histories and one-day targets obey the implemented synthetic timing rule | Whether that timing rule suits an application |
| `forecast_outputs_valid` | Every expected origin returned acceptable required outputs | Whether the model assumptions are plausible |
| `all_expected_cases_scored` | Point scoring completed for every expected case | Whether errors are acceptable for a use case |
| `requested_observation_quantiles_valid` | All requested quantiles are supplied, finite, and ordered | Predictive calibration |
| `requested_highest_density_regions_valid` | Requested sets have valid segments and nesting | Generic proof of probability mass or density optimality |
| `all_uncertainty_cases_scored` | Every requested uncertainty case was evaluated successfully | Approval based on coverage or other metrics |

Uncertainty checks are required only for declared capabilities. The training
mean has no uncertainty outputs, and its corresponding checks are explicitly
inapplicable. A structurally valid HPD region's mathematical interpretation
remains the adapter's responsibility; the local-level adapter has independent
normal-distribution checks in the test suite.

`manifest.json.status="completed"` means required computational checks passed.
Point `scores.complete` alone is insufficient: uncertainty scoring can fail
independently and fail the run. `review.json.status="pending"` is expected after
successful execution. No automatic assumption approval or release decision
exists. Invalid input/configuration raises before creating a run; forecast
failures are retained per origin and later origins continue.

## Evidence from this review

- Ran the full behavioral suite: **118 tests passed**. This includes independent
  Gaussian-conditioning calculations, future-value isolation, frozen-output
  sequencing, visible failures, and legacy compatibility; see the
  [test map](../tests/README.md).
- Ran both models on both regenerated series with **170/170 point cases scored
  and no failures in each run**. Both local-level runs also scored 170/170
  uncertainty cases and passed all seven required checks.
- Reconstructed each local-level model from its saved configuration and replayed
  the permanent snapshot, task, and interval levels. Point forecasts, quantiles,
  HPD segments, evaluations, scores, and checks matched byte for byte. Manifest
  timestamps are execution metadata and were not compared for equality.
- Drew both runs and compared them on each series while model calls and the
  runner/snapshot retrieval entry points were patched to raise. Reporting
  succeeded and left every existing source artifact unchanged. The comparison
  recorded uncertainty comparison as inapplicable because the baseline has none.
- Exercised the local-level CLI on the mean-shift input. It produced the same
  point forecasts as the library run, a chart, and a separate seeded 200-step
  prior trajectory with pending review. Library runs do not create prior files.

The point-score results reproduce the existing model guide:

| Series | Model | MAE | RMSE |
|---|---|---:|---:|
| Mean shift | Training mean | 2.4197 | 2.9462 |
| Mean shift | Local level | 0.9018 | 1.1744 |
| Constant mean | Training mean | 0.7683 | 0.9638 |
| Constant mean | Local level | 0.8412 | 1.0413 |

The local-level 95% HPD region contained 162/170 actuals (95.29%) on the
mean-shift series and 165/170 (97.06%) on the constant-mean series. These are
descriptive development results, not calibration certificates or acceptance
thresholds. The mean-shift chart shows the local-level forecast adjusting after
the jump; the training mean retains the earlier low values in its expanding
average. The constant-mean results show that adapting more quickly does not
make one model universally preferable.

The preserved input hashes were:

| Dataset | Snapshot ID (SHA-256) |
|---|---|
| `synthetic-mean-shift` | `c7e9fbed4871a4e77166afc8e91b03de38881731bca8a8a370bdc91ebfa5cdf5` |
| `synthetic-constant-mean` | `2510db2af47e4b61c8f356ddc41df922d8992611afaa94cb75def102bec6dc68` |

The store, runs, charts, and replay artifacts were generated under a temporary
directory, outside the repository's existing data/run stores. They are not a
committed evidence archive. This page retains the findings; the
[walkthrough](forecasting.md) explains how to repeat the workflow. Preserving
important complete stores/runs remains an explicit caller action.

## Questions carried into steps 2 and 3

Step 2 is to agree and implement model evidence: sensitivity to the initial
prior and fixed noise settings, fitted-model criticism, how that evidence is
saved, and which failures stop computation versus require human interpretation.
One prior trajectory and correct arithmetic do not answer those questions.
No diagnostics framework or coverage-based approval rule was added in this review.

Step 3 is to choose a real-data question and define the source, target,
availability, revisions, calendar, benchmark, and untouched final evaluation
period. Current runs assume immediate availability with no revisions, and all
forecast dates are development evidence. Code fingerprints do not restore
source/environment, and frozen files are not protected from external edits;
matching code, locked dependencies, preserved inputs, and run evidence must be
retained for reproduction.
