# Saved run files

This directory holds local execution evidence. Git tracks this README but ignores
other contents: pushing code does not upload runs or the preserved data they use.
For a runnable example, follow [the walkthrough](../docs/forecasting.md).

## Read these first

1. `manifest.json`: overall status, permanent input, expanded experiment/task, model settings, output
   capabilities, requested levels, scoring conventions, environment, and source
   fingerprints.
2. `checks.json`: required validation/completion outcomes.
3. `predictions.csv` and `forecast.png`: forecasts and their historical alignment.
4. `scores.json`: metrics plus expected/scored/failed **case counts**.
5. `review.json`: pending human interpretation/decision and limitations.

`completed` means the required computation succeeded. `failed` means at least one
required output/evaluation failed. `running` may indicate an interrupted run.
Point `scores.complete` is not overall success: uncertainty or a declared benchmark can fail separately.
Read the manifest and required checks before interpreting partial metrics.
Empirical interval **coverage** is the fraction of evaluated actuals inside a
region; it is different from completion of the requested cases.

Human review is not completed automatically. The review JSON is a record you can
update manually; no review command or automated release policy exists.

## Files written by the runner

For valid input/configuration, `run_backtest` creates a new, caller-named directory.
It never reuses an existing output directory. New run manifests have format
version 3, including a version-1 experiment specification and its canonical hash.
Snapshot/tag formats remain separate version-1 records in the input store.
Version-1/2 saved runs can still be reported without backfilled experiment claims
or uncertainty. Comparisons require matching metadata, including declared
experiments and history hashes when present; legacy/new metadata may differ.

| File | When present | Contents |
|---|---|---|
| `manifest.json` | Every started run | Configuration/provenance and execution status |
| `case_schedule.csv` | Before any model calls | Candidate targets, issuance, available-history counts, and selection/exclusion reasons |
| `fit_index.csv` | Adapters declaring fit evidence | Every case's fit directory/history hash, acceptance or error; written before actual selection |
| `fits/NNNNNN/fit.json` | Successful evidence construction, including failed computation | Fit format 1; runner-owned identity, posterior, settings, checks, diagnostics, sampler provenance |
| `fits/NNNNNN/draws.csv` | Fits with sampler draws | Chain/draw dimensions, separate warmup, original/transformed parameter, paired future observation, log target, acceptance |
| `predictions.csv` | After forecast generation | Issuance, training selection/end/history hash, target time, mean, status/error; no actuals |
| `benchmark_predictions.csv` | Declared benchmarks | Same cases/histories with benchmark names and frozen point predictions/failures |
| `quantiles.csv` | Quantile-capable adapter | Target, probability, value, status/error |
| `highest_density.csv` | Highest-density-capable adapter | Target, mass level, segment index, lower/upper, status/error; disconnected segments stay separate |
| `evaluation.csv` | After point scoring | Declared actual vintage and its availability time, prediction, signed/absolute/squared error, status/failure |
| `benchmark_evaluation.csv` | Declared benchmarks, after freezing | Same actual vintages and point errors for each benchmark |
| `interval_evaluation.csv` | Quantile-capable adapter | Equal-tail bounds, actual, membership, width, interval score, endpoint pinball losses |
| `highest_density_evaluation.csv` | Highest-density-capable adapter | Actual, set membership, total segment width, segment count, status/failure |
| `scores.json` | After evaluation | Primary point metrics/counts, required benchmark summaries, and a separate uncertainty summary |
| `checks.json` | At normal completion, including failed cases | Fixed required/optional check outcomes |
| `review.json` | At normal completion, including failed cases | Pending decision and limitations |

Fit evidence and point, benchmark, quantile, and highest-density forecast files are written/closed before
actuals are joined for scoring. Failed forecasts retain required placeholder
rows. One case can create several rows: for example, 170 cases at two region
levels produce 340 region-evaluation rows. A disconnected region can require
multiple segment rows in `highest_density.csv`. Those rows are not extra cases.

The runner/report never rewrites frozen forecast files, but files remain writable
by other programs. Invalid input/configuration raises before directory creation.
Filesystem errors can leave partial output with a `running` status. There is no
crash recovery or saved-run integrity audit. [The uncertainty contract](../docs/uncertainty.md)
defines exact region/quantile validation and scoring rules.

## Additional script and report files

| File | Who creates it | Meaning |
|---|---|---|
| `forecast.png` | Explicit `plot_run` call; all three run scripts call it automatically | Saved means/actuals and the widest available HPD region by default |
| `fits/NNNNNN/trace.png` | Explicit `plot_fit` or normal-mean CLI trace flag | Saved audit traces/densities against exact references, with warmup separation |
| `prior_predictive.csv` | Local-level script only | One 200-step simulated latent-level/observation trajectory before conditioning on data |
| `prior_predictive.json` | Local-level script only | Seed, count, Python version, fixed settings, and pending prior review |
| `comparison.json` | Explicit `compare_runs` call, in a new comparison directory | Sources, model settings/scores, common cases, and applicable uncertainty comparisons |
| `comparison.png` | Same comparison call | Saved actuals/means and available regions |

The library runner does not simulate/save priors or draw a chart. The local-level
script simulates before fitting but saves its prior files after the backtest
creates the run directory. Prior plausibility remains a human question. Extra
script/report failures can occur after a completed backtest; the manifest does
not certify all optional artifacts were produced.

Reporting reads saved results only. `plot_run` prefers highest-density regions,
then equal-tail intervals if supplied; point-only runs have no band. An explicit
method/level must have been saved. Existing `forecast.png` files are rejected:
[the uncertainty guide](../docs/uncertainty.md#run-and-inspect) shows how to copy a
run for another chart without refitting.

Comparisons require complete matching inputs/experiments/tasks, training cases
and history hashes, and actual vintages.
Uncertainty is compared only at mutually supplied kinds and levels. No winner
is selected automatically, and source files are not altered. Paths in the
comparison summary identify local runs; keep those source folders when exporting
important comparisons.

`NormalMean` saves exact posterior/predictive evidence for every fit. Its optional
MCMC audit retains completed/partial chains, diagnostics and reference tolerances;
failed required criteria reject that forecast and fail the run. Exact-only fits
have inapplicable MCMC checks and no invented chains. Audited forecasts also use
exact integration. A malformed/missing fit leaves an error in the index rather
than a fabricated fit directory. Persistence errors propagate and can leave a
`running` manifest. See [fit fields, interpretation and replay](../docs/parameter-uncertainty.md).

`NormalMeanVariance` uses the same files and failure rules. Joint-fit metadata
adds normal/inverse-gamma posterior parameters, the exact Student-t law,
nullable undefined variances with explicit reasons, and saved reference
distributions. Draws add observation variance, log variance, its standardized
coordinate and the nonconstant transform Jacobian. Its explicit trace chart
has three rows: mean, observation variance, and future observation. Moment
audits whose variances do not exist are listed with reasons; applicable
diagnostics remain required. See [the joint-model guide](../docs/normal-mean-variance.md).

The [experiment guide](../docs/experiments.md) explains available release
selection, actual-vintage choices, required benchmarks, and separate development/
final invocations. Declaring a final period does not certify that data remained
untouched by humans or other tools. Earlier final observations can enter later
training histories only once available, under a fixed predeclared model/configuration.

## Separate local-level diagnostic folders

Explicit `run_local_level_diagnostics` calls create a different artifact type:
`diagnostics.json` (diagnostic format version 1), `update_paths.csv`,
`scenario_summaries.json`, and pending `review.json`. Explicit plotting adds
`sensitivity.png` and `innovations.png`; `scripts/diagnose_local_level.py` calls
plotting automatically. Backtests and their run scripts do not create these
files. These folders are not inputs to `plot_run` or `compare_runs`.

Paths include initialization and show assumption effects on the updated level,
uncertainty, gains, conditional initial-mean weights, and pre-observation errors.
The reference and every named alternative retain full settings and completion
or numerical failure. A failed reference leaves alternative comparisons
unavailable; no reference path is invented. Strong influence is not a failure
criterion, and review remains pending. See [diagnostics](../docs/diagnostics.md)
for field meanings, reporting, replay, and caller actions. Keep the diagnostic
folder and its whole input store together when retaining this evidence.
This separate workflow processes its complete input snapshot; it does not apply
an experiment's final-period exclusions. Use an appropriate development-only
input when preserving a genuinely untouched final period.

## Reproduction and retention

Keep the entire input store, important run folders, matching source code, and
locked environment. A run saves dataset plus permanent snapshot ID; acquisition
receipts remain in the snapshot store, not as separate receipts in `runs/`.
Source fingerprints identify code but do not retrieve it. [The walkthrough](../docs/forecasting.md#repetition-errors-and-reproduction)
explains replay arguments. Backup, retention, automatic restoration, and iteration
lineage remain caller actions or future work.
