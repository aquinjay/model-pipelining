# Saved run files

This directory holds local execution evidence. Git tracks this README but ignores
other contents: pushing code does not upload runs or the preserved data they use.
For a runnable example, follow [the walkthrough](../docs/forecasting.md).

## Read these first

1. `manifest.json`: overall status, permanent input, task, model settings, output
   capabilities, requested levels, scoring conventions, environment, and source
   fingerprints.
2. `checks.json`: required validation/completion outcomes.
3. `predictions.csv` and `forecast.png`: forecasts and their historical alignment.
4. `scores.json`: metrics plus expected/scored/failed **case counts**.
5. `review.json`: pending human interpretation/decision and limitations.

`completed` means the required computation succeeded. `failed` means at least one
required output/evaluation failed. `running` may indicate an interrupted run.
Point `scores.complete` is not overall success: uncertainty can fail separately.
Read the manifest and required checks before interpreting partial metrics.
Empirical interval **coverage** is the fraction of evaluated actuals inside a
region; it is different from completion of the requested cases.

Human review is not completed automatically. The review JSON is a record you can
update manually; no review command or automated release policy exists.

## Files written by the runner

For valid input/configuration, `run_backtest` creates a new, caller-named directory.
It never reuses an existing output directory. New run manifests have format
version 2; snapshot/tag formats remain separate version-1 records in the input
store. Version-1 saved point runs can still be reported without backfilled bands.

| File | When present | Contents |
|---|---|---|
| `manifest.json` | Every started run | Configuration/provenance and execution status |
| `predictions.csv` | After forecast generation | Issuance, training selection, target time, mean, status/error; no actuals |
| `quantiles.csv` | Quantile-capable adapter | Target, probability, value, status/error |
| `highest_density.csv` | Highest-density-capable adapter | Target, mass level, segment index, lower/upper, status/error; disconnected segments stay separate |
| `evaluation.csv` | After point scoring | Actual, prediction, signed error, absolute/squared error, status/failure |
| `interval_evaluation.csv` | Quantile-capable adapter | Equal-tail bounds, actual, membership, width, interval score, endpoint pinball losses |
| `highest_density_evaluation.csv` | Highest-density-capable adapter | Actual, set membership, total segment width, segment count, status/failure |
| `scores.json` | After evaluation | Point metrics/case counts and a separate uncertainty summary |
| `checks.json` | At normal completion, including failed cases | Fixed required/optional check outcomes |
| `review.json` | At normal completion, including failed cases | Pending decision and limitations |

Point, quantile, and highest-density forecast files are written/closed before
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
| `forecast.png` | Explicit `plot_run` call; both run scripts call it automatically | Saved means/actuals and the widest available HPD region by default |
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

Comparisons require complete matching inputs/task/training cases/actuals.
Uncertainty is compared only at mutually supplied kinds and levels. No winner
is selected automatically, and source files are not altered. Paths in the
comparison summary identify local runs; keep those source folders when exporting
important comparisons.

## Reproduction and retention

Keep the entire input store, important run folders, matching source code, and
locked environment. A run saves dataset plus permanent snapshot ID; acquisition
receipts remain in the snapshot store, not as separate receipts in `runs/`.
Source fingerprints identify code but do not retrieve it. [The walkthrough](../docs/forecasting.md#repetition-errors-and-reproduction)
explains replay arguments. Backup, retention, automatic restoration, and iteration
lineage remain caller actions or future work.
