# First forecasting workflow

Implemented: 2026-11-13. This is a small development exercise using the existing
synthetic data, with a training-mean model and shared execution, scoring, and
saved results. There is no Bayesian model or general experiment framework yet.

## What the first model does

After observing the first 30 days, take their arithmetic mean and predict day 31.
Then use days 1–31 to predict day 32, continuing through the last observation.
This is an expanding-history, rolling-origin backtest: each historical prediction
uses only the observations available at its issuance time.

The task is deliberately narrow:

| Choice | Behavior |
|---|---|
| Input | One preserved CSV with `timestamp` and `value` columns; other columns are ignored |
| Calendar | Exactly one UTC day between consecutive observations, including weekends |
| Availability | Each value is available at its timestamp; no delay or revisions |
| Issuance | Immediately after observing the current value |
| Target | The next day's value, in the original units |
| Initial history | 30 observations by default; caller can set `minimum_history` |
| Training window | All observations up to and including issuance |
| Model output | One finite point forecast representing a predictive mean |
| Evaluation | All remaining days; MAE and RMSE, with explicit coverage counts |
| Role | Development evidence; no untouched final evaluation period |

Availability is an assumption of these synthetic datasets. File acquisition and
registration timestamps do not establish historical availability for real data.
The generator's known means and change date are simulation truth, never model inputs.

## Prepare the input explicitly

Install dependencies using the [project setup](../README.md#development-environment).
matplotlib is now included for a saved PNG chart. If the synthetic files are absent,
generate them from the repository root:

```bash
.venv/bin/python scripts/generate_synthetic.py
```

The generator refuses to overwrite existing files. Use `--output-dir` to generate
another copy instead. Then launch Python with `PYTHONPATH=src .venv/bin/python`:

```python
from model_pipelining.snapshots import register_snapshot
from model_pipelining.tags import assign_tag

snapshot = register_snapshot(
    "data/synthetic/mean_shift.csv",
    dataset="synthetic-mean-shift",
    note="Input for the first forecasting exercise",
)
assign_tag(
    dataset=snapshot.dataset,
    tag="baseline-input",
    snapshot_id=snapshot.snapshot_id,
    changed_by="manual-review",
)
```

Registration can take an existing completed file directly. If you need a separate
ingestion copy and its saved receipt, use the [copy-to-tag workflow](../src/model_pipelining/README.md#one-complete-workflow).
If your chosen snapshot already has a tag, reuse it and skip these preparation
calls. Running a backtest never acquires, registers, or tags data automatically.

## Run the baseline

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --dataset synthetic-mean-shift \
  --tag baseline-input \
  --output-directory runs/mean-shift-001
```

The script automatically runs the backtest, saves its artifacts, draws
`forecast.png` from those saved results, and prints the scores. It exits with
status 1 if any forecast/evaluation case failed, while retaining the evidence.
Use a new output directory for each run; existing directories are rejected.
Use `--snapshot-id ID` instead of `--tag` to select a permanent reference directly.
`--store` selects another snapshot store and `--minimum-history` changes the initial history.

The library keeps running and reporting as separate caller actions:

```python
from model_pipelining.backtest import run_backtest
from model_pipelining.reporting import plot_run

run = run_backtest(
    dataset="synthetic-mean-shift",
    tag="baseline-input",
    minimum_history=30,
    output_directory="runs/mean-shift-library-001",
)
chart = plot_run(run)
```

`run_backtest` returns an absolute `Path`, including when individual cases failed.
Inspect `manifest.json` and `checks.json` before treating its results as successful.
`plot_run` needs only saved artifacts, never the original input or a model call.
It refuses an unfinished run or an existing chart. No GUI is required.

## What is saved

| File | Contents |
|---|---|
| `manifest.json` | Permanent dataset/snapshot reference, selected tag as context, task, model settings, Python/pandas versions, implementation fingerprints, start/finish times and status |
| `predictions.csv` | Issuance, training start/cutoff/count, target time, prediction, and per-origin failure details; no actual targets |
| `evaluation.csv` | Predictions joined to actuals by target time, signed error (`prediction - actual`), absolute/squared errors, and failures |
| `scores.json` | MAE, RMSE, expected/scored/failed counts, and whether all cases were scored |
| `checks.json` | Required input/calendar, eligible-history/alignment, finite-output, and scoring-coverage outcomes |
| `review.json` | Pending human review and limitations; successful execution never approves a model |
| `forecast.png` | Optional chart written by the reporting call; actuals versus predictions with status and coverage |

Tag selection resolves once before execution; the run saves the resolved ID. Tag
moves do not change a saved run. Input validation rejects missing columns, invalid
or naive timestamps, duplicates, unordered dates, daily gaps, nonnumeric/missing/
nonfinite values, or insufficient history. It does not repair data. Validation and
configuration failures raise before creating a run directory; they are not saved
as completed runs.

The runner passes an immutable history and target time to the model. Predictions
are written and closed before actuals are joined for scoring. A model exception
or invalid output records a failed origin and does not stop later origins.
Scoring overflow is also recorded. Scores use successful cases only, with `null`
metrics when none succeeded. Partial scores must not be compared as complete runs.
Any failed case makes the run status `failed` and a required check fail.

These are a fixed set of checks owned by this runner. A configurable check plan,
adapter diagnostics, and release policy remain future work; a model cannot omit
one of the current checks. The Python model boundary assumes trusted code and
does not prevent a model from independently reading files.

To reproduce a baseline run, read `input.dataset`, `input.snapshot_id`, and
`task.minimum_history` from its manifest and supply them to another
`run_backtest` call with a new output directory. Keep the preserved data, matching
source code, and locked environment; source fingerprints identify code but do not
save or retrieve it. Baseline forecasts and scores repeat; run timestamps differ.
There is no automatic code/environment restoration or general model registry.

Run directories and snapshots remain local and ignored by Git. The runner never
rewrites saved forecasts, but files are still writable by other programs. Use one
writer. Filesystem errors propagate and interrupted runs may retain `running`
status; crash recovery and artifact integrity verification are not implemented.

## Where to read the implementation

- [models.py](../src/model_pipelining/models.py): the request, point-forecast interface, and arithmetic-mean model.
- [backtest.py](../src/model_pipelining/backtest.py): data validation, historical selection, frozen predictions, scoring, and run files.
- [reporting.py](../src/model_pipelining/reporting.py): charting saved results.
- [test_backtest.py](../tests/test_backtest.py): known answers, future-value isolation, failure visibility, reproduction, and reporting checks.

Review this working path before extending the interface for Bayesian models,
probabilistic outputs, features, tuning, multiple horizons, or real-world timing.
