# Forecasting walkthrough

Start here to understand and run the current tooling. This walkthrough uses the
artificial sample data already supplied by the project, runs both implemented
models, and shows what to inspect. Setup is in
[the main README](../README.md#development-environment).

The objective is to verify a reusable workflow: the calculations must be correct,
models must receive only eligible history, and saved inputs/settings must let
us repeat a run. Comparing forecast scores is a separate use of the saved
results, rather than the acceptance criterion for this implementation.

## Understand the example

The input has 200 daily observations with `timestamp` and `value` columns. Each
value is assumed available at its timestamp, with no publication delay or
revision. After observing days 1–30, predict day 31. Then use days 1–31 to predict
day 32, and continue through day 200. This produces **170 forecast cases**.

The two models use the same cases:

| Model | What it calculates | Uncertainty supplied |
|---|---|---|
| `TrainingMean` | Average of all eligible past values | None |
| `LocalLevel` | Updated estimate of a changing hidden level | Observation quantiles and highest-density prediction regions, conditional on fixed model settings |

A **backtest** repeats forecasts at historical dates. An **origin** is when a
forecast is issued. A **case** is one origin and its next-day target. The
**actual** is the observed target value, joined only after predictions are saved.
A **run** is one model/configuration executed over the selected cases. An
**adapter** is the model code that accepts the shared request and returns its
supported outputs.

Each forecast target is held out from that forecast's training history. It can
become an observed past value at a later origin. The whole example is used for
development; it does not reserve an untouched final test period.

## Prepare the input explicitly

Run shell commands from the repository root. If `data/synthetic/` does not already
contain the sample files, generate them:

```bash
.venv/bin/python scripts/generate_synthetic.py
```

The generator refuses to overwrite its files. Reuse existing sample files when
repeating the example, or use `--output-dir` and adjust the input path below.
The [data guide](../data/README.md) describes both scenarios and their settings.

Launch Python:

```bash
PYTHONPATH=src .venv/bin/python
```

Keep this Python session open for the next four sections; the examples pass
variables from one section to the next. You can also place those Python blocks
in one script and launch it with the same interpreter and `PYTHONPATH=src`.

```python
from model_pipelining.snapshots import register_snapshot
from model_pipelining.tags import assign_tag, resolve_tag

snapshot = register_snapshot(
    "data/synthetic/mean_shift.csv",
    dataset="synthetic-mean-shift",
    note="Input for the forecasting walkthrough",
)
assign_tag(
    dataset=snapshot.dataset,
    tag="baseline-input",
    snapshot_id=snapshot.snapshot_id,
    changed_by="walkthrough",
)
selected_snapshot_id = resolve_tag(dataset=snapshot.dataset, tag="baseline-input")
```

A **snapshot** preserves the file bytes. Its permanent reference is the dataset
name plus snapshot ID. A **tag** is a readable name that can be moved explicitly.
We resolve it once here and pass the same permanent ID to both runs, so a later
tag move cannot change which data either model uses.

This example registers an existing completed file directly. A separate ingestion
copy is optional; [the ingestion guide](ingestion.md) explains when and how to
copy a file and save its receipt. Starting a backtest never performs registration
or tag assignment automatically.

## Run the baseline

```python
from model_pipelining.backtest import run_backtest
from model_pipelining.reporting import plot_run

baseline_run = run_backtest(
    dataset=snapshot.dataset,
    snapshot_id=selected_snapshot_id,
    minimum_history=30,
    output_directory="runs/tutorial-mean-001",
)
plot_run(baseline_run)
```

Without a `model` argument, the runner uses `TrainingMean`. It verifies the
preserved file, validates its daily calendar and values, selects each training
history, calls the model, and saves results. `plot_run` is a separate call that
reads those saved results and creates `forecast.png`; it never fits a model.
The baseline supplies a mean only, so it has no uncertainty band.

## Run the Bayesian model

```python
from model_pipelining.local_level import LocalLevel

local_run = run_backtest(
    dataset=snapshot.dataset,
    snapshot_id=selected_snapshot_id,
    minimum_history=30,
    model=LocalLevel(
        initial_mean=0,
        initial_variance=100,
        observation_variance=1,
        evolution_variance=0.1,
    ),
    interval_levels=(0.8, 0.95),
    output_directory="runs/tutorial-local-level-001",
)
plot_run(local_run)
```

These four model settings are explicit illustrative assumptions, held fixed for
the run. The model learns about the hidden level from past values; it does not
estimate those variance settings. The [model guide](local-level.md) explains
the assumptions and equations.

The runner requests the uncertainty kinds that this adapter declares. Its
regions describe the **next observed value**, conditional on the fixed settings.
The chart defaults to the widest supplied highest posterior predictive density
(HPD) region: 95% here. For this Gaussian model, HPD and equal-tail bounds
coincide; the shared format keeps the methods distinct for other models.
See [the uncertainty guide](uncertainty.md) for the precise meaning and outputs.

## Read the results

```python
import json

for run in (baseline_run, local_run):
    manifest = json.loads((run / "manifest.json").read_text())
    scores = json.loads((run / "scores.json").read_text())
    print(run.name, manifest["status"], scores["uncertainty"]["status"])
```

For successful runs, expect `completed / not_applicable` for the baseline and
`completed / completed` for the local-level model. Inspect these files in order:

1. **`manifest.json`** identifies the permanent input, model settings, requested
   outputs, and overall execution status.
2. **`checks.json`** shows whether required validation and scoring checks passed.
3. **`predictions.csv` and `forecast.png`** show what was predicted and when.
4. **`scores.json`** summarizes point errors and applicable uncertainty results,
   with expected/scored/failed case counts.
5. **`review.json`** starts with a pending human decision and stated limitations.

`completed` means the required computational work succeeded; it is not approval
of the model assumptions. `failed` means a required output or evaluation failed.
`running` may indicate an interrupted run. Point `scores.complete` covers point
scoring only: uncertainty can fail while point scoring succeeds. Use the overall
manifest status and checks to assess execution.

Required forecast failures are saved and later origins continue. Point and
uncertainty forecasts are written and closed before actuals are used for scoring.
“Frozen” means the runner/report never rewrites them; the files remain writable
by other programs. Partial scores disclose their case counts and are not a
complete-run comparison.

Human review is a separate action: inspect assumptions and evidence and record
your interpretation/decision in `review.json` if you want to retain it. There is
no review command or automated release policy. A near-nominal coverage rate or
a better point score does not by itself verify an implementation.

## Compare saved results

```python
from model_pipelining.reporting import compare_runs

comparison = compare_runs(
    baseline_run,
    local_run,
    output_directory="runs/tutorial-comparison-001",
)
```

This creates `comparison.json` and `comparison.png`, using only saved results.
The function requires matching permanent inputs, task settings, training
coordinates, target dates, actuals, and complete execution. It does not refit,
tune, or select a winner. The baseline supplies no uncertainty, so its uncertainty
comparison is explicitly inapplicable; the available local-level band can still
be shown. After this section, you can exit the Python session.

## Command-line scripts as an alternative

After explicit input preparation, these scripts combine backtesting and charting:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --output-directory runs/tutorial-script-mean-001

PYTHONPATH=src .venv/bin/python scripts/run_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --interval-levels 0.8 0.95 \
  --output-directory runs/tutorial-script-local-level-001
```

Each script resolves a supplied tag once. For a repeatable selection shared
across separate invocations, pass `--snapshot-id ID` instead of `--tag` using the
same saved ID. The local-level script also simulates one 200-step prior trajectory
before fitting and saves it for human review. The library example above does
not simulate/save priors automatically. [The model guide](local-level.md#run-it)
details this extra behavior. The scripts return a nonzero exit status on failure;
filesystem/reporting errors propagate even if the backtest itself completed.

## Repetition, errors, and reproduction

Use an unused run/comparison directory for every invocation. Existing output
directories and `forecast.png` files are rejected. Repeating registration/tagging
is allowed and records another event; it does not overwrite preserved bytes.

Input validation rejects missing columns, invalid/naive timestamps, unordered or
duplicate dates, daily gaps, missing/nonnumeric/nonfinite values, and insufficient
history. It does not repair data. Invalid input/configuration raises before a run
directory is created. Forecast-stage errors are recorded per case; filesystem
errors can leave an interrupted directory. [The API reference](../src/model_pipelining/README.md)
documents signatures and return values.

For reproduction, use the saved permanent input, model configuration, minimum
history, and requested interval levels, with matching code/environment and a new
output directory. Timestamps will differ; numerical artifacts repeat. Source
fingerprints identify code but do not restore it. There is no automatic model
registry or environment restoration.

Run folders and preserved data are local and ignored by Git. Copying code to
another machine does not copy this evidence. Keep the whole input store and
important run folders together with matching code/locked dependencies.
[The run-file reference](../runs/README.md) explains every saved file.
