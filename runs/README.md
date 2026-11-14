# Local run artifacts

Save local forecasting runs here. Git ignores other contents of this directory by default; this orientation file is tracked. Pushing code does not upload run evidence.

`run_backtest` writes a new, caller-selected directory containing `manifest.json`, `predictions.csv`, `evaluation.csv`, `scores.json`, `checks.json`, and a pending `review.json`. It saves the permanent dataset/snapshot reference, task/model settings, training selections, forecast failures, and score coverage. Predictions are closed before scoring and never rewritten by the runner or report. Files remain writable by other programs.

Call `plot_run(run_directory)` separately to produce `forecast.png` from saved results. The `scripts/run_baseline.py` entry point automatically calls both functions. Existing run directories and charts are rejected. Per-case failures are retained and make the run `failed`; interrupted runs may remain `running`. Review the manifest, required checks, and coverage counts before interpreting scores. Human review is never completed automatically.

Input registration is an earlier explicit action and does not start a run. Snapshots and acquisition receipts stay in the snapshot store, defaulting to `data/versioned/`; the run references the preserved data by dataset and ID. There is no separate ingestion receipt file to save here. Keep the input store, matching code, locked environment, and important run directories for reproduction. Automatic backup, retention, restoration, simulation artifacts, and iteration lineage remain future work.

See [the forecasting walkthrough](../docs/forecasting.md) for commands, exact assumptions, artifacts, and reproduction, and [the data workflow](../src/model_pipelining/README.md#one-complete-workflow) for input preparation.
