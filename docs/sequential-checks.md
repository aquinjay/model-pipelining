# Sequential predictive diagnostics

`run_sequential_checks` extends historical prediction criticism to models with
learned observation variance. Current providers support `NormalMeanVariance`
(exact mode) and `LocalLevelVariance`. The older fixed-variance diagnostic
command remains available unchanged.

Each forecast uses only releases eligible at its historical issuance. The runner
writes every predictive law to `predictive_distributions.json` before selecting
actuals for scoring. Actual releases must be available by the explicit development
cutoff; final targets and later releases are excluded. Scoring follows the task's
first/latest-eligible vintage policy. Cases use the experiment's minimum history,
horizon and issuance delay. This is development evidence, including overlapping
multi-step forecasts when configured.

For each target, `sequential.csv` saves the prediction error, predictive variance,
standardized error and probability integral transform (PIT). PIT evaluates the
full predictive CDF at the actual observation: a Gaussian mixture for the learned
local-level model, or an exact Student-t for the constant-mean/variance model.
Standardizing mixture errors does not make them Gaussian. Undefined predictive
variance leaves standardization null with a reason; PIT remains available.

`summary.json` records error location, squared standardized errors, PIT tail
counts and residual correlations at up to 20 case lags. Correlations use valid
pairs without joining gaps across failed cases; insufficient/constant pairs give
null. Patterns are descriptive, with no automatic acceptance threshold. Human
review stays pending. Numerical failures and precision evidence remain visible,
and later origins continue. I/O errors propagate; destinations must be unused.

```bash
PYTHONPATH=src .venv/bin/python scripts/check_sequential.py \
  --model local-level-variance \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59+00:00 \
  --experiment configs/daily_forecast_experiment.json \
  --config configs/local_level_variance_priors.json \
  --output-directory runs/sequential-example-001
```

Prepare/register the input and tag first as in [the walkthrough](forecasting.md).
The command uses the config's reference settings, saves checks and automatically
draws `sequential.png`. Library callers invoke `plot_sequential_checks` separately.
The chart reads saved errors/PIT/correlations only, without fitting or input access.

Add `sequential_checks` to `EvidencePlan.required_workflows` to require this
calculation through [the explicit evidence runner](evidence.md). Separate evidence
is never started automatically by a backtest. The common provider method
`predictive(request)` supplies a serializable observation law and inference
metadata; no actual target value is passed to it.

Independent tests check mixture and Student-t CDFs, prediction freezing, exclusion
of a target from its own fit, replay, final/future isolation, retained precision
failures, both CLI models and required workflow execution. These checks support
calculation and information boundaries; they do not establish model suitability.

The initial exercise completed 130 cases on each synthetic history, with artifacts
under `runs/sequential-validation-1suduhmc/`. Mean squared standardized error was
1.131 on the stable fixture and 1.384 on the abrupt-shift fixture. These are
descriptive development findings, with human review pending. All 300 tests passed.
