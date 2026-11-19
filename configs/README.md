# Configuration

The [innovation moment check](../docs/innovation-moment-check.md) inherits the
observations, model and query grids from a completed state-grid study. The
caller selects a source and new output directory. The script freezes Gibbs
budgets, seeds and comparison thresholds in the output design. A separate
`--refine-control` invocation repeats a named control with longer chains.

The [state-grid study](../docs/innovation-lattice.md) inherits the frozen model,
queries, seed, confidence and draw budget from a saved probability-bound study.
The caller supplies a parent `--source-directory` and new `--output-directory`.
The command runs the three pilot cases by default. Select `--all-cases` for the
full design or `--samples` for a separate draw budget.

The [probability-bound follow-up](../docs/innovation-probability-bounds.md)
reuses a saved particle study's frozen design and original failures. It has no
separate JSON configuration: callers explicitly choose `--source-directory`, a
new `--output-directory`, and fixed `--samples`, `--seed`, `--confidence` flags on
`scripts/check_innovation_probability_bounds.py`. The command automatically
freezes the inputs and source files and divides confidence error across the
declared cases/probes; it never changes the parent particle configuration.

[innovation_particles.json](innovation_particles.json) separately declares the
particle numerical exercise: fixed model, particle/replicate/seed/refinement
settings, finite query grids, explicit short oracle cases and long synthetic
generator settings. Run `scripts/run_innovation_particles.py --config configs/innovation_particles.json --output-directory runs/particles-new`
explicitly; omission of `--config` chooses this default file. The command freezes
every generated value before fitting, saves all attempts/oracle comparisons and
automatically plots the saved results. Unresolved cases produce exit 1 without
discarding evidence. See [method, limits and results](../docs/innovation-particles.md).

[innovation_reference.json](innovation_reference.json) is a separate bounded
numerical exercise: fixed transition settings, unknown-V prior, precision
settings, named short histories and predictive/state/V query grids. It is not
an experiment/backtest configuration. Explicitly run
`scripts/run_innovation_reference.py --config configs/innovation_reference.json --output-directory runs/reference-new`;
omitting `--config` selects this fixed default file. The command freezes expanded
inputs before calculating, runs both independent references and Gaussian limits,
and plots saved results automatically. See [scope and results](../docs/innovation-reference.md).

The backtest takes an optional versioned experiment through Python arguments or
`--experiment FILE` in its four run scripts. The separate diagnostic command
accepts an explicitly selected JSON file via `--config`. These are different
schemas. No command discovers files from this directory automatically.

## Experiment example

[daily_forecast_experiment.json](daily_forecast_experiment.json) declares the
target/source descriptions, columns/units, daily calendar/gap policy, horizon,
minimum available history, issuance, release/revision rules, actual vintage,
benchmark names, and development/final target boundaries. The runner saves all
expanded defaults and a canonical specification hash in manifest version 3.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --experiment configs/daily_forecast_experiment.json \
  --output-directory runs/config-experiment-development-001
```

The sample yields 130 development cases with both point benchmarks. A separate
explicit invocation with `--evaluation-role final` selects its 40 final targets;
it is not started automatically. These are infrastructure demonstrations on
known synthetic fixtures, not evidence of untouched model evaluation. Do not
combine `--experiment` with `--minimum-history`; set the count in the task.
See [the full contract and examples](../docs/experiments.md). Input preparation,
model selection, preserving final-data separation, and human interpretation
remain caller actions. No application or inference backend is selected here.

[daily_evidence_experiment.json](daily_evidence_experiment.json) is a version-2
experiment with the same task/evaluation and explicit required predictive checks
and prior comparisons. `scripts/run_evidence.py --experiment FILE` executes those
workflows only when invoked; the backtest does not start them. Its evidence plan
sets replication count/seed and comparison probability, with optional named
per-origin fit requirements. Version-1 experiments load with empty requirements.
See [the evidence guide](../docs/evidence.md) for a complete command and statuses.

## Diagnostic example

[local_level_diagnostics.json](local_level_diagnostics.json) contains `reference`
local-level settings and named `scenarios` with overrides of that reference.
Each scenario inherits every unchanged reference setting. Omitted reference
settings use the adapter defaults. The command saves every expanded configuration
in `scenario_summaries.json`; retain those values for replay. Names must be
nonblank and cannot be `reference`. Settings are illustrative caller choices,
not a tuning grid or a rule that prior influence should be small.

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --config configs/local_level_diagnostics.json \
  --output-directory runs/config-diagnostics-example-001
```

Prepare the snapshot/tag first as in [the walkthrough](../docs/forecasting.md).
The command calculates and draws diagnostics automatically; read
[their meaning and failure policy](../docs/diagnostics.md). Use an unused output
directory. Acquisition, registration, tagging, and human interpretation remain
explicit caller actions.

## Backtest arguments

| Setting group | How it is selected | Where the run saves it |
|---|---|---|
| Input | `dataset` plus exactly one of `snapshot_id` or `tag`; resolve a tag once | `input` pins the permanent dataset/ID |
| Task/evaluation | `experiment` or legacy daily arguments; default minimum history 30 | Expanded `experiment`, `experiment_sha256`, and task summary |
| Model | `TrainingMean` by default or explicit `LocalLevel(...)` / `NormalMean(...)` / `NormalMeanVariance(...)` / `LocalLevelVariance(...)` / `LocalLevelJointVariance(...)` | `model.configuration` and model identity/capabilities |
| Uncertainty request | `interval_levels=(0.8, 0.95)`; all run scripts accept `--interval-levels 0.8 0.95` | `evaluation` |
| Local storage | Snapshot `store` and a new `output_directory` | Local locations are separate from permanent input identity |

[The model guide](../docs/local-level.md#settings-and-timing) defines the four
fixed local-level settings and their units. [The uncertainty guide](../docs/uncertainty.md)
explains which declared outputs are requested and how reporting chooses a saved
method/level. The point-only baseline creates no uncertainty just because
interval levels are configured.

Reproduction restores saved input/task/model/evaluation choices with matching
code/environment into a new run directory. There is no automatic tuning or model
registry. Acquisition receipts describe input history and are not configuration
files or permanent input paths. Start with [the walkthrough](../docs/forecasting.md)
for an executable example.

## Unknown-mean sampler audit

[normal_mean_mcmc.json](normal_mean_mcmc.json) selects independent-chain sampler
settings and computation/reference tolerances. Use it explicitly with
`scripts/run_normal_mean.py --mcmc-config configs/normal_mean_mcmc.json` alongside
the usual input/output arguments. Omitted fields use `MCMCSettings` defaults;
unknown fields and invalid settings raise before run creation. `--mcmc` selects
defaults without a file. Omit both for exact-only inference. The prior/noise
flags select model assumptions separately; this file does not tune priors.

The runner saves expanded settings, per-fit checks, seeds, diagnostics and draws
automatically. A requested audit's failed criteria fail the forecast case and
run. Forecasts use exact parameter integration in either mode. Reporting and
human review do not select inference settings. See [equations, defaults,
commands, evidence and failure rules](../docs/parameter-uncertainty.md).

## Joint mean-and-variance audit

[normal_mean_variance_mcmc.json](normal_mean_variance_mcmc.json) uses the same
`MCMCSettings` schema with 4 chains, 1500 warmup/6000 retained draws per chain
and proposal scale 1.7 for the two-coordinate audit. Select it explicitly with
`scripts/run_normal_mean.py --learn-variance --mcmc-config configs/normal_mean_variance_mcmc.json`
and the usual input/output arguments. It stores 30,000 rows per fit; start with
a short declared development period. The computation thresholds are unchanged.

`--learn-variance` selects the normal/inverse-gamma model. Its prior flags are
`--prior-mean`, `--prior-mean-precision`, `--prior-shape`, and `--prior-scale`.
The known-variance flags `--prior-variance` and `--observation-variance` conflict
with this switch. Selecting an audit never changes the exact Student-t forecast
method. See [the joint-model guide](../docs/normal-mean-variance.md).

## Prior and posterior predictive checks

The separate `scripts/check_normal_mean_variance.py` accepts explicit prior
flags (`--prior-mean`, `--prior-mean-precision`, `--prior-shape`, `--prior-scale`),
`--replications` (default 1000), and `--seed` (default 0). Exact independent
joint draws need no MCMC config. `--prior-only --observation-count N` needs no
input. Observed checks require `--dataset`, one snapshot/tag, and aware `--as-of`.

An optional `--experiment configs/daily_forecast_experiment.json` supplies
columns, release/revision rules, minimum history and a final boundary. Only
development role is allowed, with as-of time strictly before `final_start`.
Forecast horizon/delay, actual vintage and benchmarks do not schedule or score
these replications. Configuration is saved automatically; no settings are
discovered or selected from discrepancy results. See [commands and semantics](../docs/predictive-checks.md).

## Prior influence comparisons

[normal_mean_variance_priors.json](normal_mean_variance_priors.json) supplies
`reference` settings and named `scenarios` whose overrides inherit unchanged
reference settings. It compares location/precision and inverse-gamma shape/scale
choices. For example, shape 30/scale 29 preserves prior expected variance 1
while changing its concentration. These are illustrative choices, not a search
grid, model ranking or requirement that influence be small.

[local_level_variance_priors.json](local_level_variance_priors.json) supplies
the equivalent reference/overrides for the evolving-level model. Select it
with `scripts/compare_priors.py --model local-level-variance`. Alternatives may
change initial mean/variance and inverse-gamma shape/scale; evolution variance
and numerical settings must match across priors. No single prior-location
coefficient or conjugate scale decomposition is claimed for this model.

Use `scripts/compare_priors.py --config configs/normal_mean_variance_priors.json`
with a dataset, one snapshot/tag, an aware `--as-of`, and a new output directory.
An optional development `--experiment` supplies release rules, daily issuance
delay and final boundaries; `--interval-level` defaults to 0.95. No MCMC settings
are accepted. The CLI saves expanded settings, exact prior/learning paths and
differences, then draws charts automatically. See [timing, quantities and examples](../docs/prior-influence.md).

## Learned local-level observation variance

[local_level_quadrature.json](local_level_quadrature.json) explicitly selects
`QuadratureSettings` for controlled log-variance integration. Use it with
`scripts/run_local_level.py --learn-observation-variance --quadrature-config FILE`
or the predictive-check CLI with `--model local-level-variance`. Omit it for
defaults. The model keeps fixed evolution variance and an independent Gaussian
initial-state prior. `--prior-shape`/`--prior-scale` configure the observation-
variance prior; `--observation-variance` conflicts with learned mode. Numerical
settings are saved and required failures reject a forecast; they do not tune
model assumptions. See [methods, precision and commands](../docs/local-level-variance.md).

Sequential checks reuse the reference settings in existing prior configurations. Select `scripts/check_sequential.py --config FILE` explicitly; add `sequential_checks` to an experiment evidence plan to require it. See [usage](../docs/sequential-checks.md).

`scripts/calibrate_model.py` reuses reference settings from the prior JSON files. Trial count, observation count and seed are explicit CLI options, separate from historical experiment settings. See [calibration](../docs/calibration.md).

[local_level_joint_priors.json](local_level_joint_priors.json) declares the joint model reference: explicit categorical evolution-variance support/probabilities and a continuous inverse-gamma observation-variance prior. `scripts/run_joint_local_level.py` uses it explicitly. It is a discrete modeling assumption, not an automatically tuned numerical grid. See [the model](../docs/joint-variance.md).

The joint prior config includes explicit `evolution_weights`, V-scale and initial-level alternatives. Evidence/prior CLIs restore list-valued JSON settings through the model configuration loader before applying comparisons. See [joint evidence](../docs/joint-evidence.md).

Joint calibration uses `scripts/calibrate_model.py --model local-level-joint-variance --config configs/local_level_joint_priors.json`. Set `initial_variance: 0` explicitly to investigate known-initial mixed-state atoms. Calibration options remain separate from historical experiment evidence settings. See [the mixed calibration guide](../docs/mixed-calibration.md).

## Saved review requirements

The explicit [three-model study](../docs/model-review-study.md) expands the three
existing prior configuration files above, retaining every named alternative.
Its experiment requires predictive, prior-influence and sequential evidence;
the study saves expanded settings and configuration hashes before execution.
It runs only when the caller invokes `scripts/review_model_study.py`.
The separate `scripts/check_study_priors.py` restores expanded alternatives from
that saved design, rather than rereading possibly changed configuration files.

`scripts/review_experiment.py` reads the evidence plan from the saved forecast;
it accepts explicit forecast/evidence/calibration folders and a new output folder,
without discovering configurations or running missing work. Calibration is required
only with `--require-calibration`. See [review usage](../docs/experiment-review.md).

[Full distribution scores](../docs/distribution-scoring.md) run automatically for
adapters declaring `supports_distribution`. No scoring-specific model setting,
extra simulation or density floor is configured. Saved comparisons require the
same task/units/vintages and scoring method; point-only adapters remain supported.

[Repository CI](../docs/continuous-integration.md) installs from `uv.lock`, checks
the pinned interpreter and compares a regenerated temporary requirements export.
Configuration/dependency updates must keep the declaration, lock and pip export
together. Large calibration settings remain explicit scientific workflow choices.
