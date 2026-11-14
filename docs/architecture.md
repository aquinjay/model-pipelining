# Architecture

Status: high-level design. Names describe responsibilities; they are not a finalized API or a list of modules that must all exist immediately.

The first executable boundary now exists in [the forecasting workflow](forecasting.md): `models.py` defines an immutable history/issuance/target-time request and a small point-forecast protocol; `backtest.py` selects expanding histories, calls the model, freezes predictions, scores, and saves evidence; `reporting.py` reads saved results. This implements one daily CSV task with fixed required checks. The richer capabilities, diagnostics, configurable check plan, and release policy below remain design guidance.

## Separate the problem, model, and execution

| Component | Owns | Boundary |
|---|---|---|
| Application specification | Target meaning, units, calendar, issuance, allowed features, benchmarks, and evaluation cases | Must not embed a particular model's internals |
| Data preparation | Source access, preserved snapshots, quality checks, and availability information | A reproducible extract does not automatically reconstruct historical availability |
| Model adapter | Model-specific input transformation, fitting, prediction, capabilities, and specialized checks | Must not own the shared backtest, artifact layout, or delivery policy |
| Runner | Historical origins, eligible training views, stage dispatch, validation, and recorded execution policy | Should not branch on model-family names |
| Evaluation | Scores and comparisons compatible with the returned predictions | Must not infer uncertainty or latent states that were never supplied |
| Artifact storage | Run identity, manifests, predictions, evidence, lineage, and review records | Preserve issued predictions and the versions used for scoring |
| Reporting | Displays saved results and limitations | Must not silently refit models to produce a report |

Start with a few files that express these boundaries. Extract additional modules only when concrete use demonstrates the need.

## Inputs and outputs

At a high level, every run needs a task specification, a model configuration, a data reference, an evaluation protocol, and a check plan.

Every forecast needs an unambiguous issuance time, training cutoff, series and target coordinates, horizon, target scale/version, and model/run identity. Point forecasts should identify their statistic, such as a predictive mean. Optional uncertainty must identify what it represents.

The first experiment saves task/model/data references in a JSON manifest and predictions/evaluation in CSV; [its record descriptions](forecasting.md#what-is-saved) document the current scope. Broader records remain open. A documented conceptual boundary should not become a large unused schema ahead of implementation.

## Capabilities and checks

Models declare the outputs and operations they actually support. Examples include point forecasts, quantiles, predictive samples, incremental updates, or latent-state diagnostics. Shared evaluation selects compatible calculations from those capabilities.

The proposed checking boundary lets an adapter describe its checks and execute a selected check in a stage-specific context. It may delegate implementation to a separate diagnostic component. Shared validators use a common result format too.

| Question | Bayesian example | Non-Bayesian example |
|---|---|---|
| Are the assumptions plausible before fitting? | Prior predictive simulations | Feature availability and intended-use coverage checks |
| Did inference or fitting work? | Recursion verification; sampler diagnostics when used | Fit completion and output validity |
| What does the fitted model miss? | Posterior predictive replications and prior sensitivity | Residual analysis, ablation, and regularization sensitivity |
| How does prediction perform? | Point scores and supported distributional scores | Point scores and any explicitly supported probabilistic outputs |

These checks share lifecycle and evidence conventions while retaining their different meanings. Feature importance alone is not assumption validation; a Bayesian label does not make MCMC diagnostics applicable.

The experiment owns the required check plan, applicability policy, criteria, and failure actions. An adapter cannot waive a requirement by omitting a check. Missing, failed, errored, deferred, or unresolved required evidence remains visible and blocks release according to the plan. Exploratory execution can still save evidence; run completion and release readiness are separate judgments.

## Information boundaries

- At each backtest origin, use data and features available at that issuance time.
- Fit learned preprocessing and choose tuning settings within the allowed development/training boundaries.
- Keep future actual targets out of prediction requests and any check contexts used at issuance.
- Freeze predictions before joining actuals for scoring.
- Label fitted-data predictive replications separately from held-out forecasts.
- Preserve a final evaluation period; if its results guide revision, record its new role as development evidence.

## Simulation and model revision

The application/test harness can supply known scenarios for any model. Generative models can additionally supply simulations from their own assumptions. Prior prediction, parameter/state recovery, formal simulation-based calibration, and future predictive calibration answer distinct questions.

Run expensive simulation suites at an appropriate model/configuration scope, with recorded provenance and budget. Do not automatically repeat them inside every backtest fold. Reassess applicability after relevant model, code, prior, or inference changes.

Preserve the parent iteration, motivating discrepancy, changed assumption, evidence, and review decision. Comparisons should explain changes in conclusions as well as changes in scores.

## Scope boundary

The first implementation is a local forecasting experiment, executed manually. Automated deployment, alerting, tracking services, dashboards, causal-effect estimation, and a universal plugin system remain later possibilities.
