# Roadmap

These milestones describe proposed outcomes. The repository currently contains documentation and placeholders only. Nothing below has been implemented.

## 0. Establish the design

- [x] Write the project purpose and initial scope.
- [x] Describe application, adapter, and shared-pipeline responsibilities.
- [x] Preserve the Bayesian and non-Bayesian checking approach.
- [x] Create a Git-ready repository layout and task/review templates.
- [ ] Agree on the first synthetic experiment and its observable acceptance criteria.

## 1. Small synthetic experiment

Seeded synthetic data → training-mean baseline → rolling backtest → saved results → one chart.

The initial implementation should demonstrate correct target alignment, no future information in prediction, explicit failed folds, and reconstruction from preserved inputs/configuration. A deliberately invalid output or missing required check must be visible. The report reads saved results without fitting.

Review the harness before adding model complexity: keep, simplify, or restart using the evidence preserved from this slice.

## 2. One real-data application

Add one source and one numerical target with explicit timing, calendar, snapshot, and availability assumptions. A next-session market-return task is a candidate; the instrument, source, and dates remain undecided.

Hold the task and comparison protocol fixed before interpreting model improvements. Forecasting performance and trading performance are separate evaluations.

## 3. First Bayesian adapter

Add a local-level BDLM through the same runner and storage path. Begin with fixed observation/evolution variances and explicit initial-state assumptions. Document uncertainty omitted by that simplification.

Include prior predictive simulation, synthetic inference verification, fitted-model criticism/sensitivity, and compatible held-out scores. Compare with the baseline without rewriting shared execution. MCMC diagnostics are inapplicable to an analytic filter.

## 4. Prove the boundary with a contrasting model

Add a point-only model such as gradient boosting. Supply its feature, fitting, and model-criticism checks through the same check boundary. Keep uncertainty evaluation explicitly inapplicable unless supported outputs are added.

## Later, when a need is demonstrated

Multiple horizons, richer Bayesian models, an operational-data application, a run index, dashboards, scheduled execution, and monitoring policies. Each extension needs a stated problem and evidence that it helps.

## Next design slice

Write the smallest synthetic-task specification together: generated process, target, frequency, issuance/cutoff, minimum history, origins, baseline, point statistic, scores, and expected artifacts. Choose the Python/runtime and first dependencies when implementing that slice.

Turn each deliverable into a small GitHub issue using the repository's task template. This roadmap is not a commitment to implement all possible extensions.
