# Checkpoint and next session

Updated: 2026-11-19. Start here when resuming the modeling work, then consult the
[roadmap](roadmap.md) and [modeling review](modeling-review.md).

## Where we left off

The research checkpoint merged in [PR #7](https://github.com/aquinjay/model-pipelining/pull/7)
as `b62b70f`. [PR #8](https://github.com/aquinjay/model-pipelining/pull/8) merged
the Python 3.12 profile and 366-case public NOAA forecast as `f7aec5a`. The
seasonal follow-up selected an annual curve plus one temperature lag on the
2024 development dates. With the settings frozen before the 2025 download,
the model beat both controls on the 365 later dates. Its MAE was 2.945 °C
versus 3.214 °C for persistence, and its mean forecast error was +0.423 °C.
See the [verified setup](python312-public-forecast.md), [development study](noaa-seasonality.md),
and [later-year results](noaa-later-period.md).

The project is developing repeatable, model-independent forecasting experiments:
preserved inputs, release-aware histories, explicit assumptions, comparable saved
results and separate computational checks and human interpretation. Application
selection remains deferred. The evidence-interpretation phase is complete;
the [three-model development study](model-review-study.md) records results,
prior robustness, limitations and the next bounded modeling question.
Bayesian dynamic linear models are the first main use case. Keep work on them
bounded to their concrete needs and keep shared code usable for harmonic
regression, other state-space models, and multivariate forecasting. Use the
[reading guide](reading-guide.md) to trace the statistics and code behind each
step.
The [short-history innovation reference](innovation-reference.md) and
[longer-history particle investigation](innovation-particles.md) are implemented.
The state-grid proposal passed the existing probability criteria on all 17
frozen cases. A continuous-state Gibbs calculation independently agreed on the
saved moments and predictive density for `abrupt-42`. Repeated grid fits took
about 16 seconds each. Use the existing Gaussian local-level adapters for the
first BDLM workflows and keep the rare-jump model as a research option. A
weekly harmonic example now exercises the same shared runner and reporting.
See the [independent check](innovation-moment-check.md) and
[harmonic example](reading-guide.md#run-the-weekly-harmonic-example).

The `feature/innovation-inference-study` branch records the development review,
numerical repair, innovation prototypes, tests and curated evidence. It starts
from merged checkpoint `0ff5def`. Review and hosted CI belong to the PR stage.
Commit `092efcd` records the development review, reference and prior probability
follow-up. The subsequent checkpoint records the state-grid study, independent
Gibbs check, harmonic example, and reading guide. It passed 449 tests and the
documentation walkthrough locally. Full run directories are ignored by Git.

The preceding merged experiment-review checkpoint is recorded on `main` through
[PR #6](https://github.com/aquinjay/model-pipelining/pull/6), merge commit
`0ff5def`. Local `main` was synchronized after the merge. The three implementation
commits are:

| Commit | Completed work |
|---|---|
| `22e1adb` | Saved experiment review: collect forecast, historical evidence and calibration; check identities, receipts, requirements and omissions; write Markdown/JSON with human review pending |
| `69a58db` | Freeze full observation laws before scoring; calculate Student-t/Gaussian-mixture CRPS and negative log density; compare matching saved cases with paired differences and charts |
| `72045dd` | Locked-environment CI, documentation links/anchors/example checks and temporary execution of the forecasting walkthrough |

All **391 tests passed** in a fresh Python 3.14.10 environment with 24 locked
dependencies. Dependency/export checks, documentation checks and both 170-case
walkthrough runs passed. Hosted
[push CI](https://github.com/aquinjay/model-pipelining/actions/runs/44343984745)
and [PR CI](https://github.com/aquinjay/model-pipelining/actions/runs/44344085293)
also passed before merging. These describe the checkpoint, rather than a promise
that future edits have already been verified.

## Completed development review — 2026-11-19

The development study and numerical repair are recorded in `092efcd` on the
feature branch, based on the merged checkpoint above.

| Evidence | Completed scope |
|---|---|
| Reference reviews | Nine ready-for-review combinations: three models on mean shift, stable and gradual change |
| Forecast sensitivity | 45 forecasts, 5,850 matching development cases, every declared prior alternative |
| Required historical evidence | 27 predictive/prior-influence/sequential workflows; 18,000 reference replications |
| Alternative predictive criticism | 36 additional workflows, 72,000 prior/posterior replications |
| Fresh reference calibration | 1,000 20-observation trials per model, 3,000 total; zero replacement-run failures |
| Saved aggregation/audit | 180 window rows, 7,020 paired-loss rows, 102 verified report links; pre-change identity and sequential/frozen-law PIT agreement |
| Repository checks | 401 tests, local links/examples, temporary two-model 170-case walkthrough and whitespace checks passed |

A first joint calibration trial failed because numerical conditional quantiles
did not bracket the marginal target under an almost dominant W component. The
repair widens the failing side to checked integration support, retaining
tolerances. The original failed trial is saved separately and all 1,000 truth/
data files replay exactly after repair. A seeded independently bracketed-root
regression supports it. This is the same-seed repair exercise, not an independent
calibration follow-up. Human review files remain pending; the written study
interprets evidence without granting model approval.

## Modeling and evidence already available

| Area | Existing capability and reference |
|---|---|
| Data and execution | Preserved snapshots/tags, release/revision selection, explicit development/final boundaries, matching benchmarks and retained failures; [experiment contract](experiments.md) |
| Constant-mean models | Unknown mean with known observation variance, and unknown constant mean/variance with exact Student-t prediction; optional MCMC audits against exact references; [parameter inference](parameter-uncertainty.md), [mean/variance model](normal-mean-variance.md) |
| Dynamic models | Fixed-variance local level, learned observation variance with fixed evolution variance, and joint observation/categorical evolution variance learning; [scalar model](local-level-variance.md), [joint model](joint-variance.md) |
| Predictive criticism | Prior/posterior dataset replication, saved discrepancies, full-CDF sequential PIT/errors and residual dependence; [predictive checks](predictive-checks.md), [sequential checks](sequential-checks.md) |
| Prior influence | Matched learning paths, predictive differences and joint-model evolution probabilities; influence is descriptive; [prior comparisons](prior-influence.md), [joint evidence](joint-evidence.md) |
| Inference validation | Independent numerical references, prior/data calibration and recovery, discrete/mixed CDF handling and retained original/follow-up exercises; [calibration](calibration.md), [mixed calibration](mixed-calibration.md) |
| Review and comparison | Compatible saved evidence bundles, full-law losses, matched comparisons and charts; [review guide](experiment-review.md), [scoring guide](distribution-scoring.md) |

The completed review uses these capabilities. Scientific model adequacy and human
approval remain open even when all computational checks pass. Additional MCMC
work should follow a specific inference need found in the review.

## Completed reference prototype — 2026-11-19

`innovation_reference.py` enumerates every positive-prior small/large daily
innovation sequence, learning V while transition probability/scales stay fixed.
Batch Gaussian/adaptive integration and independent Kalman smoothing/log-V
quadrature agree within 4.78e-11 on seven frozen cases. Zero/one probability and
coincident-scale limits agree with the existing scalar model within 3.90e-12.
Known-state atoms and V/allocation dependence remain explicit. All numerical
case files replay exactly. That reference checkpoint added no long-history
approximation or forecast adapter; its suite passed 415 tests, including fourteen reference
checks and CLI/failure/replay tests.

The eight-observation cap permits 256 past sequences (65,536 joint sequence/V
nodes in the recorded case). Nondegenerate 160-step enumeration would require
about 1.46e48 paths before V integration. The saved calculation is an oracle for
a later approximation, not an execution method for the earlier full study.

## Completed three-task review phase

### 1. Complete experiment reviews — done

Collected all nine reference reviews with explicit snapshots/settings, timing,
actual vintages, required evidence and final boundaries. Old plans/source hashes
were incompatible, so the new exercise preserves earlier evidence separately.
Fresh calibration retains its model-level 20-observation scope and seeds/failures;
it does not certify every historical sample size. Each review links charts and
sources and lists the existing environment omissions explicitly.

### 2. Investigate discrepancies and priors — done

Compared stable/gradual/abrupt data with identical noise and fixed 70/20/40-case
before/onset/later windows. Losses, coverage/width, PIT, residual dependence,
learning paths and whole-dataset replications retain all declared prior choices.
Joint W/V improves stable-period behavior but has higher onset losses than fixed
W on the shift. Its extra adaptation cost concentrates around one additional
update; later dynamic forecasts are very similar. Larger-V priors reverse the
small aggregate CRPS ranking, while the replicated shift-pattern CUSUM discrepancy
persists. Gradual change retains a separate later-window bias. See the study's
tracked data and chart; none of these descriptive results selects a winner.

### 3. Record the next modeling question — done

The models serve the infrastructure benchmark and remain references. The
[decision](decisions.md#interpret-model-evidence-before-choosing-the-next-transition--2026-11-19)
proposes investigating occasional large state innovations while retaining quiet
periods. The study gives numerical references, Gaussian limits, precision,
predictive criticism, calibration and matched-comparison requirements. A new
adapter, backend or application is not implemented by this review.

## Particle study results — 2026-11-19

Work is wrapped up at a complete numerical investigation. The implementation is
`src/model_pipelining/innovation_particles.py`; the explicit study command is
`scripts/run_innovation_particles.py` with `configs/innovation_particles.json`.
See [the particle guide](innovation-particles.md) for its API, exact scope,
results and reproduction. No forecast/provider integration was started.

- The particle checkpoint passed **431 tests**, including sixteen new particle
  tests, and passed the documentation checks and both temporary 170-case runs.
- The frozen 17-case study is complete as an investigation but its precision
  status is **unresolved**. All six 160-observation cases pass empirical gates
  using 16,384 or 32,768 particles per independent run (32 runs per level).
- Two short cases fail oracle uncertainty comparisons: quiet paths `011100` and
  `101001` were unvisited despite positive oracle mass; gradual path `000111`
  was underestimated. There are three failed envelopes. Do not treat zero MCSE
  as proof of zero probability or bypass these checks.
- The saved audit passes: eight source hashes, 34 level aggregations, eleven
  exact short-case replays and twelve selected long-replicate replays. This
  verifies saved computation, not scientific adequacy or the failed precision
  contract. The full long study was not replayed.
- Main evidence: `runs/innovation-particles-verified-20261119/`, especially
  `summary.json`, `validation.json`, `cases/short-quiet.json` and
  `cases/short-gradual.json`. Curated summary/figure are in `docs/artifacts/` and
  `docs/figures/`. Earlier interrupted runs are retained and explained in
  `execution.json` and the guide.

## Next work

1. Keep the frozen 2025 result as a point-forecast benchmark. Further NOAA
   studies should address the warm bias or the quiet-day errors on new
   development data and reserve another period for evaluation.
2. A small two-series follow-up can compare separate forecasts of Central
   Park's daily minimum and maximum on matched dates. Count forecasts with a
   minimum above the maximum before deciding whether joint forecasting needs
   a shared interface change. An operational NOAA workflow still needs
   documented release timing and vintages. Workplace testing remains deferred.
3. Revisit rare-jump integration if the task needs it. Use the saved timings and
   cross-checks to assess the cost, then specify the required predictive outputs
   and fresh-path criticism, prior comparisons, and calibration.

## Independent moment and density checkpoint

Both Gibbs budgets on `abrupt-42` passed against the original grid and two grid
reruns. Five short oracle controls passed at the first budget. `short-abrupt`
needed longer chains to meet the frozen precision criteria. A 32-draw grid
limit returned `unresolved` and preserved the failure record. No additional
dependency was needed for the check or the 72-case harmonic example. See the
[study and reproduction commands](innovation-moment-check.md).

## State-grid checkpoint — 2026-11-19

The three-case pilot used `short-abrupt`, `short-dominant` and `abrupt-42`.
The direct-convolution rerun passed all three at the original sample count,
confidence allocation, and width limit of 0.10. The full 17-case repeat used
the same settings. The earlier FFT pilot remains archived because a dense
Gaussian limit exposed significant roundoff in small likelihoods.
All 446 tests passed. The audit recalculated all 17 interval sets and matched
the three pilot case files byte for byte.

The grid likelihood bounds hold pathwise in exact arithmetic. Independent
weighted draws and deterministic grid/tail allowances satisfy the probability
criteria at the saved probes. Point estimates of densities and moments remain
descriptive. The existing particle fits are empirical comparisons for long
histories. See the [derivation and full results](innovation-lattice.md).

## Conservative probability checkpoint — 2026-11-19

`src/model_pipelining/innovation_probability_bounds.py` adds a separate fixed-budget
importance calculation with an analytic [0,1] weight envelope. Simultaneous
empirical Bernstein numerator/normalizer bounds remain nonzero for unvisited
events and return unresolved precision when the normalizer cannot be bounded
away from zero. This is a mathematical finite-sample probability contract under
iid/exact-arithmetic assumptions, not a bound on floating error, moments or
densities. The original particle method and study remain unchanged.

- The explicit command is `scripts/check_innovation_probability_bounds.py`; see
  [usage, proof, results and next question](innovation-probability-bounds.md).
- The frozen follow-up uses 262,144 independent draws per case with a 95% error
  allocation across all 17 original cases and their probes. All eleven short
  oracles' 872 probability quantities are enclosed. Three short cases and all
  six long cases still fail the declared width/normalizer precision gate.
- The quiet paths `011100`/`101001` and gradual path `000111` now have positive
  uncertainty intervals containing their original oracle masses. Missing paths
  and nearly dominant priors have explicit regressions.
- Evidence is in `runs/innovation-probability-bounds-checked-20261119/`, with
  copied original inputs/failures, implementation sources and fingerprints,
  sufficient statistics, interval/check recalculations and a saved-only chart.
  Curated results are tracked under `docs/artifacts/` and `docs/figures/`.
- The audit verifies ten archived/current source hashes and 31 archived/unchanged
  parent inputs, recalculates all intervals/checks and replays all 17 independent
  fixed-budget calls exactly. It does not rerun the original particle study.
- That checkpoint passed **442 tests**, including eleven probability-bound tests. The
  three original failed envelopes and the exploratory bound runs are retained.

## Saved local evidence to inspect

These paths existed at this checkpoint. Generated run folders are ignored by Git
and may be absent in another checkout; the tracked guides explain reproduction.
Keep source folders with their reports and inspect their manifests before reuse.

| Local path | What it contains |
|---|---|
| `runs/noaa-2025-frozen-20261119/` | Plan saved before downloading 2025, exact 2023–2024 history, 2025 snapshots, matched forecasts, and the later-year audit inputs |
| `runs/noaa-seasonality-verified-20261119/` | Baseline error diagnostics, matched forecasts and reports for three regressions, and their aggregate and grouped point scores |
| `runs/noaa-central-park-20261119/verified-py312/` | Public NOAA raw and normalized snapshots, 366 scored forecasts, benchmark, chart, and design |
| `runs/innovation-moment-check-verified-20261119/` | Frozen controls, independent Gibbs budgets on abrupt-42, grid timing and insufficient-budget result |
| `runs/innovation-moment-control-refinement-20261119/` | Independent longer short-abrupt control against the same oracle and criteria |
| `runs/weekly-harmonic-example-20261119/` | 72 matched scalar forecasts and comparison report for the harmonic example and training mean |
| `runs/innovation-lattice-full-20261119/` | State-grid results for all 17 cases, copied input and source artifacts, validation and comparisons |
| `runs/innovation-lattice-pilot-direct-20261119/` | Three-case pilot with direct convolution at the full declared budget |
| `runs/innovation-probability-bounds-checked-20261119/` | Fixed-budget independent probability follow-up, original inputs/failures, source snapshots, bounds and unresolved precision; see its guide |
| `runs/innovation-particles-verified-20261119/` | Frozen 17-case particle/oracle investigation, all refinements and empirical uncertainty failures; see the particle guide for results and validation scope |
| `runs/innovation-reference-verified-20261119/` | Seven frozen short numerical histories, both independent calculations, three Gaussian limits, feasibility/timing, saved chart and validation; curated exports are linked in the guide |
| `runs/model-review-study-20261119-fixed/` | Complete 45-forecast study, nine reference reviews, current calibration, matched comparisons, summaries, retained continuation sources and `validation.json`; use `study.json.results[].latest_review` for latest reports |
| `runs/model-review-prior-checks-20261119/` | All 36 alternative-prior predictive workflows, frozen parent design, charts and `validation.json` |
| `runs/model-review-study-20261119/` | Retained first failed calibration exercise and interrupted historical evidence; joint trial 538 is preserved |
| `runs/experiment-review-validation-jdg8hqd_/review-final/report.md` | First complete assembly exercise: 130 development forecasts, all three historical workflows, 200 predictive replications per phase and two small 50-trial calibration exercises; those trials validate assembly mechanics |
| `runs/distribution-scoring-validation-89wgplk4/` | Three matching 130-case mean-shift forecasts, two full-law comparisons and a joint-model review |
| `runs/distribution-scoring-validation-89wgplk4/comparison-2/distribution_scores.png` | Student-t versus joint-variance CRPS/log-loss chart |
| `runs/dynamics-study-initial/` | Earlier controlled stable/gradual/abrupt scenario study |
| `runs/joint-calibration-followup-1000-814/` | Separate 1000-trial joint-model follow-up; retain its original study and finite-simulation context described in the mixed calibration guide |

On the known mean-shift fixture, mean CRPS was 1.64837 / 0.66828 / 0.66253 for the
constant-mean / learned-noise / joint models; mean negative log density was
2.73546 / 1.64919 / 1.67372. The joint model improved CRPS slightly relative to
the learned-noise model while its log loss was slightly worse. These are descriptive
development observations to investigate, not an untouched evaluation or a final
model choice.

## How to resume

Read this note and the linked study guides, then inspect the working tree and
use the retained evidence for the pilot. Run a new experiment, assemble a review,
or run a large calibration study explicitly when needed. Configured CI runs
automatically on branch pushes and pull requests.

```bash
git status --short --branch
PYTHONPATH=src .venv/bin/python scripts/check_documentation.py
```

For future changes, update the affected docs and run the relevant checks. Use
the [test guide](../tests/README.md) and [CI guide](continuous-integration.md) for
verification commands and writable cache settings.
