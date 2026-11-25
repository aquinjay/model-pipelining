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
The [temperature pair study](noaa-temperature-pair.md) then matched a separate
minimum-temperature fit with that frozen maximum run. Both model families
had zero crossed point forecasts over 365 dates. Range MAE was 2.263 °C for
the seasonal pair and 2.655 °C for last-value. The seasonal pair's error
correlation was 0.603.

The `feature/noaa-temperature-pair` branch records the paired study and
handoff on top of `e385248`, the commit for the 2024 seasonal comparison and
frozen 2025 maximum evaluation. The branch starts from `main` at `f7aec5a`.
The paired study passed 86 targeted tests on Python 3.12 and replayed five
saved files byte for byte. See the [paired result and audit](artifacts/noaa-extrema-summary.json).

The `feature/model-configuration-restore` branch started from the NOAA branch. It
fixes restoring saved MCMC settings for the constant-mean models and gives every
adapter `from_configuration`. Scripts now build models and prior scenarios
through one catalog, and calibration rejects malformed parameter names when it
selects the provider. Forecast outputs and saved formats are unchanged. See the
[decision](decisions.md#rebuild-adapters-from-saved-settings-through-one-catalog--2026-11-19).

Both branches merged through [PR #9](https://github.com/aquinjay/model-pipelining/pull/9)
(`65035a1`) and [PR #10](https://github.com/aquinjay/model-pipelining/pull/10)
(`6dd958b`). The `feature/shared-names-and-helpers` branch then defines
benchmark and workflow names once and writes saved JSON/CSV through
`artifacts.py`; see the
[decision](decisions.md#define-shared-names-once-and-write-artifacts-through-one-module--2026-11-19).
It merged through [PR #11](https://github.com/aquinjay/model-pipelining/pull/11)
(`e54def4`). The `feature/harmonic-adapter-in-src` branch moves the weekly harmonic
adapter into the package. The frozen NOAA scripts stayed in place; see the
[decision](decisions.md#move-the-weekly-harmonic-adapter-into-the-package-keep-frozen-noaa-scripts-unchanged--2026-11-19).
It merged through [PR #12](https://github.com/aquinjay/model-pipelining/pull/12)
(`5162d72`). The `feature/python312-ci` branch adds a CI job that runs the suite on
the Python 3.12 workplace profile; see the
[decision](decisions.md#check-the-python-312-workplace-profile-in-ci--2026-11-19).

Benchmark validation again raises `ValueError` for malformed names. The
temperature adapter, NOAA parser and saved-forecast helpers are now in the
package, and the paired study records their source hashes. The frozen 2025
maximum study replays with its original scripts. See the
[decision](decisions.md#package-temperature-code-and-preserve-the-frozen-noaa-replay).

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

### Completed: model-independent saved decision layer (Phase 1, Slice 1 — 2026-11-24)

The [model-independent decision layer](decision-value.md#model-independent-decision-layer)
is implemented in [decisions.py](../src/model_pipelining/decisions.py) with
continuous asymmetric (newsvendor) and expense-concentration extensions in
[decision_value.py](../src/model_pipelining/decision_value.py) and CLI
`scripts/run_decisions.py` (`issue`, `score`, `compare`).

- **Pre-outcome frozen issuance:** `DecisionSpec` pins event/action/target/horizon,
  policy (`binary_cost_loss` or `continuous_asymmetric`), calibration reference,
  and explicit fallback before scoring. Issuance reads only forecast-time
  artifacts and freezes `recommendations.csv` with SHA-256 receipts.
- **Separate outcome scoring and replay:** `score_decisions` writes expenses,
  tail-concentration diagnostics (median, p90, p95, max, worst-5% share), and
  pending `review.json` to a new directory without mutating issued files. Both
  the saved Bayesian `LocalLevel` Central Park forecast (`runs/noaa-central-park-20261119/verified-py312/forecast/`)
  and the frozen NOAA range probabilities (`runs/noaa-range-2026-h1/`) were
  verified end to end and replayed byte for byte under
  `runs/decision-layer-verified-20261124/`. Unit tests verify that mutating a
  later outcome updates expense while issued recommendations stay fixed.
- **Paired comparisons:** `compare_decision_scores` verifies identical specs and
  cohorts and reports mean/median differences, win/loss/tie rates, action
  agreement, worst-5% difference concentration, and Bartlett HAC SEs.

### Completed: robust evaluation summaries, probabilistic baselines, and decision review integration (Phase 1, Slice 2 — 2026-11-24)

- **Shared robust diagnostics ([robust_scoring.py](../src/model_pipelining/robust_scoring.py)):**
  Computes signed bias (mean/median error), MdAE, p90/p95/max errors, worst-5%
  loss concentration (with signed-loss handling for negative log density),
  paired win/loss/tie rates, and deterministic Bartlett (Newey-West) HAC SEs.
- **Upgraded comparisons and unified review:** `compare_runs` in
  [reporting.py](../src/model_pipelining/reporting.py) now saves
  `point_comparison`, per-run CRPS/log-score tail summaries, paired HAC
  stability diagnostics, and a 3-panel `distribution_scores.png` including
  cumulative CRPS difference. `build_experiment_review` in
  [experiment_review.py](../src/model_pipelining/experiment_review.py) (and
  `scripts/review_experiment.py`) surfaces bias, MdAE, worst-5% concentration,
  HPD coverage/width, and pinball losses, and optionally links and verifies
  saved `decision_directories` (`--decision-directories`, `--require-decisions`).
- **Probabilistic baselines ([probabilistic_baselines.py](../src/model_pipelining/probabilistic_baselines.py)):**
  Zero-tuning reference Student-t `GaussianClimatology` and multi-horizon
  `ResidualPersistence` adapters implement the full `Forecaster` contract
  (`quantiles`, `highest_density`, `distribution`, `from_configuration`), with
  CLI `scripts/run_probabilistic_baseline.py`.
- **Empirical findings (`runs/robust-evaluation-study-20261124/`):**
  1. On Central Park 2024 TMAX (366 cases), `ResidualPersistence` beat fixed
     `LocalLevel` on CRPS (2.104 vs 2.119), HPD95 width (15.24 vs 15.68 °C),
     binary $\ge 25$ °C decision expense (0.1148 vs 0.1243, HAC SE 0.0044), and
     3:1 asymmetric expense (4.539 vs 4.618).
  2. Expanding `GaussianClimatology` had negative binary decision value (−0.466)
     on the bimodal seasonal cycle, motivating an empirical/nonparametric or
     seasonal distribution family in Phase 3.
  3. On the 130-case synthetic mean-shift study (`LocalLevelVariance` vs
     `LocalLevelJointVariance`), paired CRPS (+0.0058 ± 0.0153 HAC SE),
     log-score (−0.0245 ± 0.0479 HAC SE), and squared-error (−0.0535 ± 0.1051
     HAC SE) differences are well inside noise, with 34–49% of total absolute
     difference concentrated in 7 transition days.

### Completed: empirical distributions, nonparametric baselines, and Brier/Murphy diagnostics (Phase 1, Slice 3 — 2026-11-24)

- **Empirical weighted-sample laws (`empirical_samples`):** Added to
  [distribution_scoring.py](../src/model_pipelining/distribution_scoring.py)
  (exact $O(k \log k)$ nonnegative step-integral CRPS; `log_score=None` without
  inventing a density) and [decisions.py](../src/model_pipelining/decisions.py)
  (exact strict/inclusive discrete event probabilities and left-continuous
  inverse CDF quantiles). Cross-family comparisons (continuous vs. empirical)
  work seamlessly in `compare_runs` and `experiment_review.py`.
- **Nonparametric baselines ([probabilistic_baselines.py](../src/model_pipelining/probabilistic_baselines.py)):**
  Added `EmpiricalClimatology` and `EmpiricalResidualPersistence` (optional
  rolling `window`, `supports_highest_density=False`), preserving skewness and
  multimodality.
- **Calibration & Murphy diagnostics ([decision_value.py](../src/model_pipelining/decision_value.py), [decisions.py](../src/model_pipelining/decisions.py)):**
  Added exact Brier reliability/resolution/uncertainty decomposition (plus ECE,
  probability bias, and calibration-baseline BSS) and Ehm et al. (2016) Murphy
  curves and dominance checks (`first_dominates`, `second_dominates`, `tied`,
  `crossing`), rendered in `decision_scores.png` (reliability diagram panel) and
  `decision_comparison.png` (Murphy diagram panel).
- **Empirical findings (`runs/option-a-empirical-study-20261124/`):**
  1. On Central Park 2024 TMAX, `EmpiricalResidualPersistence` beat both
     `LocalLevel` and Student-t `ResidualPersistence` on CRPS (2.0995 vs 2.1036
     vs 2.1188), Brier (0.0736), reliability (0.0028), resolution (0.1467), and
     ECE (2.17%), halving probability bias (−0.0086 vs −0.0209). Rolling 30-day
     `EmpiricalClimatology` reached `V = +0.647` with zero probability bias.
  2. On NOAA 2026 H1 range forecasts, Brier decomposition showed `same_day`
     pairing had more than 2× the resolution of `independent` (0.03407 vs
     0.01566) but 3× worse reliability error (0.04525 vs 0.01497, ECE 16.2% vs
     10.1%), and Murphy curves crossed (independent better on 14/19 ratios,
     same-day better on 4/19).

### Up next: viewer, live issuance/settlement, and calendar flexibility

1. **Local read-only comparison and decision viewer (Phase 2A):**
   Build a small local read-only viewer (e.g., Streamlit) over saved run,
   comparison, and decision directories showing observed values, HPD intervals,
   cohort-mismatch checks, cumulative loss/expense differences, reliability and
   Murphy diagrams, and an interactive cost-loss / asymmetric-cost control.
2. **Two-stage live issuance and personal shadow trial (Phase 2B):**
   Separate forecast issuance in `backtest.py` (for targets without realized
   outcomes yet) from later settlement so the framework can run a shadow trial
   on one recurring, reversible personal decision with predeclared rule,
   deadline, and miss cost.
3. **Calendar flexibility, packaging, and run lineage (Phase 3 & 4):**
   Relax `daily_utc` + `missing_policy="reject"` in `experiments.py` for
   real-world calendars/gaps, enable editable packaging in `pyproject.toml`,
   and add a lightweight run index / iteration lineage over `runs/`.

### NOAA research checkpoint

The [range-probability study](noaa-range-probability.md#januaryjune-2026-results)
is complete. The plan pinned the full 2025 calibration series, model settings
and sources before retrieving January–June 2026 data. On 181 matched dates,
with 64 ranges of at least 10 °C, independent pairing had a mean Brier score of
0.22415 and same-day pairing had 0.23935. Same-day pairing lowered mean
negative-range probability from 5.01% to 0.91%. The [tracked audit](artifacts/noaa-range-probability-summary.json)
records direct recomputation and byte-for-byte replay.

The [seasonal follow-up](noaa-range-probability.md#seasonal-calibration-follow-up)
found that April and May accounted for about 78% of excess same-day Brier loss.
Season matching narrowed the gap on saved dates. In the frozen July 1–10
pilot, both full-year methods scored better than their season-matched
counterparts. The [curated summary](artifacts/noaa-range-seasonal-summary.json)
retains the diagnostic and replay evidence. The next evaluation uses the
already frozen July 11–December 31 plan once the archive is complete. Keep
event scoring at the study level.

The observed warm bias and quiet-day errors remain separate model questions.
Workplace testing is deferred. Revisit rare-jump integration when a concrete
task needs it, using the saved accuracy and runtime checks to bound the scope.

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
| `runs/noaa-range-seasonal-2026-july-pilot-plan/` and `runs/noaa-range-seasonal-2026-july-pilot/` | Frozen seasonal method, preserved July archive, four matched forecasts and independent audit |
| `runs/noaa-range-seasonal-2026-h2-plan/` | Frozen 174-date seasonal evaluation design and copied training inputs |
| `runs/noaa-range-2026-h1-plan/` and `runs/noaa-range-2026-h1/` | Frozen 2025 inputs and design, preserved 2026 archive, matched scalar runs, paired range probabilities and scores, and replay audit |
| `runs/noaa-temperature-pair-verified-20261119/` | NOAA extrema archive, normalized minimum snapshot, matched point forecasts, error and range diagnostics, and the paired chart |
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

Read the [agent handoff](agent-handoff.md) for changes since `c3fe4ff`, then
this note, and inspect the working tree. The
[range study guide](noaa-range-probability.md) links the completed result and
explains how to replay it. Keep the frozen maximum run
under `runs/noaa-2025-frozen-20261119/`. The [pair study guide](noaa-temperature-pair.md)
explains how to reproduce the files if the local run directories are missing.
Run new experiments explicitly. Configured CI runs automatically on branch
pushes and pull requests.

```bash
git status --short --branch
PYTHONPATH=src .venv/bin/python scripts/check_documentation.py
```

The verified local Python 3.12 interpreter is
`runs/workplace-py312-20261119/profile-venv/bin/python`. Use the
[pinned profile](python312-public-forecast.md#set-up-the-verified-profile)
to recreate it when needed.

For future changes, update the affected docs and run the relevant checks. Use
the [test guide](../tests/README.md) and [CI guide](continuous-integration.md) for
verification commands and writable cache settings.
