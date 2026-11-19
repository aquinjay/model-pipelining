# Checkpoint and next session

Updated: 2026-11-19. Start here when resuming the modeling work, then consult the
[roadmap](roadmap.md) and [modeling review](modeling-review.md).

## Where we left off

The project is developing repeatable, model-independent forecasting experiments:
preserved inputs, release-aware histories, explicit assumptions, comparable saved
results and separate computational checks and human interpretation. Application
selection remains deferred. The evidence-interpretation phase is complete;
the [three-model development study](model-review-study.md) records results,
prior robustness, limitations and the next bounded modeling question.
The [short-history innovation reference](innovation-reference.md) and
[longer-history particle investigation](innovation-particles.md) are implemented.
The [conservative probability follow-up](innovation-probability-bounds.md)
enclosed all 872 short-oracle probabilities across 11 cases and replayed all
17 runs exactly. Three short cases and all six 160-observation cases still fail
the probability precision criteria. The current code checkpoint passed 442
tests and the documentation walkthrough locally.

The `feature/innovation-inference-study` branch records the development review,
numerical repair, innovation prototypes, tests and curated evidence. It starts
from merged checkpoint `0ff5def`. Review and hosted CI belong to the PR stage.
The research checkpoint preserves the failures for comparison in the next
pilot. Forecast integration requires independent precision verification.
Full run directories are ignored by Git.

The last merged implementation checkpoint is recorded on `main` through
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

The current study and numerical repair are local work after that merged
checkpoint. No new commit, push, PR or hosted CI result is implied.

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

1. Prototype an importance proposal that uses the observations to concentrate
   draws on plausible shared observation variances V and jump paths. Derive an
   explicit weight bound and preserve support for every positive-mass path.
   Targeted integration remains an alternative if the weight bound stays loose.
2. Start with `short-abrupt`, `short-dominant` and `abrupt-42` from the frozen
   study. Compare the two short histories with their independent oracles. On
   `abrupt-42`, compare with the retained particle output and check the new
   bound's precision. The retained particle output is an empirical comparison.
3. Use the existing probability criteria for the pilot: a positive lower bound
   on the normalizer and maximum interval width of 0.10 at the declared confidence
   and budget. If the pilot meets them, run all 17 saved cases, including the three
   paths that failed the original particle envelopes. Keep the model, histories
   and query grids fixed for comparison. Retain failed checks and preserve the
   Gaussian limits, state atoms, V/path joint moments and full predictive CDF and
   density checks. Recheck runtime and ancestry if particle inference changes.

After the approximation is independently verified, extend forecasting/provider
interfaces and run fresh-path predictive criticism, matched prior comparisons,
calibration/recovery and independent-seed stable/gradual/abrupt studies. The
current prototype establishes a short-history calculation, not those scientific
adequacy results. Broader adapter diversity and real-data application choice
remain separate project decisions.

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
- All **442 tests pass**, including eleven new probability-bound tests. The
  three original failed envelopes and the exploratory bound runs are retained.

## Saved local evidence to inspect

These paths existed at this checkpoint. Generated run folders are ignored by Git
and may be absent in another checkout; the tracked guides explain reproduction.
Keep source folders with their reports and inspect their manifests before reuse.

| Local path | What it contains |
|---|---|
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
