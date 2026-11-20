# Roadmap

Updated: 2026-11-19. This page records implemented scope and review priorities.
For usage, start with [the walkthrough](forecasting.md). For past choices, use
[the decision history](decisions.md); historical proposals are not build checklists.

## Resume here

The [research checkpoint](https://github.com/aquinjay/model-pipelining/pull/7)
is merged. The first concrete public task forecasts next-day maximum
temperature at Central Park. On Python 3.12.16, the Gaussian model and
last-value benchmark completed all 366 matched development cases with four
direct packages. Annual seasonality plus one lag improved on the 2024 dates and
beat both controls on overall MAE and RMSE across the 365 matched 2025 dates.
The later evaluation used the frozen model and scoring settings.
Real-time use needs publication dates and vintages. See the
[profile](python312-public-forecast.md), [seasonal comparison](noaa-seasonality.md),
and [later-year evaluation](noaa-later-period.md).

The [temperature pair study](noaa-temperature-pair.md) matches separate minimum
and maximum forecasts at the same 365 origins. The seasonal pair had zero
crossed point forecasts and lower daily-range error than persistence. Shared
errors motivate a joint distribution for a task that needs probabilities about
the range. The study uses the scalar runner and reports the pair after fitting.
The [next experiment](next-session.md#next-work) compares independent and
date-matched residual pairs for a range probability on later dates. It uses
the same point forecasts and freezes the 2025 calibration pairs before scoring.

Bayesian dynamic linear models are the first main use case within a broader
forecasting project. Keep the first integration focused on the capabilities it
needs. Use concrete examples from harmonic regression, other state-space models,
and multivariate forecasting to guide future changes to shared interfaces. A
small harmonic regression example is a candidate for checking that a different
model can use the same data selection, evaluation, and reporting. See the
[reading guide](reading-guide.md) and [model boundary](architecture.md#implemented-model-boundary).

The state-grid importance proposal passes the probability criteria on all 17
cases. An independent Gibbs calculation also agreed on the moments and
predictive probes for `abrupt-42`. Repeated-fit cost favors the existing Gaussian
local-level adapters for the first BDLM workflow. A small harmonic regression
example already exercises the shared forecast contract across 72 cases.
See the [verification results](innovation-moment-check.md) and
[next work](next-session.md#next-work).

The three planned interpretation tasks are complete. The [development study](model-review-study.md)
records nine compatible reference reviews, 45 matched forecasts, all declared
prior alternatives, 3,000 fresh calibration trials and the next bounded modeling
question. Read its results and [the updated checkpoint](next-session.md) when
resuming. The [short-history innovation reference](innovation-reference.md) is
also complete: independent batch/adaptive and filtering/quadrature calculations,
all three Gaussian limits and an explicit eight-observation cap. The
[longer-history particle prototype](innovation-particles.md) now implements
independent-filter MCSE, evidence weighting and refinement. Its oracle checks
expose rare-path uncertainty failures that must be resolved before extending
forecasting/evidence interfaces.

The study also exposed and repaired a numerical joint-V quantile bracket
failure, retaining its original trial. That checkpoint passed 401 tests. The
merged research suite passed 449 tests. The NOAA example adds three input
validation tests. The focused Python 3.12 verification passed 80 targeted tests.
The study and repair are recorded on `feature/innovation-inference-study`,
based on merged checkpoint `0ff5def`. Hosted CI and review belong to the PR
stage.

Step 1, the computational review of the current modeling workflow, is complete.
The [review record](workflow-review.md) explains one concrete forecast, saved
checks, exact replay, and the remaining assumption/evidence gaps. All 118 tests
passed, and both models completed all 170 cases on both synthetic series.
Human model approval remains pending.

Step 2 now has an explicit [local-level diagnostic workflow](diagnostics.md):
caller-chosen prior/noise influence paths, conditional initial-mean weights,
sequential innovation/PIT criticism, saved failures, and pending human review.
The user clarified that sensitivity should reveal assumption influence, not
minimize it. Strong influence is not a computational failure; no magnitude or
configuration-agreement threshold was added. The workflow is separate from the
backtest and introduces no new dependency.

The user clarified that infrastructure, rather than application selection, is
the current goal. Step 3 now supplies [reusable experiment specs](experiments.md):
target/source descriptions and columns/units, issuance and as-of release/revision
selection, explicit daily calendar/gap validation, required benchmarks on matching
cases, and separate development/final target periods. The example uses known
synthetic fixtures; it does not choose a real application or certify untouched data.

The [parameter-uncertainty detour](parameter-uncertainty.md) is implemented:
`NormalMean` learns a constant mean, computes exact marginalized observation
forecasts, and optionally audits each fit with independent emcee Metropolis
chains and ArviZ diagnostics. Optional fit-evidence dispatch saves posterior,
chain/draw, warmup, seed, transform, precision/reference checks, and partial
failures before scoring. Required audit failures fail cases/runs. Both modes
use exact forecast integration; MCMC-only forecast/HPD estimation remains open.
Prior influence remains descriptive, not a quantity to minimize. Parameter
learning now also includes an unknown observation variance:
[NormalMeanVariance](normal-mean-variance.md) uses normal/inverse-gamma updates,
exact Student-t forecasts and an optional joint sampler audit. Heavy-tail moment
existence and the transform Jacobian are explicit. [Local-level observation
variance](local-level-variance.md) is now learned with checked scalar integration;
the scalar model keeps evolution variance fixed. [The joint model](joint-variance.md)
learns categorical evolution probabilities. Application selection and a broader
release/check framework can wait.

[Prior and posterior predictive checks](predictive-checks.md) now replicate
whole datasets from the exact joint prior/posterior of `NormalMeanVariance`.
Prior-only runs need no input. Observed checks use an explicit as-of development
history with release/vintage selection and a final-boundary guard. They save
location/spread/tail/dependence/level-change discrepancies and their simulation
precision; human interpretation stays pending, with no discrepancy failure
threshold.

[Prior influence comparisons](prior-influence.md) now compare explicit
mean-location/precision and variance-shape/scale alternatives from the
empty-history prior through matched historical as-of updates. Paths save exact
mean/variance/predictive summaries, scale components and signed reference
differences. Later revisions cannot rewrite earlier updates. Strong influence
and undefined moments remain explicit evidence, not computational failure.

To understand or repeat the reviewed workflow:

1. Follow [the workflow](forecasting.md) to see how a preserved input becomes
   two repeatable runs and saved reports.
2. Review [the local-level model](local-level.md): its fixed settings, initial-prior
   timing, forward calculation, and independently checked results.
3. Review [shared uncertainty](uncertainty.md): next-observation semantics, HPD
   priority, capabilities, frozen outputs, compatible evaluation, and failures.
4. Use [the API reference](../src/model_pipelining/README.md) and
   [test map](../tests/README.md) to inspect exact interfaces and supporting checks.

Implementation acceptance means correct calculations, no future data in model
requests, explicit failures, and reproducible saved results. Better scores,
faster reactions, or a particular empirical interval coverage are not acceptance
thresholds. Human interpretation of assumptions and evidence remains open.

The baseline workflow was introduced in `eabd66e`. PR #1 merged unknown-mean
parameter learning and its sampler audit; PR #2 merged joint mean/variance
learning. PR #3 added prior/posterior predictive checking and prior influence
comparisons; the current checkout adds learned local-level observation variance
and its forecast/check/comparison workflows. Use Git status/history
to distinguish local edits from committed/published code.

The subsequent [evidence consolidation](evidence.md) implements model-supplied
simulation/summary providers and version-2 experiment evidence requirements.
The explicit evidence runner executes requested existing workflows on one pinned
history; forecast execution independently enforces caller-required fit names.
No new model, sampler, application or automatic human decision is introduced.

## Implemented scope

| Area | Implemented | Detailed reference |
|---|---|---|
| Environment | CPython 3.14, development pin 3.14.10, uv lock and pip export, unittest | [Setup](../README.md#development-environment) |
| Synthetic sources | Two seeded baseline 200-day series and a shared-noise gradual-change review fixture | [Data](../data/README.md), [development study](model-review-study.md) |
| Ingestion | Local copy receipts and CSV/XLSX table reading | [Ingestion](ingestion.md) |
| Preservation | One-file SHA-256 snapshots, verified retrieval, repeat registration events, optional receipt context | [Snapshots](snapshots.md) |
| References | Dataset-scoped explicit tags, saved assignment history, permanent run input IDs | [Tags](snapshots.md#assign-resolve-and-inspect-tags) |
| Execution | Daily model-independent experiments, release-aware expanding histories, one fixed positive-day horizon per run; defaults preserve the original next-day task | [Experiments](experiments.md) |
| Models | Training mean and last-value point models; fixed-variance Gaussian local level with exact filtering | [Model guide](local-level.md) |
| Parameter learning | Unknown constant Gaussian mean with known or learned observation variance; exact normal/Student-t predictions, optional independent-chain audits and saved fit evidence | [Parameter uncertainty](parameter-uncertainty.md), [joint model](normal-mean-variance.md) |
| Uncertainty | Observation quantiles and highest-density regions; disconnected regions supported by the shared format | [Uncertainty](uncertainty.md) |
| Evaluation | Point errors, quantile/equal-tail scores, separate HPD coverage/total width, explicit case completion | [Uncertainty evaluation](uncertainty.md#validation-and-evaluation) |
| Reporting | Saved-only reviews, charts and matched-case comparisons with full-law losses; HPD preferred; old point artifacts readable | [Run files](../runs/README.md) |
| Model evidence | Explicit full-path prior/noise influence, conditional initial-mean weights, sequential criticism, saved-only diagnostic charts | [Diagnostics](diagnostics.md) |
| Predictive checks | Data-free prior and full selected-history posterior replications for the joint mean/variance model; descriptive discrepancies, finite-simulation precision and saved-only charts | [Predictive checks](predictive-checks.md) |
| Prior comparisons | Exact learning/predictive summaries, matched as-of release histories, variance-scale decomposition and signed differences for caller-chosen priors | [Prior influence](prior-influence.md) |
| Dynamic parameter learning | Learned observation variance, fixed or categorical evolution variance, checked quadrature/mixture forecasts and dynamic evidence | [Scalar model](local-level-variance.md), [joint model](joint-variance.md) |
| Innovation reference | Fixed small/large transition variances and probability, all-sequence integration through eight observations, independent conditioning/integration checks | [Bounded reference](innovation-reference.md) |
| Evidence requirements | Provider-based calculations, caller-required fit names/workflows, explicit pinned-history execution and failure receipts | [Experiment evidence](evidence.md) |
| Task/evaluation contracts | Version-2 task/evaluation/evidence specs (version 1 readable); manifest version 3, target roles, saved case exclusions, selected-history hashes, explicit actual vintage, required benchmarks | [Experiments](experiments.md) |

## Verification at this checkpoint

The merged checkpoint passed 391 tests. The short-history reference checkpoint
passed 415 tests, and the latest local checkpoint passed 449.
[The test guide](../tests/README.md) maps them
to behavior: data preservation, independent posterior/predictive calculations,
requested uncertainty validation, disjoint region membership/display, future
isolation, frozen-file replay, legacy compatibility, and failure handling.
The 12 diagnostic tests include independent batch-conditioning influence
calculations, strong persistent prior influence completing successfully,
prior/noise distinctions, saved failures, future isolation, replay, and CLI
scenario inheritance. Both synthetic series were exercised with the reference
and four illustrative alternatives over all 200 observations.

The 16 experiment tests cover delayed releases and revisions, distinct scoring
vintages, available-period counts, target-based development/final separation,
matching benchmark cases, all-forecast freezing, visible benchmark failures,
arbitrary positive-day horizons and adapter rejection, experiment replay,
history-identity comparison, version-2 reporting, and both CLI entry points.
The 200-day fixture also exercised both models with 130 development and 40
final-role cases, each with both point benchmarks; the Bayesian development
run's numerical artifacts replayed exactly. Final-role results are infrastructure
evidence on a known fixture, not an untouched scientific evaluation.

Both models were exercised on both existing synthetic series, with 170 common
forecast dates per series. The uncertainty run was replayed exactly, including
quantiles, HPD segments, evaluations, and scores. The documentation walkthrough
is also checked in temporary storage. These observations establish the stated
local behavior, not calibration or approval of a real-data model.

The local-level filter still uses the standard library. The parameter-learning
slice declares NumPy, SciPy, emcee and pre-1.0 ArviZ explicitly, with regenerated
lockfile/pip export. CPython 3.14.10 imports, installed compatibility, and the full
suite passed. The 28 new tests cover independent quadrature, exact HPD semantics,
prior influence, sampler target ratios and posterior/predictive agreement,
biased/stuck/missing diagnostics, partial backend failures, global-RNG isolation,
fit persistence/freezing/replay, delayed releases/final roles, both inference CLI
modes, and saved-only traces. Human approval remains pending.

Saved local exercises under `runs/parameter-uncertainty-final-20261114/` completed
170 exact cases on each synthetic series. Each series also completed five
explicit final-role audit cases at target indices 100–104, with both point
benchmarks, and an exact matched-cohort run. Every fit/draw and numerical forecast
file replayed exactly; exact/audit forecasts and scores matched. The largest
R-hat was 1.003979 and smallest bulk/tail ESS were 1550.38/1329.21 across the
exercised diagnostics. Saved-only charts and comparisons succeeded. These known
fixtures demonstrate mechanics, not untouched final evaluation or scientific
model approval. Generated artifacts are local and ignored by Git.

The joint model adds 20 tests; all 194 tests passed. Both seeded synthetic
series completed 170 exact forecast cases, plus three explicitly selected
final-role audit cases at target indices 100–102 with both point benchmarks.
Exact/audit predictions and scores matched, and all numerical fit/draw and run
files replayed byte for byte (excluding manifest timestamps). Saved-only charts
and comparisons succeeded. Evidence is local under
`runs/variance-learning-20261117-t2f9oedw/`. Across those joint audits, maximum
R-hat was 1.005306 and minimum bulk/tail ESS were 2638.73/3362.73. These known
fixtures demonstrate mechanics, not untouched final evaluation or approval of
the constant-mean assumption. No new dependency was added.

The predictive-checking slice adds 27 tests. All 221 tests passed. Independent
joint moments, cross-position covariance, predictive probabilities, discrepancy
arithmetic, undefined quantities, partial failures, release/final isolation,
seed/replay behavior, saved-only plots and both CLI modes are covered. A
data-free prior check and checks on both synthetic series each generated 1000
replications per phase. Observed runs used the first 160 development values,
with the remaining 40 excluded, and all numerical files replayed exactly.
Artifacts are local under `runs/predictive-checks-20261117-te2natcp/`.

For the stable series, posterior upper-tail fractions for lag correlation,
half-mean difference and CUSUM were 0.517, 0.233 and 0.465. For the shift series,
all three had zero exceedances out of 1000 (Wilson upper bound about 0.00383).
This exposes temporal structure poorly reproduced by the constant-mean model,
while computation completes and human review remains pending. These are
descriptive development observations, not approval criteria or calibrated tests.

The prior-comparison slice adds 19 tests; all 240 tests passed. They cover hand updates/scale components,
counterfactual prior-location coefficients, independent interval mass checks,
variance concentration with fixed prior expected variance, persistent influence,
undefined moments, early paths, historical releases/revisions, final isolation,
shared selector validation, predictive-check endpoints, partial reference/update
failures, replay/order/tag behavior, saved-only charts and CLI inheritance.
Both synthetic series exercised six configurations over the first 160 values,
with 162 points each including the prior, daily as-of updates and explicit
cutoff. Six numerical files per series replayed exactly; human interpretation
remains pending. Artifacts are local under
`runs/prior-influence-20261117-ngp8vopg/`.

For the shift fixture at the cutoff, the reference and concentrated-variance
priors had the same mean location (11.966468) but predictive HPD widths 10.255558
and 9.105625. The stronger mean prior retained a location coefficient 0.157895
and mean location 11.656603. These are descriptive consequences of the chosen
settings, not a ranking or a claim that influence should disappear.

The learned local-level slice adds 29 tests; all 269 tests passed. Independent
batch Gaussian likelihood/state calculations, adaptive scalar integration and
a known-level conjugate limit support the numerical claims. Tests also cover
mixture gaps and wide-tail root precision, smoothing for initial-state draws,
fresh process innovations, undefined prior moments, required precision failures,
replay, future/final isolation, and legacy/new CLI modes.

Both synthetic series completed 130 development forecasts on the first 160
values, with the final 40 excluded. All 143 numerical/fit files per series
replayed byte for byte, excluding manifest timestamps and PNGs. Each series
also generated 1000 prior and posterior fresh-path replications and compared
five priors across 162 points. Maximum accepted quadrature order was 1024 and
maximum omitted-tail probability estimate was 7.842e-9. Artifacts are local in
`runs/local-level-variance-20261117-bs3pk_xh/`.

Posterior upper-tail fractions for lag correlation/half-mean difference/CUSUM
were 1.000/0.480/1.000 on the stable fixture and 0.104/0.047/0.002 on the shift
fixture. These are descriptive model criticism, not calibrated tests or an
approval threshold. Computational completion leaves their interpretation open.

The evidence consolidation adds 22 tests, including a model unknown to shared
workflow code, missing/duplicated replication output, malformed summaries,
missing mandatory checks, caller-required fits, unavailable capabilities, tag
movement between workflows, version migration/comparison, replay/final isolation,
legacy saved-only charts, I/O and both CLI modes. The full suite has 291 tests.
On both original synthetic histories, the consolidated workflows reproduced
18 saved evidence files byte for byte, 324 learning rows and both selected prior
summary records. Artifacts are local under
`runs/evidence-interface-20261117-fewaws1h/`; the evidence plan used 1000 prior
and posterior replications and two priors over 162 learning points each.

## Earlier priorities and remaining design choices

The November 19 [development review](model-review-study.md) completes the
collected-evidence interpretation below. The dated sequence remains useful
context; current work should follow the bounded question at the top of this page.

The [completed modeling review](modeling-review.md) follows implementation and
verification of the three learned local-level tasks. It finds the work aligned
with repeatable, model-independent forecasting infrastructure while retaining
the synthetic discrepancies as open modeling evidence.

- **Evidence consolidation completed:** [the provider/experiment boundary](evidence.md)
  now separates model calculations from shared workflow responsibilities. Caller-owned
  fit names and required separate workflows fail visibly when evidence is missing.
  Further abstraction should follow a concrete need rather than adding a registry
  or another backend by default.
- **Review collected evidence completed:** prior predictive implications,
  posterior discrepancies, prior influence and windowed losses are interpreted
  together in the development study. The small dynamic-model CRPS ordering
  changes under the larger-V prior, while the shift-pattern discrepancy persists.
  A reference prototype should investigate that concrete question; current
  models remain infrastructure benchmarks.
- **Further model evidence:** extend whole-dataset predictive checks to another
  model only when useful. An arbitrary-stage configurable check plan, exploratory
  overrides, and application-specific human/release criteria remain open.
- **Further parameter learning:** observation variance is now learned in both
  constant-mean and evolving-level models. The joint model also learns an explicit
  categorical evolution variance. Continuous evolution priors or new transition
  laws require their own reference and evidence design. Generic
  MCMC-only prediction still needs an explicit precision/HPD design; the new
  local-level model uses controlled scalar quadrature rather than sampled forecasts.
- **Real-data application (deferred):** fill the implemented task/evaluation
  contract with an actual source and verified historical release claims when
  application choice becomes useful; it is not the immediate project objective.
- **Another adapter:** use a concrete forecasting example to exercise a different
  set of assumptions and capabilities. Harmonic regression is a candidate for
  seasonal predictors. Multivariate forecasting will also need explicit series
  alignment and a way to express dependence when a model returns a joint forecast.

The Bayesian adapter was selected before the real-data application. This is an
order of learning tasks, not a requirement to finish every possible storage or
modeling feature first.

## Later, when a need is demonstrated

Multi-file snapshots, processing lineage, richer acquisition provenance, remote
sources/storage, concurrent writers, crash recovery, backup/retention automation,
multiple simultaneous horizons, richer calendars, tuning, a run index, dashboards, scheduling,
and monitoring remain possible extensions. git-annex is optional; DVC or another
execution tool is unselected. Neither is a prerequisite to reviewing the current
working path. See [the historical storage proposal](data-versioning.md) for the
broader requirements and alternatives that have been retained for reference.

## Completed validation and dynamics sequence

All five authorized slices are implemented and independently committed:

1. [Sequential learned-variance checks](sequential-checks.md): frozen historical laws, full-CDF PIT and residual dependence.
2. [Simulation calibration/recovery](calibration.md): data-free exact/numerical posterior checks with retained failures.
3. [Dynamics investigation and joint variance learning](joint-variance.md): controlled stable/gradual/abrupt comparisons and explicit categorical W uncertainty.
4. [Joint generative criticism and prior influence](joint-evidence.md): dependent W/V/state draws, fresh paths and fixed-support prior comparisons.
5. [Mixed calibration](mixed-calibration.md): genuine atom randomization, interval mass and known-parameter exclusion.

That sequence finished with 343 tests. Numerical computation is verified separately from model
adequacy and finite-simulation interpretation. Further model changes should follow
the saved discrepancies; continuous-W priors, different transition laws, a real
application and release automation remain separate decisions.

## Experiment review and comparison sequence — 2026-11-18

The first slice, [saved experiment review](experiment-review.md), is implemented:
explicit compatible sources, receipts/requirements, omissions, calibration and
chart links in one report, with human interpretation pending. Eighteen added tests
bring the suite to 361. A 130-case development exercise assembled all three
workflows and two small calibration runs without changing source files.

The authorized sequence adds full predictive distribution scores and then CI
for the locked environment, tests and documentation checks. Large calibration
and scientific model/release decisions remain explicit caller actions.

The second review/comparison slice, [full-law proper losses](distribution-scoring.md),
is implemented: Student-t and full Gaussian mixtures, frozen laws, visible
failures and complete matched saved comparisons/charts. All five Bayesian adapters
supply laws from the same fit. Twenty-one new tests bring the suite to 382.
No richer model or automatic selection rule was introduced.

The third slice, [repository CI](continuous-integration.md), is implemented with
pinned actions/uv/Python, lock/export checks, the complete suite, documentation
links/anchors/example syntax and temporary walkthrough execution. Nine added
tests bring the suite to 391. The fresh locked environment and local workflow
linter passed. Hosted push and PR checks passed before [PR #6](https://github.com/aquinjay/model-pipelining/pull/6)
merged into main at `0ff5def`.

All three authorized review/comparison tasks are complete. The next bounded work
is to interpret one collected experiment review, stating the remaining predictive
discrepancies and what a proposed model change would need to explain. Further
model complexity, continuous-W priors or MCMC-only prediction should follow a
concrete unmet need in that evidence. Current progress supports the broader goal:
repeatable, model-independent experiments with comparable saved results and
explicit computational versus human review.

## Completed development interpretation — 2026-11-19

The [study](model-review-study.md) completes the three planned review tasks:
nine ready-for-review reference combinations, all 45 declared forecasts, 27
required historical workflows, 3,000 current calibration trials and 36
alternative-prior predictive exercises. It retains the original failed numerical
trial and the repair, compares fixed development windows and documents a bounded
transition question. Curated summaries/figure are tracked; full source evidence
and validation receipts remain local. The 401-test suite and documentation
walkthrough pass locally. This checkpoint selects a reference investigation and
its validation requirements; it does not approve or implement the candidate model.

## Bounded innovation reference completed — 2026-11-19

The [reference prototype](innovation-reference.md) completes the short-history
investigation: every positive-prior sequence, independent batch/adaptive and
filtering/quadrature calculations, direct/exact references and Gaussian limits.
Seven frozen cases agree within 4.78e-11 and replay exactly; all three existing-
model limits agree within 3.90e-12. Fourteen new tests bring the suite to 415.
The eight-observation cap and 2**160 nondegenerate path count make the remaining
feasibility question explicit. A controlled longer-history approximation comes
next, with error evidence against this oracle before forecasting/provider
integration and broader criticism/calibration. No longer-history algorithm or
MCMC backend is selected here.

## Longer-history particle investigation — 2026-11-19

The [particle prototype](innovation-particles.md) implements the next feasibility
step: sampled shared V/labels, analytical Gaussian states, independent-filter
evidence weighting, joint moments, jackknife MCSE, particle-count refinement and
saved ancestry diagnostics. Prior and Gaussian limits use native checked
quadrature automatically; known-state atoms and fresh future probabilities stay
explicit. Sixteen added tests bring the locally passing suite to 431.

The frozen study compares eleven short/limit/atom cases with the independent
oracle and investigates six matched 160-observation histories. Rare unvisited or
underrepresented paths expose overconfident empirical envelopes even when the
reported aggregate precision gates pass. The failed checks remain part of the
evidence for the conservative probability follow-up below.

## Conservative probability investigation — 2026-11-19

The [independent probability check](innovation-probability-bounds.md) implements
an analytic importance-weight envelope and simultaneous finite-sample bounds for
label/path, state/V and full predictive CDF probes. Unvisited positive-mass events
retain a positive uncertainty radius; a normalizer with a zero lower bound
leaves explicitly unresolved intervals. All 872 short-oracle probability checks
are enclosed, but three short and all six long cases fail useful precision at
the fixed 262,144-draw budget. Forecast/provider integration remains gated.
The audit verifies ten source and 31 input fingerprints, recalculates every
interval/check and replays all 17 independent calls exactly. Eleven added tests
bring the suite to 442; passing software checks do not resolve wide bounds.
The subsequent [state-grid proposal](innovation-lattice.md) passes the
probability criteria on all cases while retaining the original comparators.
