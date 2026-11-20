# Decisions and open questions

The [independent moment and density check](innovation-moment-check.md) agrees
with the state-grid approximation on `abrupt-42`. Use the existing Gaussian
local-level adapters for the first BDLM workflow and keep external modeling
frameworks optional at the adapter boundary.

This is a chronological design history. A dated entry records what was decided
at that time; later entries can supersede it. Use [the README](../README.md),
[the API reference](../src/model_pipelining/README.md), and [the roadmap](roadmap.md)
for current behavior and review priorities.

## Current choices at a glance

- Preserve one local file per snapshot, identified by dataset plus SHA-256.
- Keep acquisition, registration, tag assignment, and execution separate calls.
- Run point models, Bayesian local-level adapters with fixed or learned observation variance, constant-mean Gaussian adapters with known or learned observation variance, or the weekly harmonic regression adapter through shared daily experiments, with adapter-specific timing limits. Every repository adapter rebuilds from its saved configuration; scripts construct models through one catalog that the runner never consults.
- Save a predictive mean and each declared observation-uncertainty kind. Prefer highest-density prediction regions in reporting.
- Keep failed cases visible, human interpretation separate, and data/run evidence local.
- Treat prior/noise sensitivity as descriptive influence evidence; strong influence does not fail computation. Save explicit local-level diagnostics separately from backtest results.
- Declare target/timing, release/revision selection, benchmarks, and evaluation roles in versioned experiment specs; defer application selection. Unknown-mean parameter integration and optional independent-chain MCMC audits are implemented.
- Use model-supplied evidence calculations and experiment-owned fit names/separate workflow requirements; execute separate evidence explicitly and retain failed or unavailable requirements.
- Use CPython 3.14 with pandas, openpyxl, matplotlib, NumPy, SciPy, emcee and pre-1.0 ArviZ for research. Keep a separate Python 3.12.16 workplace profile (pandas, NumPy, SciPy, matplotlib) for CSV Gaussian workflows. CI checks both on every push. Installable packaging, arbitrary-stage checks, and release automation remain open.
- Use public NOAA daily data with availability simulated from archive date labels. Freeze model and scoring settings before downloading evaluation data, and leave frozen study scripts unchanged so their plans can replay.
- Use the packaged temperature regression and NOAA helpers for new studies and record their source hashes in study designs.
- Define benchmark and workflow names once, and write saved JSON/CSV through one writer module.

## How to read earlier entries

The empty dependency list, multi-file identity, undecided naming, and deferred
Bayesian/uncertainty interfaces in early entries describe earlier checkpoints.
They do not override the current implementation. The November 3 proposal is
retained in [the historical storage design](data-versioning.md).

## Initial direction — 2026-11-01

| Decision | Reason |
|---|---|
| Begin with time-indexed numerical forecasting | Gives the first implementation a concrete, bounded task |
| Start on synthetic data with a baseline | Exposes timing and artifact mistakes before model complexity |
| Separate applications, model adapters, and shared execution | Changing model families should not require rebuilding the pipeline |
| Share checking stages/results and let models supply specialized checks | Supports Bayesian workflow and other model families without false equivalences |
| Experiment owns check requirements | Models must not silently bypass required evidence |
| Preserve frozen predictions, provenance, failures, and iteration lineage | Makes comparisons and revisions inspectable |
| Keep documentation independent of the Obsidian vault | A GitHub reader should understand the project without local paths or plugins |

## Environment decisions — 2026-11-01

- **Interpreter:** standard CPython 3.14, supported range `>=3.14,<3.15`, with 3.14.10 as the shared development pin.
- **Portable environment:** standard `venv` and pip are supported. uv is an optional local tool for interpreter management, resolution, and environment synchronization; the pipeline will not require it at runtime.
- **Version declarations:** project metadata and direct dependencies live in `pyproject.toml`. The project starts at version 0.1.0, with no stable public API promised yet.
- **Reproduction:** commit `.python-version`, `uv.lock`, and the generated `requirements.txt` alongside `pyproject.toml`. Export all selected dependency groups for the initial development environment. A pip-only machine consumes that export; it does not need uv.
- **Dependency starting point (superseded):** the project initially had no selected packages. pandas/openpyxl were added for table reading and matplotlib for reporting in later slices.
- **Packaging deferred:** this is currently an environment-managed project with no installable local package. The uv-specific `package = false` setting prevents attempts to install the placeholder source directory. Standard pip setup installs exported dependencies only.
- **Work portability:** a different interpreter or platform must be checked deliberately. The current range records our development target, not a guarantee about an unknown work deployment runtime.

## Data versioning decisions — 2026-11-03

The agreed direction, alternatives, open details, and ordered continuation plan were saved in [Data versioning: decisions and next steps](data-versioning.md). No versioning code existed at that point. The narrower implementation selected later is recorded below.

The November 3 proposal considered snapshots of one or more related files, compared file hashes, and recorded acquisition context. The later single-file implementation supersedes its collection identity and does not implement its full lineage/override contract. The proposal also required explicit dataset-scoped tags, pinned run input IDs, processing lineage, persisted quality evidence, experimental overrides, and replaceable storage destinations. Tags/input pinning are implemented; the current references identify which broader requirements remain planned.

## Ingestion first; preservation and execution later — 2026-11-09

- **Earlier preservation tool direction (superseded below):** use git-annex for preserving and retrieving versioned file contents. git-annex has not been installed or configured for this project.
- **Execution tooling deferred:** consider DVC or another tool for dependencies, stages, execution, and caching when the runnable workflow needs it. DVC is a candidate, not a selected dependency. Avoid duplicating execution responsibilities between tools and the project runner.
- **Immediate priority:** begin a small library of ingestion functions with one function for a local file and a companion CSV/XLSX table reader with worksheet selection by name or index. Functions for Snowflake, S3, SFTP, and other sources can follow as concrete needs arise. Acquisition remains separate from parsing, snapshot registration, validation, and model execution.
- **Local implementation:** [Local file ingestion and table reading](ingestion.md) describes the implemented interface and behavioral checks. pandas and openpyxl are declared, locked, and exported for table reading. Standard-library unittest is sufficient for this slice. The package is imported through `PYTHONPATH=src`; installable packaging is deferred.
- **Sequence change:** this supersedes the earlier instruction to implement the full local snapshot/tag system before the next code slice. The snapshot identity, provenance, lineage, and validation requirements above remain future requirements; the first ingestion receipt does not satisfy them by itself.

## Small local snapshot implementation — 2026-11-09

- **Immediate scope:** implement the [small local snapshot workflow](snapshots.md) for user review before adding more features. It registers one local file, preserves its bytes, records each successful registration, and retrieves it with an integrity check.
- **Tooling:** use standard-library `hashlib`, local files, and JSON. This supersedes the earlier selection of git-annex as the next preservation integration. git-annex remains an optional later integration; DVC or another execution tool is still deferred.
- **Identity:** a reference is a dataset name plus the SHA-256 of the file bytes. Matching bytes reuse a snapshot within a dataset; each registration gets a distinct event record. Renamed identical inputs retain the first preserved filename. Unlike the older multi-file proposal, this first single-file identity does not include the filename. A multi-file contract remains future work.
- **Recorded time:** `registered_at` is when registration records the local file; it does not claim the original acquisition time or historical availability. The input file must remain unchanged during registration.
- **Inspection and notes:** `list_snapshots` returns metadata sorted by ID; `list_acquisitions` returns all registration records sorted by time and ID. Both return lists of plain dictionaries for one dataset without modifying storage. Registration accepts an optional string `note` stored in the event, independent of snapshot identity. Older records without notes remain readable as `note=None`.
- **Initial boundary:** keep one writer at a time and all records/data local. Multi-file snapshots, processing lineage, data-quality checks, remote storage, and crash recovery remain saved enhancements. Basic tags were added in the later slice below. After the walkthroughs and documentation updates, the user requested commit and push of this slice on 2026-11-09.

## Explicit dataset tags — 2026-11-09

- **Operations:** `assign_tag`, `resolve_tag`, `list_tags`, and `list_tag_history` live in `src/model_pipelining/tags.py`, with [examples and limits](snapshots.md#assign-resolve-and-inspect-tags). Assignment is explicit and verifies the target in the same dataset. Registration never automatically advances tags.
- **History and publication:** each tag has one JSON file containing its ordered assignments; the final assignment is the current target. A completed staged file replaces the previous record so target and history publish together. One writer is required; crash durability is deferred.
- **Context:** every explicit assignment records old/new IDs, time, a caller-supplied `changed_by` person/process name, and an optional note. Reassigning the same ID still records the request. This is contextual history rather than authenticated or tamper-proof evidence.
- **Run references:** resolve a tag once and save the dataset plus permanent snapshot ID. Moving the tag does not change previously selected input IDs. Tagging does not run models or override required data-quality evidence. Deletion, renaming, and automatic tag movement are not implemented.

## Carry local ingestion receipts into registration — 2026-11-09

- **Selected change:** the user requested retaining the ingestion receipt with a snapshot registration. `register_snapshot` accepts optional `ingestion_receipt: IngestResult | None`, alongside the existing file path. Direct file registration remains supported.
- **Saved context:** the registration event contains an `ingestion_receipt` dictionary with `source`, `destination`, UTC ISO `acquired_at`, and `size_bytes`, or `None` when omitted. `source_path` still identifies the registered file, and `registered_at` remains separate from the earlier copy completion time. Neither timestamp establishes historical availability.
- **Checks and limits:** the receipt must describe the registered destination and byte count, with a timezone-aware acquisition time. These checks happen before creating the store. The original ingestion source need not still exist. Receipts contain no hash and cannot prove that same-size edits did not occur after ingestion; the acquired file must remain unchanged until registration.
- **Identity and compatibility:** receipt context belongs to each registration and changes neither snapshot IDs nor manifests. Repeated identical inputs can carry different receipts. The optional field extends event format version 1; listing older records returns `ingestion_receipt=None` without rewriting files. No new dependencies or storage framework are needed.
- **Evidence at that checkpoint:** all 57 behavioral tests passed, including retained context after source removal and store relocation, distinct registration/acquisition times, receipt-independent identity, UTC normalization, invalid receipts, and older records.

## Build shared execution around a training-mean model — 2026-11-13

- **Question and decision:** after tagging, the user selected building the first modeling infrastructure around a working simple forecast, on a new branch (`feature/first-forecast-workflow`). Start with the existing synthetic daily series and a training-mean baseline rather than designing a larger unused interface first.
- **Task:** predict tomorrow's `value` immediately after today's observation; 30 initial observations by default, then expanding history. Validate ordered/unique daily UTC timestamps and finite numeric values. Immediate availability without revisions is an explicit synthetic-data assumption, not inferred from ingestion timestamps. All forecast dates are development evidence; no untouched final evaluation period is reserved.
- **Boundary:** `models.py` supplies immutable observation/request records, a point-forecast protocol, and the baseline. A model receives eligible history, issuance, and target time, and returns a finite predictive mean. The runner owns scheduling, coordinates, validation, evaluation, and artifacts. Models refit from each request; retained training state, probabilistic capabilities, and adapter diagnostics are deferred. The interface assumes trusted Python code.
- **Data and evidence:** resolve any tag once, verify its snapshot, and save the permanent dataset/ID. Save JSON manifest/configuration/checks/scores/review and CSV predictions/evaluation. Close predictions before joining actuals for scoring. Per-origin model failures/invalid outputs and scoring overflow are retained; execution continues and coverage is explicit. A failed case fails the run. Invalid input/configuration raises before run creation. Fixed checks belong to the runner; configurable plans and release rules remain open.
- **Reporting:** draw a PNG using matplotlib from saved results, with no model calls or input-store access. The library makes reporting explicit; the baseline script calls it automatically. matplotlib is a runtime dependency, declared, locked, and exported alongside the existing table readers.
- **Limits and alternatives:** a general adapter/plugin registry, execution service, larger storage layer, tuning system, real-data task, and Bayesian adapter can follow demonstrated needs. Local run files remain writable; existing output directories/charts are rejected. Interrupted runs may retain `running` status. Source hashes and environment versions identify the implementation but do not restore it; callers preserve matching code, environment, and inputs. Human review stays pending even when checks pass.
- **Evidence:** known forecasts and scores, future-value isolation, tag/store changes, failed cases, overflow, and reporting from saved artifacts are covered by behavioral tests. See [the executable workflow and reproduction instructions](forecasting.md).

## First local-level Bayesian adapter — 2026-11-13

- **Question and scope:** the user selected implementing the planned Bayesian model through the existing pipeline before adding real data. This work began on `feature/local-level-bayesian`. The baseline was merged into `main` at `eabd66e`.
- **Acceptance criteria:** verify posterior calculations independently, enforce eligible-history isolation, and preserve reproducible settings/results through the existing model interface. The user clarified that comparative MAE and speed of response are secondary illustrations, not objectives or acceptance criteria. No parameter tuning or model selection is performed.
- **Model:** a Gaussian local-level model with caller-selected fixed observation/evolution variances and an explicit Gaussian initial prior. The prior is before the first daily transition; filtering and simulation use that same convention. Observation variance is positive; initial/evolution variances may be zero. Settings are finite, validated, immutable, and saved in the existing run manifest. No new dependencies or sampler are needed.
- **Initial boundary (expanded in the next entry):** `LocalLevel.forecast` initially returned tomorrow's predictive mean after refiltering its eligible history. The runner and saved point-forecast schema remain unchanged. `filter_history` exposes pre-observation moments, innovations, and posterior level moments for inspection; these are training diagnostics, not held-out runner forecasts. Internal variances are conditional on fixed settings; shared forecast uncertainty and parameter-uncertainty treatment are deferred.
- **Simulation and orchestration:** the adapter exposes a seeded prior simulator without using global random state. The library leaves simulation and reporting as explicit calls. `scripts/run_local_level.py` automatically simulates one 200-step trajectory before fitting, runs the existing backtest, saves the trajectory/seed/configuration with pending prior review, and draws a chart. Simulation is not an automatic prior-plausibility check or specialized runtime gate.
- **Comparison:** `compare_runs` reads saved evidence only, requiring matching permanent inputs, tasks, training coordinates, target dates, actuals, and complete scoring. It saves PNG/JSON comparisons without fitting, tuning, altering source runs, or selecting a winner. Failed or mismatched runs are rejected. Both synthetic scenarios were exercised to keep the illustration's scope clear.
- **Evidence and limits:** hand calculations, normal conjugacy, independent joint-Gaussian conditioning, seeded prior moments and latent-state errors, isolation/replay, and saved-report behavior are covered in tests. Human prior interpretation, specialized diagnostic requirements, prior sensitivity, posterior predictive criticism, and probabilistic forecast evaluation remain open. See [the model guide](local-level.md) and [mathematical references](references.md).

## Shared uncertainty with highest-density priority — 2026-11-13

- **Request and priority:** the user selected shared uncertainty outputs after the local-level model, then explicitly requested highest posterior density support as more important than other interval types. The continuation stayed on the same branch and preserved the pending Bayesian work. Acceptance remains correct semantics/calculations, information isolation, and reproducible shared execution; no tuning, ranking, or empirical-coverage target is introduced.
- **Target and boundary:** shared `Forecast` records carry a predictive mean, optional observation quantiles, and optional highest-density regions. Both uncertainty kinds target the future observed value in original units. They are not latent-state or parameter credible intervals. Adapters declare boolean capabilities, and the runner requires every supported output at caller-selected levels (default 80%/95%). Older numerical adapters remain point-only; `PointForecaster` is retained as an alias for the expanded `Forecaster` protocol.
- **Highest-density semantics:** density regions are adapter-supplied sets of one or more disjoint segments, not inferred from a few quantiles. Shared checks enforce finite/completed/nested set structure and preserve gaps. Each adapter remains responsible for probability mass and density optimality. The Gaussian adapter computes exact posterior-predictive regions; equal-tail and highest-density bounds coincide only because of that law's symmetry/unimodality. Highest-density regions are the primary display when available.
- **Bayesian calculation:** the next-observation normal variance is posterior level variance plus one evolution variance plus observation variance. Results remain conditional on fixed settings and omit variance-parameter uncertainty. `LocalLevel.forecast` now returns `Forecast`; direct callers use `.mean` instead of treating the result as a float. `predictive_distribution` exposes the conditional normal law for inspection.
- **Preservation and evaluation:** version-2 manifests record requests, capabilities, targets, display preference, and scoring conventions. Point CSV columns remain compatible. Quantiles and highest-density segments are frozen in separate files before actuals are used. Equal-tail bounds receive the conventional Winkler score; quantiles receive unscaled pinball loss. Highest-density regions receive descriptive set coverage, total width, and segment count; the equal-tail score is not applied to HPD sets. Full-density log scores/CRPS are deferred.
- **Failure and compatibility:** missing or structurally invalid declared outputs fail the forecast visibly. Uncertainty scoring accepts all requested methods/levels as one case; overflow fails that case without discarding independent valid point scores. Any required uncertainty failure fails the overall run. Point-only uncertainty is explicitly inapplicable. Old version-1 saved point runs still render/compare without refitting or backfilled intervals. Comparisons use common supported uncertainty methods/levels and never fabricate baseline uncertainty.
- **Evidence:** hand-calculated predictive variance, known normal quantiles/regions, independent score examples, disconnected-set membership and plotting, capability failures, future isolation, frozen-file replay, and old-artifact compatibility are covered by tests. No dependencies are added. See [shared uncertainty](uncertainty.md), [the Bayesian adapter](local-level.md), and [references](references.md).

## Documentation readership and ownership — 2026-11-13

- **Problem:** usage steps, design goals, historical proposals, and implementation detail had accumulated together, with repeated examples using different paths. This made the current workflow and review order difficult to follow.
- **Organization:** the README is the entry point; the forecasting guide owns one end-to-end example; the model and uncertainty guides explain meaning/calculations; the source README is the API lookup; directory guides describe stored data/results; architecture separates implementation from planned design. Historical proposals and dated decisions retain context without acting as current setup/build instructions.
- **Consistency:** distinguish case completion from empirical region coverage, snapshot/run record versions, hidden-state versus future-observation uncertainty, and library calls versus script automation. Fixed model settings, pending human review, scope limits, and failure behavior stay explicit.
- **Evidence:** every repository Markdown file was reviewed; local links/anchors were checked. Walkthrough/library, CLI, alternate-chart, input/snapshot/tag, and CSV/XLSX examples were exercised in temporary storage. Documentation changes do not alter the runtime implementation.

## Review the current workflow before extending it — 2026-11-14

- **Sequence:** the user selected reviewing the existing modeling workflow now,
  then model diagnostics/evidence, then a real-data application. The review does
  not implement those later steps or choose their detailed design.
- **Evidence:** at `497680f`, all 118 tests passed. Both models completed 170
  cases on each regenerated synthetic series; local-level numerical artifacts
  and checks replayed exactly. Saved-result reporting succeeded with model and
  snapshot retrieval entry points patched to raise and preserved source evidence.
  The local-level CLI also saved its separate prior trajectory with pending review.
- **Interpretation:** no blocker was found in the exercised synthetic workflow.
  Computation, human model approval, and application suitability remain separate.
  Prior/noise sensitivity, fitted-model criticism, failure policy, real-data
  availability, and a final evaluation period remain open. No runtime behavior
  or automated approval rule changed; run review records remain pending.
- **Record:** [Workflow review](workflow-review.md) retains concrete examples,
  check meanings, results, and limitations. Generated stores/runs were isolated
  in temporary storage; their retention is not implied by committing this review.

## Make assumption influence explicit — 2026-11-14

- **User clarification:** sensitivity should inform us how much the prior affects
  the update path, not establish that a prior should have little effect. Strong
  or persistent influence is evidence for interpretation, not a computational
  failure or a quantity to minimize. Noise assumptions likewise shape updating.
- **Scope:** implement a separate local-level diagnostic workflow on a verified
  snapshot, with a caller-chosen reference and named alternatives. Process the
  full series from initialization with forward updates only. Save level means,
  state/next-observation SDs, gains, signed reference differences, full settings,
  and conditional initial-mean weights. Prior location, prior precision, and
  the two noise assumptions remain distinct; no automatic grid or selection.
- **Influence semantics:** the cumulative product of `V/Q` is the derivative of
  the updated level mean with respect to its initial mean, holding all variance
  settings fixed. It is not total prior influence or the prior's fraction of the
  estimate. Initial variance/noise counterfactuals change observation weights and
  uncertainty too, so their complete paths must be inspected.
- **Criticism:** save pre-observation standardized innovations, conditional normal
  PIT, and mean/mean-square/lag-correlation summaries, with saved-only charts.
  This is sequential predictive criticism, not full-data posterior predictive
  replication or an untouched final evaluation.
- **Failure policy:** invalid input/options raise before output creation.
  Numerical failures remain visible per configuration, later configurations
  continue, and any failure makes computation failed. Influence magnitude,
  path agreement, and residual patterns have no pass/fail thresholds. Human
  interpretation stays pending; no release gate or model ranking is added.
- **Ownership:** `diagnostics.py` supplies calculation and separate saved-result
  plotting; `scripts/diagnose_local_level.py` explicitly loads a JSON reference
  and scenario overrides and calls both. Backtest behavior/formats are unchanged.
  Expanded settings, input IDs, scope, implementation fingerprints, and errors
  are saved for inspection/replay; no dependency is added.
- **Evidence:** all 130 tests passed, including an independent batch-conditioning
  derivative, a hand calculation, a persistent full-influence case that succeeds,
  prior/noise distinctions, isolation/replay, failures, saved-only charts, and
  CLI inheritance. Both 200-observation synthetic series completed with a
  reference and four illustrative alternatives. See [diagnostics](diagnostics.md).

## Prioritize reusable experiments before an application — 2026-11-14

- **User correction:** application selection is not the immediate goal. Implement
  the four task/evaluation responsibilities as reusable infrastructure, retaining
  a near-term modeling detour for parameter uncertainty and MCMC considerations.
- **Contract:** frozen `ForecastTask`, `EvaluationPlan`, and `ExperimentSpec`
  records describe target/source intent, columns/units, daily calendar/gap rule,
  horizon, issuance, minimum available history, release/revision selection,
  scoring vintage, benchmarks, and target-based development/final boundaries.
  Explicit JSON loading is available in both run scripts; no application/source
  or inference backend is chosen. Legacy calls retain the effective 30-day,
  immediate/unrevised development default.
- **Information rules:** select latest available releases at or before issuance
  for periods no later than the origin; reject ambiguous/unordered vintages.
  Choose first-release or latest-in-snapshot actuals separately, after freezing
  forecasts. Daily calendar gaps are rejected, not imputed. One fixed positive-day
  horizon is supported per run; adapters can reject unsupported coordinates.
- **Evaluation:** target timestamps define development before the final start
  and final cases in a fixed half-open interval. Final runs are explicit caller
  actions; earlier final observations may enter later histories once available.
  This enforces request selection, not an assertion that humans have not seen
  final data. Full-input validation and trusted Python boundaries remain.
- **Benchmarks/evidence:** optional required training-mean/last-value forecasts
  share each primary request and scoring vintage. All forecasts are frozen
  before actual selection; any declared benchmark failure fails the run. New
  manifest version 3 records the expanded version-1 spec/hash; case schedules,
  history fingerprints, and actual release times preserve selection evidence.
  Reports read versions 1/2/3 and do not infer missing legacy task claims.
- **Next model work:** distinguish parameter/posterior-predictive uncertainty
  from finite-computation error. Select a small independently checkable model,
  then add relevant fit/chain metadata and diagnostics with a maintained backend
  after runtime checks. No sampler or parameter learning is added by this slice;
  see [the planned detour](parameter-uncertainty.md).
- **Verification:** all 146 tests passed. New counterexamples cover release and
  revision isolation, scoring vintages, final/development separation, matched
  benchmark calculations/freezing/failures, task replay, history identity,
  version-2 compatibility, and both CLIs. Both models exercised the known 200-day
  fixture with 130 development and 40 final-role cases plus both point benchmarks;
  Bayesian development artifacts replayed exactly. No new dependency is added.

## Verify parameter integration and a maintained sampler against an exact model — 2026-11-14

- **Request:** implement the next modeling detour thoroughly after reusable
  experiments, keeping application selection deferred. Work is on
  `feature/parameter-uncertainty`.
- **Model:** learn a constant Gaussian mean with a normal prior and known
  observation variance. Exact posterior and next-observation integration provide
  independent testable references; predictive variance includes parameter and
  observation uncertainty. Positive finite variance settings are caller choices;
  prior influence is saved descriptively without a magnitude criterion.
- **Inference decision:** keep forecasts exact in both modes and introduce MCMC
  as an explicitly labeled optional per-fit audit. This avoids adding sampled
  quantile/HPD estimation around a law already available exactly. It does not
  implement MCMC-only prediction or unknown local-level variance fitting.
- **Backend:** emcee GaussianMove, with a separate one-walker Metropolis sampler
  per independent chain. Do not treat interacting ensemble walkers as independent
  chains. Exact centering/scaling deliberately aids this calibration task.
  ArviZ's exercised pre-1.0 API supplies rank R-hat, bulk/tail ESS and mean MCSE.
  NumPy/SciPy are explicit direct dependencies; declaration, lock and pip export
  were updated together and exercised on CPython 3.14.10. More complex gradient/
  compiled backends and a home-grown sampler were unnecessary for this target.
- **Evidence boundary:** optional `fit(request)` returns a forecast with finite
  metadata/checks/diagnostics/draws. The runner pins history/input/coordinate
  identity, saves per-fit evidence, requires valid checks before accepting a
  forecast, and freezes everything before scoring. Existing adapters remain on
  `forecast`. Fit format 1 extends run format 3 without backfilling legacy runs.
- **Computation:** explicit saved defaults require R-hat <=1.01, bulk/tail ESS
  >=400, MCSE/SD <=0.1 and exact-reference agreement within 6 MCSE plus floating-
  point allowance for each diagnosed parameter/predictive quantity. Warmup is
  retained but excluded. Paired future draws and conditional CDF/HPD probabilities
  audit prediction. MCSE is not predictive uncertainty. HMC evidence is
  inapplicable, and acceptance fractions remain available.
- **Failure/replay:** partial chain errors retain rows and continue other chains;
  required diagnostic failures reject that forecast and fail the run while later
  origins continue. No analytic bypass occurs when an audit was requested.
  Persistence errors propagate and can leave running output. Seeds depend only
  on eligible history/coordinates/base seed/chain, never whole-snapshot future
  contents or execution order. Saved-only trace reporting never refits. Human
  interpretation remains pending.
- **Verification:** all 174 tests passed, including independent raw-density/
  predictive quadrature, HPD mass/density, strong prior influence, sampler target
  ratios, posterior/predictive agreement, stuck/biased/missing diagnostics,
  partial backend failures, future/release/final isolation, seed/global-RNG
  behavior, persistence/freezing, CLI modes and exact evidence replay. Both
  synthetic series completed 170 exact cases and five matched audit cases at
  target indices 100–104 with both benchmarks. All audit fit/draw files replayed;
  forecasts/scores matched exact counterparts. Largest R-hat was 1.003979;
  smallest bulk/tail ESS were 1550.38/1329.21. Known final-role fixtures establish
  mechanics, not untouched data or model approval. See [the implemented guide](parameter-uncertainty.md).

## Learn mean and observation variance against an exact reference — 2026-11-17

- **Request:** extend parameter learning to observation variance before attempting
  unknown local-level noise parameters. Application selection remains deferred.
- **Model:** add `NormalMeanVariance` with a proper normal/inverse-gamma prior.
  Exact joint updating yields Student-t mean-parameter and observation-predictive
  marginals. Prediction integrates both parameters; scale is not standard
  deviation. Existing known-variance and local-level adapters retain their models.
- **Inference:** optional independent two-coordinate emcee Gaussian Metropolis
  chains audit the exact reference. Conditional mean standardization and
  centered/scaled log variance assist the audit. The target evaluates prior,
  sufficient-statistic likelihood and the nonconstant transform Jacobian.
  No new dependency, adaptation, cache, or sampled endpoint estimator is added.
- **Moments:** all proper positive shape/scale/precision settings are allowed
  when requested forecasts are representable. Undefined posterior/predictive
  variances are explicit nulls. Moment MCSE audits require the variances of their
  quantities to exist; omissions have saved reasons. Precision, log variance,
  standardized conditional mean and bounded predictive checks remain required.
  Unrepresentable existing moments and missing applicable diagnostics fail.
- **Workflow:** reuse fit format 1/run format 3 and shared persistence, checks,
  information boundaries and seeds. `run_normal_mean.py --learn-variance`
  explicitly selects the new model; known/learned variance flags cannot mix.
  A separate sampler config requests more draws for two coordinates with
  unchanged computation thresholds. Saved-only plotting adds variance traces
  against inverse-gamma references and reads legacy normal fits.
- **Evidence:** independent raw prior-times-likelihood quadrature and density
  ratios, predictive mixture integration, HPD mass/density, transform-Jacobian
  ratios, joint sampler agreement, heavy-tail omissions, stuck/partial failures,
  future/RNG isolation, frozen evidence/replay, release/final cases and CLI modes
  are covered by the new tests. See [the guide](normal-mean-variance.md).
  All 194 tests passed. Both synthetic series completed 170 exact cases and
  three matched audit cases each, with both point benchmarks. Exact/audit
  predictions and scores matched, and numerical artifacts and fits replayed
  byte for byte. Saved-only charts/comparisons succeeded. Across the exercised
  joint audits, maximum R-hat was 1.005306 and minimum bulk/tail ESS were
  2638.73/3362.73. Artifacts are local and ignored by Git.
  Computational success does not approve priors or the constant-mean assumption.
  Unknown local-level variances and MCMC-only predictive estimation remain open.

## Prior and posterior predictive criticism before prior comparisons — 2026-11-17

- **Request/order:** implement prior and posterior predictive checks now, then
  work on prior influence comparisons as a separate next task.
- **Scope:** `NormalMeanVariance` only. Draw independent exact joint prior or
  posterior parameter pairs, with one pair shared by an entire replicated
  dataset. Use direct inverse-gamma/conditional-normal generation; no MCMC is
  needed. Independent marginal Student-t draws per position are not equivalent.
- **Information boundary:** prior-only runs need no input. Observed checks pin
  a verified snapshot and select latest eligible releases using an explicit
  aware as-of time. Only development role is allowed, as-of must precede any
  final boundary, and minimum available history is enforced. The posterior is
  fitted once to that selected history. Forecast scheduling/scoring settings
  are not reused as a hidden check schedule or as held-out claims.
- **Evidence:** persist all parameter/dataset draws and fixed location, spread,
  tail and temporal discrepancies. Save inclusive tail counts with valid/missing
  denominators, MCSEs and Wilson intervals. Fractions are descriptive, not
  calibrated test p-values. Undefined statistics are explicit; discrepancies
  never select a model or fail computation. Review remains pending.
- **Failures/replay:** independent PCG64 streams per phase/replication avoid
  global RNG changes and prevent one failure from shifting later draws.
  Simulation/statistic failures retain evidence and fail computation; a failed
  posterior retains prior results. Persistence errors propagate. Saved-only
  plots never fit/simulate. The explicit CLI calls calculation then plotting;
  backtests do not run these checks automatically. No new dependency is added.
- **Verification:** all 221 tests passed, including 27 new tests for joint
  moments/covariance, discrepancies, no-data prior behavior, undefined values,
  partial failures, future/releases/final isolation, seeds/replay, CLI modes and
  saved-only plots. Both synthetic series generated 1000 prior and posterior
  replications of their first 160 observations, excluding the final 40; numerical
  files replayed exactly. Stable-series temporal discrepancies were ordinary in
  these simulations; the shift series had zero upper-tail exceedances for lag
  correlation, half-mean difference and CUSUM. Zero counts retain positive
  Wilson upper bounds. Results are development evidence and require human
  interpretation. See [the workflow](predictive-checks.md).

## Compare prior influence through matched historical learning paths — 2026-11-17

- **Request:** implement prior influence comparisons after prior/posterior checks.
  Strong influence remains descriptive evidence, not a failure or target to reduce.
- **Scope:** explicit reference and named alternative `NormalMeanVariance`
  priors, with exact calculations only. Paths start at the empty-history prior,
  then the first available observation, and continue through a development
  cutoff. Mean location/precision and variance shape/scale overrides inherit
  unchanged reference settings. No simulation, MCMC, tuning or ranking is added.
- **Timing:** extract a shared latest-release selector for backtests, predictive
  checks and comparisons, preserving their semantics. Every prior receives
  identical as-of histories at daily period+issuance delay and the explicit
  cutoff. Refits incorporate eligible revisions without rewriting earlier rows.
  Minimum history applies at the cutoff; declared final boundaries are enforced.
- **Evidence:** save posterior parameters, conditional prior-location weights,
  scale decomposition, mean-parameter HPD, inverse-gamma equal-tail variance
  intervals, Student-t observation-predictive HPD and signed reference
  differences. Undefined moments and unavailable differences are explicit.
  The conditional coefficient is not total prior attribution. Scenario groups
  identify changed inputs, not an isolated attribution of all output effects.
- **Failures/reporting:** per-update failures retain errors and later updates
  continue; missing reference calculations never fabricate differences. Strong
  persistent influence completes successfully. Charts read saved evidence,
  with explicit log/symmetric-log display scales, and no model calls. The
  comparison CLI calculates then plots automatically; other workflows do not
  invoke it automatically. All affected documentation is updated together.
- **Verification:** tests cover hand calculations/decomposition, counterfactual
  mean effects, interval integration, variance concentration, strong influence,
  undefined moments, release/final isolation, shared selection, replay, failed
  updates/references, saved-only plots and CLI inheritance. Both synthetic series
  exercised six configurations on 160 development values, with 162 points each
  including the prior/daily updates/cutoff, and replayed six numerical files per
  series exactly. All 240 tests passed. The shift example showed identical mean
  locations but differing predictive widths under variance-prior concentration;
  a stronger mean prior retained a larger conditional location coefficient.
  See [the workflow](prior-influence.md). No new dependency
  or human model-approval rule is introduced.

## Connect state dynamics with learned observation variance — 2026-11-17

- **Request/order:** implement observation-variance learning for the local-level
  model, integrate its uncertainty into prediction, extend predictive checks
  and prior comparisons, document and test the work, then review broader project
  direction after those tasks are complete.
- **Model:** `LocalLevelVariance` retains fixed evolution variance and an
  independent Gaussian initial-state prior, with an inverse-gamma observation-
  variance prior. Evolution scales by actual elapsed UTC days. The prior is
  before the first one-day transition. This is not a conjugate common-scale DLM
  and its general predictive law is not Student-t. The old adapter is unchanged.
- **Inference:** one-dimensional log-variance Gauss-Legendre integration with
  support expansion, two successful node refinements, marginal-CDF checks and
  probability/first-moment tail bounds. Fixed-variance Kalman likelihood/state
  calculations integrate latent states analytically. No new sampler/dependency
  is needed. Numerical settings/versions and failed precision evidence are saved.
- **Prediction:** use the integrated Gaussian mixture for means, quantiles and
  density-threshold HPD sets. Never average conditional endpoints. Preserve
  disconnected segments; require numerical grid-refinement/mass/boundary checks.
  This is explicit approximation evidence, not a completeness proof for every
  pathological mixture. Undefined source moments are null; unrepresentable or
  unresolved calculations fail. Tail brackets do not loosen bulk root precision.
- **Replications:** draw variance and initial state jointly from their posterior,
  using conditional Gaussian smoothing within the selected history, then draw
  fresh process innovations and observation noise. Replicating a fitted latent
  path would check a different conditional object and is not used here.
  Independent per-replication streams, data boundaries and descriptive discrepancy
  semantics remain intact. Quadrature precision and Monte Carlo error are separate.
- **Prior comparisons:** vary initial-state and observation-variance priors on
  common historical releases; hold evolution variance/numerical settings fixed.
  State and variance intervals are equal-tail, predictive sets are mixture HPD.
  Inapplicable scalar prior coefficients, conjugate parameters/decompositions and
  Student-t degrees of freedom are null. Strong influence remains descriptive.
- **Verification:** independent batch Gaussian likelihood and initial-state
  conditioning, separate adaptive scalar integration, the known-level conjugate
  limit, mixture HPD/gaps, fresh-state-path moments, source isolation/replay,
  failure persistence and all CLI modes are checked. The model/check/comparison
  guide records precision limits and caller actions. The broader context review
  is performed after implementation verification, as requested.
  All 269 tests passed. Both fixtures completed 130 development forecasts,
  excluding 40 final values, and all 143 numerical/fit files per series replayed
  exactly. Each also exercised 1000 prior/posterior fresh-path replications and
  five prior configurations over 162 points. Accepted quadrature used at most
  1024 nodes with maximum tail probability estimate 7.842e-9. Saved discrepancy
  fractions remain descriptive; no model approval follows from completion.
- **Subsequent direction review:** after completing the three tasks and their
  verification, [the modeling review](modeling-review.md) finds the slice aligned
  with repeatable forecasting infrastructure. It records remaining stable/shift
  discrepancies and recommends consolidating existing evidence capabilities and
  experiment-owned requirements before adding another model/backend. This is a
  next-step recommendation, not an implemented check framework or model approval.

## Consolidate existing evidence and experiment requirements — 2026-11-17

- **Order:** preserve the verified learned local-level work first in commit
  `40b9d2b`, then consolidate its existing evidence workflows. No new model or
  sampler was needed to complete this infrastructure step.
- **Provider boundary:** optional `model.evidence_provider()` supplies simulation,
  prior/parameter/predictive summaries, scenario constraints and source/presentation
  metadata. Conjugate and learned local-level calculations live in separate
  provider modules. Shared workflows no longer branch on concrete model classes;
  they retain selection, discrepancies, differences, persistence and structural
  validation. Existing records are intentionally bounded to mean/variance/state
  evidence rather than inventing a universal parameter schema or registry.
- **Experiment ownership:** `EvidencePlan` declares required existing workflows,
  named primary-model fit checks and replication/seed/interval settings. Specs
  emit version 2; version 1 loads with default empty requirements. Reports
  normalize explicit v1/v2 specs for comparison without inventing missing metadata.
- **Execution:** the explicit `run_experiment_evidence`/`scripts/run_evidence.py`
  entry point pins a snapshot/history once, executes requested development
  workflows, checks child identity/settings and mandatory check names, and saves
  failed requirements when capability/scenarios/checks are missing. Later
  workflows continue; I/O propagates. `run_backtest` separately enforces named
  fit checks, saves independent experiment requirements and records separate
  workflow requirements as pending. Neither run claims to complete human review.
- **Compatibility:** standalone workflows keep their explicit options, RNG streams,
  statistics and numerical records. Saved presentation metadata drives new plots;
  legacy artifact kinds remain readable without fitting. Artifact hashes are
  receipts, not a release system or immutable storage guarantee.
- **Verification:** all 291 tests passed, including 22 new interface/requirement
  tests covering an adapter unknown to shared workflow code, omitted/duplicated
  replications, malformed summaries, missing/optional checks, unsupported evidence,
  pinned tags/history/settings, replay/final isolation, numerical/I/O failures,
  v1/v2 comparison and both CLI modes. On each original synthetic history, 18
  evidence files replayed byte for byte, 324 prior-learning rows matched, and
  both selected prior-summary records matched. Artifacts remain local under
  `runs/evidence-interface-20261117-fewaws1h/`. See [the contract](evidence.md).

## Working proposals

- Choose installable package/build configuration when distribution is needed; current imports use `PYTHONPATH=src`.
- Extend the first JSON/CSV run artifacts only as another model or task requires it.
- Extend local-level criticism beyond the implemented influence/innovation evidence when an application needs it; select any broader specialized check plan explicitly.
- Application selection is deferred; the earlier market-series proposal is not the immediate continuation plan.

These are starting choices to revisit deliberately, rather than implemented interfaces or completed experiments.

## Open questions

- Which additional synthetic processes would expose Bayesian or feature-model failure cases?
- Which additional dependencies or development tools are justified by a selected next task? The existing stack is declared and locked.
- Which standard build backend should package the implementation when installable distribution is needed?
- How should the shared mean/quantile/density-region boundary grow for full predictive samples/densities and specialized checks?
- Which checks stop execution, and which allow exploratory evidence collection while blocking release?
- What requires explicit human interpretation in the first review?
- Which additional artifacts are needed beyond the first inspectable JSON/CSV records?
- Which instrument/source/target would make the later real-data exercise informative?
- Which export/backup practice should preserve important local data and runs? Repository code is already published under Apache License 2.0.

## Record a consequential change

Use a dated entry with the question, decision, rationale, alternatives considered, consequences, and evidence or issue/PR links. Preserve superseded choices so later changes remain understandable.

## Sequential learned-variance diagnostics — 2026-11-18

Task 1 extends criticism through provider-supplied serialized predictive laws. Predictions are written before scoring; historical issuance controls training releases and the development cutoff controls eligible scoring releases. Full mixture/Student-t CDFs supply PIT. Undefined variance is explicit; residual patterns have no approval threshold. Existing fixed-variance diagnostics remain readable. The workflow is explicitly invoked and can be experiment-required; its CLI plots automatically. Nine new tests exercise calculation and information boundaries. See [the contract](sequential-checks.md).

## Continuous-CDF simulation calibration — 2026-11-18

Task 2 uses repeated independent prior/data draws and marginal posterior CDFs, medians and equal-tail intervals. Known states are excluded. This evaluates current exact/numerical inference without pretending to audit sampled ranks or an MCMC backend. Truth/data are saved before fits; failed fits remain visible and invalidate unconditional interpretation of success-only bands. Uniform-reference bands account for multiple marginal parameters. Computation and human interpretation remain separate. Ten new tests and two 200-trial exercises support the slice; see [the calibration guide](calibration.md).

## Categorical evolution-variance prior — 2026-11-18

Task 3 first compared W=0/.01/.1/1 on matched stable/gradual/abrupt fixtures with identical noise. Different dynamics helped different processes. The initial joint model therefore uses an explicit categorical W prior and independent inverse-gamma V prior, enumerating all W and using checked scalar conditional evidence. This is an exact discrete model for W, not a claim of continuous W integration. Posterior weights, full predictive mixtures and V/W covariance are saved. Independent batch Gaussian/adaptive integration verifies weights/CDF/confounding; one support point recovers the scalar model. A mixture-median roundoff issue exposed by that reference was fixed with ulp-scaled endpoint verification. Thirteen new tests support the slice; see [details](joint-variance.md).

## Joint predictive and prior evidence — 2026-11-18

After completing the first three tasks, selected two follow-ups: joint-model generative criticism/prior influence, then discrete-aware calibration. Task 4 draws W/V/initial state jointly and resimulates innovations. Prior comparisons vary probabilities on fixed support and save W mass/bounds/covariance; optional columns remain null for earlier models. Gaussian/point state mixtures retain true atoms rather than inventing positive variance. Metadata controls additional plots; old saved evidence remains readable. Ten new tests cover probability, information and failure boundaries. See [joint evidence](joint-evidence.md).

## Discrete and mixed posterior calibration — 2026-11-18

Task 5 extends data-free calibration to the categorical-W model. Providers supply strict/inclusive CDF bounds and closed-interval posterior mass; shared execution randomizes only actual jumps using parameter-labelled streams independent of dataset generation. Fixed parameters are excluded; unrepresentable variances that should be positive fail rather than becoming invented atoms. Discrete interval coverage is interpreted alongside its actual mass. Marginal calibration does not replace independent likelihood/recovery/dependence checks. Ten new tests and retained simulation/reference exercises support the change; an initial finite-simulation band crossing remains documented with its independent follow-up. See [the guide](mixed-calibration.md).

Task 5 verification completed with all 343 tests passing. The independent batch
CDF checks agreed within 2.3e-12. A separate 1000-trial joint follow-up (seed 814)
completed without failures; its marginal CDF distances were within their saved
simultaneous reference band. The initial 200-trial band crossing and mixed-state
results remain retained and documented; no automatic model approval is inferred.

## Saved experiment review — 2026-11-18

Use a version-3 forecast as the explicit identity anchor and assemble existing
evidence/calibration without refitting, discovery or source mutation. Compare
expanded configurations, history/task/boundary identities, named checks and
shared declared versions; verify existing child completion receipts. New inventories
record current bytes rather than claiming historical authentication. Existing
sequential/bundle environment omissions stay explicit. Readiness is computational;
a new human review always starts pending. Eighteen tests and a saved development
exercise support [this boundary](experiment-review.md).

## Full predictive distribution losses — 2026-11-18

Require an explicit model capability and serialized law from the same forecast
fit. Freeze all laws before actual selection; score deterministic Student-t and
full Gaussian mixtures with lower-is-better CRPS/negative natural log density.
Do not infer laws from intervals, prune modes or floor tail density. Undefined
first moments and unresolved numerical precision fail visibly. Saved comparisons
require complete matching cohorts/methods/units, recompute aggregates from saved
case scores and retain paired differences/charts without choosing a winner.
Twenty-one tests support [the contract](distribution-scoring.md). An older overflow
test now explicitly opts its intentionally altered region adapter out of the new
full-law capability so it continues to test region scoring independently.

## Locked-environment repository CI — 2026-11-18

Define a single Ubuntu 24.04 job for branch pushes/pull requests and manual reruns.
Pin the checkout/setup actions, uv and interpreter; install from the lock, verify
dependency compatibility and pip-export parity, then run all tests and local
documentation checks. Execute the forecasting guide's Python examples verbatim
in temporary storage. Separate parsing of other examples from execution, and
exclude ignored run/dependency artifacts from discovery. Nine checker tests, a
fresh 24-dependency installation, the 170-case walkthrough and actionlint support
[the workflow](continuous-integration.md). Large simulation studies and model
interpretation remain explicit; branch protection is an administrator decision.

## Resume checkpoint — 2026-11-18

Record completed work and the next three evidence-interpretation tasks in a
tracked [next-session note](next-session.md), linked from the README and roadmap.
The review/scoring/CI implementation is merged at `0ff5def` with passing hosted
push/PR checks. Future work begins with complete reviews, then scenario/prior
criticism and a documented modeling decision; this checkpoint does not select a
new model or execute those planned studies.

## Interpret model evidence before choosing the next transition — 2026-11-19

- **Question:** use the completed forecasting/evidence/review infrastructure to
  explain current model failures and choose a bounded next modeling question.
- **Design:** start with the original mean-shift input, then stable and gradual
  controls with identical noise. Freeze development/final boundaries, 70/20/40
  target windows, reference settings, all declared alternative priors and seeds
  before execution. Require all three historical evidence workflows plus
  compatible model-level calibration for each reference review. Prior influence
  remains descriptive; no score or discrepancy threshold approves a model.
- **Compatibility:** earlier forecast plans and calibration source hashes differ.
  Preserve them and generate an identified current exercise. Keep calibration's
  20-observation prior/data scope separate from historical criticism and final
  evaluation. All final-role targets remain excluded.
- **Failure found:** fresh joint calibration failed at trial 538 of seed 1911.
  Near-total mass on one W component exposed a numerical conditional-quantile
  bracket error. Widen a failing side to checked integration support, retaining
  root/CDF tolerances. Save the failed original exercise; the same 1,000 truth/
  data files replay exactly after repair. A seeded regression compares a
  separately bracketed root. All 3,000 replacement reference trials compute;
  this replay is not an independent follow-up or adequacy approval.
- **Execution:** explicit research scripts compose existing library workflows.
  A continuation script reuses compatible complete artifacts, preserves partial
  output and runs independent model/scenario jobs concurrently. This addresses
  the concrete cost of this study without introducing a general scheduler or
  changing inference. A JSON-list/Python-tuple reuse bug was detected and fixed;
  its regression uses an actual joint forecast. Native artifacts retain their
  own source hashes, and continuation source versions are retained locally.
- **Interpretation:** the joint model improves stable-period behavior while its
  abrupt-onset losses exceed the fixed-W model's. Later dynamic forecasts recover,
  but replicated abrupt-change CUSUM remains unusual across every tested prior.
  Lower aggregate CRPS therefore does not establish that the transition problem
  is resolved. Stable/gradual controls and all declared priors remain visible.
- **Next question:** investigate occasional large innovations while preserving
  quiet periods. Start with fixed small/large innovation scales and mixture
  probability; require a short-history independent enumeration/batch-Gaussian
  reference, Gaussian limiting cases, explicit approximation precision,
  fresh-path criticism, calibration and matched independent-seed comparisons.
  No new transition adapter, continuous-W prior or MCMC forecast backend is
  selected or implemented by this review. See [the study and validation
  criteria](model-review-study.md).
- **Completion:** nine reference reviews are ready; 45 forecasts retain 5,850
  complete cases, 27 required historical workflows pass, and the 36 alternative
  predictive workflows retain 72,000 replications. The audit checks 102 report
  links, pre-change identity, PIT agreement and source versions. All 401 tests,
  documentation checks and the temporary walkthrough pass locally. Curated
  window/prior/onset summaries and a figure are tracked with the interpretation;
  full runs remain local. This work is not newly published or model-approved.

## Establish a bounded per-transition innovation oracle — 2026-11-19

- **Question:** validate the occasional-large-innovation model before choosing a
  longer-history inference method or adding a forecasting adapter.
- **Model:** fixed caller probability and ordered small/large Gaussian innovation
  variances, independent labels per daily transition, independent Gaussian initial
  state and inverse-gamma shared V. The first transition precedes the first
  observation; the next transition has the fixed prior probability. Keep
  `prior_shape > 1` for finite reference moments, allow genuinely known states,
  and cap numerical histories at eight observations.
- **Reference:** enumerate every positive-prior sequence. Dense Cholesky/batch
  Gaussian conditioning with Joseph covariance and adaptive rational-V integration
  is compared with independent scalar Kalman filtering/backward smoothing and
  refined log-V Gauss-Legendre integration. Require two successful refinements,
  probability/first-moment tail bounds and full summary/CDF/density agreement.
  Initial/final state atoms and allocation/V dependence remain explicit. No
  component pruning or matched-Gaussian predictive substitution is introduced.
- **Independent checks:** direct one-observation variance integrals and exact
  known-level inverse-gamma/Student-t calculations supplement comparison of the
  two methods. Probability zero/one and equal innovation variances recover the
  existing Gaussian model; equal scales leave labels at their independent prior.
- **Evidence:** seven frozen numerical cases replay exactly; maximum scaled
  discrepancy is 4.78e-11 and the three existing-model limits agree within
  3.90e-12. Fourteen new tests include failure persistence, all 256 paths, atoms,
  posterior dependence and CLI/replay boundaries. Curated summary/figure are
  tracked; full results remain under `runs/innovation-reference-verified-20261119/`.
- **Feasibility/next:** 256 paths are tractable; 160 nondegenerate transitions
  would require 2**160 paths before V integration. Retain this as a short-history
  oracle and design a controlled approximation against it, with explicit omitted
  mass or Monte Carlo evidence and preserved Gaussian limits. Forecast/provider
  integration, fresh-path criticism, prior comparisons and calibration follow
  that design. No long-history algorithm, generic MCMC backend or application
  is selected by this prototype. All 415 tests pass. See [methods and limits](innovation-reference.md).

## Investigate longer histories with replicated Gaussian-state particles — 2026-11-19

- **Question:** determine whether the fixed small/large innovation model can be
  approximated on the earlier 160-observation development histories before
  extending forecast/provider boundaries.
- **Method:** sample static shared V from its prior and propose each past label
  using both observation likelihoods. Integrate Gaussian states analytically,
  resample whole V/label/state ancestries systematically at declared ESS, and
  smooth conditional initial/final states at completion. Preserve real state
  atoms, V/allocation dependence and both fresh future innovation branches.
  Prior/Gaussian limits automatically use checked scalar quadrature.
- **Uncertainty:** combine independent complete filters through their estimated
  evidence, then recompute centered marginal/joint moments. Use delete-one-run
  jackknife MCSE, finite-probe/planned-look Student-t diagnostics, independent
  particle-count refinement, evidence ESS and V ancestry. These are empirical
  checks, not omitted-mass bounds, independent-particle errors or exact coverage.
  Keep every attempted level and failure; do not replace failed replications.
- **Scope:** at most 512 daily observations; stochastic calculations require
  prior shape > 2. The study freezes seven original reference histories, four
  limit/atom cases and six matched 160-observation cases, with 32 independent
  filters and 4,096..32,768 particles per filter. Ordinary imports/backtests/CI
  do not launch this explicit exercise. No new dependency or generic MCMC
  backend is introduced.
- **Decision:** retain this as a numerical feasibility prototype. Short-oracle
  checks reveal overconfident error envelopes for rare paths, including two
  entirely unvisited allocations. Small numerical discrepancies do not justify
  discarding these checks. Resolve rare-path uncertainty before forecasting,
  predictive criticism, prior comparisons and calibration. See the
  [method, frozen results and limitations](innovation-particles.md).
- **Recorded outcome:** all six long cases meet empirical refinement at 16,384
  or 32,768 particles; the overall study remains unresolved because two short
  cases miss three rare-path envelopes. Eight source hashes and 34 aggregations
  audit successfully; eleven short cases and twelve selected long replicates
  replay exactly. All 431 tests and documentation/walkthrough checks pass.
  Work is local and uncommitted. See [the handoff](next-session.md).

## Bound unseen innovation probabilities independently — 2026-11-19

- **Problem:** the retained particle study has two unvisited positive-mass paths
  and one underestimated rare path outside its empirical MCSE envelopes. Neither
  particle refinement nor sampled ancestry proves missing mass is negligible.
- **Design:** add a separate fixed-budget importance check. Propose V from
  `IG(a+n/2,b)` and labels independently from their fixed prior. The Gaussian
  likelihood bound yields scaled importance weights in [0,1]. Two-sided
  empirical Bernstein bounds, unioned over every numerator and the shared
  normalizer, give simultaneous posterior probability intervals. Unseen events
  retain a positive radius. Preserve atom-aware state CDFs and the full two-branch
  next-observation CDF. Densities, unbounded moments and floating arithmetic
  remain outside this confidence contract.
- **Execution:** the caller explicitly invokes the independent check or its
  saved-study script. Freeze parent inputs/failures, source snapshots, settings,
  query grids and error allocation before computing. Unresolved normalizers or
  widths remain saved, and the study returns exit code 1. Original particle
  inference, saved failures and its precision criteria remain unchanged.
- **Evidence:** 262,144 independent draws per case with 95% study confidence
  enclose all 872 probabilities from the eleven short oracles. Eight short
  cases meet the width gate; three short and all six 160-observation cases do
  not. The long-history normalizer has lower bound zero and intervals reach
  [0,1]. This bounds unsupported zero-uncertainty claims but does not verify the
  approximation. The [guide](innovation-probability-bounds.md) retains the three
  prior failures, derivation, scope and reproducible evidence.
- **Decision:** retain this as a conservative diagnostic and a completed design
  investigation. The [next pilot](next-session.md#next-work) prioritizes a
  data-informed importance proposal with a provable weight bound. Targeted
  integration remains an alternative. Recheck long-history cost and ancestry
  if particle inference changes. Keep Gaussian limits, atoms,
  joint dependence and full predictive probes. Forecast/provider integration,
  fresh-path criticism, prior robustness and calibration remain gated. All
  442 software tests pass; useful numerical precision remains separate.

## Use an observation-conditioned state grid for probability bounds — 2026-11-19

The previous prior proposal left three short and all six long cases unresolved.
The new proposal uses a finite state grid and a piecewise envelope over
observation precision. A gamma density maximum and a convex likelihood chord
bound each weight. A Gaussian lattice-sum bound controls the grid error for
each allocation, and a gamma tail bound accounts for omitted precision ranges.
Each draw uses conditional Gaussian smoothing to recover true state atoms and
evaluate the full predictive mixture. The retained exploratory FFT produced
incorrect small likelihoods in a dense Gaussian limit, so the study uses direct
convolution.

The corrected three-case pilot passed, then all 17 cases passed with the
original 262,144 draws, confidence allocation, and maximum interval width
of 0.10. The widest long-history interval was 0.03379. All 872 short oracle
probabilities passed the retained comparator, including the original three
rare-path failures. Callers explicitly run the pilot or the full study. The
command archives the sources and inputs and preserves failed records. See the
[derivation and results](innovation-lattice.md).

Keep the state-grid proposal as the verified probability prototype. Its
moments and predictive density estimates are descriptive. Verify these
quantities independently on a long history and check cost and failure
handling for repeated fits before deciding how to integrate forecasting and
evidence providers.

## Keep shared forecasting open to other model families

Bayesian dynamic linear models are the first main use case. The project also
covers other state-space models, harmonic regression, multivariate forecasting,
and other families as concrete tasks arise. Keep model parameters, fitting,
and specialized diagnostics in the adapter or evidence provider. Share data
eligibility, execution, forecast recording, evaluation, and reporting across
model families.

Build each shared capability around a concrete example. Harmonic regression can
exercise seasonal predictors with the existing scalar target. A joint
multivariate forecast can establish requirements for series identity, alignment,
release timing, and dependence in saved outputs. The current scalar contract
continues to support the first use case. See the [roadmap](roadmap.md) and
[reading guide](reading-guide.md).

## Choose a bounded first BDLM workflow

The continuous-state Gibbs study agrees with the state-grid results on
`abrupt-42` at both independent budgets. Its short control needed a separate
longer run to meet the same precision criteria. The full-budget state-grid
fits took about 16 seconds each on the tested fixture. Use the existing
Gaussian local-level adapters for the first BDLM workflows. Retain the
rare-jump method as research for tasks that require it. The weekly harmonic
example confirms that the current scalar contract supports a different
family without shared runner changes. See the [evidence](innovation-moment-check.md).

Keep external frameworks as optional adapter choices. PyDLM and PyMC Extras
are candidates for research comparisons or an approved adapter. Check the
workplace interpreter and approved packages before establishing a smaller
installation profile. The repository's existing lock remains the full
research environment. See the [PyDLM guide](https://pydlm.github.io/pydlm_user_guide.html)
and [PyMC Extras state-space models](https://www.pymc.io/projects/extras/en/stable/statespace/models.html).

## Verify a focused Python 3.12 workflow on public data

The first workplace workflow needs CSV input, the existing Gaussian local-level
adapter, shared scoring, and a saved chart. A clean CPython 3.12.16 environment
completed that workflow with pandas, NumPy, SciPy, and Matplotlib plus nine
transitive packages. The pinned Python 3.12 profile is separate from the full
Python 3.14 research lock. Workplace compatibility requires testing the approved
packages on the workplace platform.

Use Central Park daily maximum temperature as the first public development
task. Preserve the retrieved NOAA archive bytes and normalized input and
compare against the last observed value. The 2024 run completed all 366 cases.
The archived date labels support a simulated forecast origin. Operational use
requires publication times or vintage histories. See the [setup, task design,
and saved results](python312-public-forecast.md).

## Compare annual seasonality and persistence on the NOAA development dates

Large realized changes drove most of the baseline squared error. Compare an
annual Fourier pair, a fitted one-day lag, and their combination at the same
366 Central Park origins. The combination gave the lowest MAE and RMSE and
higher error than persistence on quiet days. The annual-only model scored
worse overall. Keep the regressions inside the explicit example and use the
existing shared run and comparison contracts. Freeze the configurations for
a later check on dates outside the 2024 development cohort. See the
[diagnostics and results](noaa-seasonality.md).

## Evaluate the frozen NOAA seasonal model on 2025

Freeze the annual-plus-lag and lag-only settings, persistence control,
2025 target dates and both point-score criteria before acquiring the 2025
observations. Preserve the exact normalized 2023–2024 input as the prefix.
Refit coefficients on each eligible expanding history. The combined model
beat both controls on MAE and RMSE. Its quiet-day errors and warm mean error
remain visible. Keep this as a later-period result under simulated archive
availability. Real-time use requires release evidence.
See the [full comparison and audit](noaa-later-period.md).

## Match separate NOAA minimum and maximum forecasts before adding joint outputs

Run the same seasonal regression and persistence baseline for minimum and
maximum temperature with matching origins and eligible dates. On the 365
matched 2025 dates, both pairs retained minimum-to-maximum order. The
seasonal pair lowered range MAE by 14.8%. Its minimum and maximum forecast
errors had a correlation of 0.603. Keep the scalar point forecasts as
benchmarks. Use a concrete probability about the range to motivate a joint
output in a later evaluation. See the [paired results](noaa-temperature-pair.md).

## Rebuild adapters from saved settings through one catalog — 2026-11-19

Saved `NormalMean` and `NormalMeanVariance` configurations with MCMC settings
could not be restored: `configuration` saves `mcmc` as a dictionary, and the
constructor requires `MCMCSettings`. Five scripts also repeated the same
model-name branches. Every repository adapter now provides
`from_configuration`, and scripts build models and prior scenarios through
`adapters.build_model` and `adapters.build_scenarios`. The catalog serves
scripts only. The runner and evidence workflows still accept any conforming
object, so a new model family needs no shared-code change to run. Provider
selection for calibration now also rejects missing or duplicate
`calibration_parameters` with `ValueError`. Model outputs and saved formats are
unchanged. Saved runs record source hashes, so artifacts made before this change
report the edited modules as changed sources. See the
[API reference](../src/model_pipelining/README.md#rebuilding-adapters-from-saved-settings).

## Define shared names once and write artifacts through one module — 2026-11-19

Benchmark names were listed in both the experiment spec and the runner.
Workflow names were listed in experiments, providers and the evidence runner.
Three private JSON writers and two CSV writers were copied or imported across
modules. Each name list now has one definition: `BENCHMARK_MODELS`,
`EVIDENCE_WORKFLOWS` and `PROVIDER_WORKFLOWS`. Shared tables
(`WORKFLOW_MANIFESTS`, `REQUIRED_WORKFLOW_CHECKS`), `read_observations` and
`tail_summary` are public. All saved JSON and CSV go through
`artifacts.write_json` and `artifacts.write_csv`. Saved bytes are unchanged.
Two behaviors are stricter. Runner outputs other than the final manifest are
now created exclusively, matching the evidence workflows. A JSON value that
cannot be serialized no longer leaves a partial file. Manifests now list
`artifacts.py` among their source hashes, and diagnostics list `experiments.py`
in place of `backtest.py`. Model-family internals, such as the innovation
modules' shared helpers, stay private. See the
[API reference](../src/model_pipelining/README.md#shared-names-and-artifact-writers).

## Move the weekly harmonic adapter into the package; keep frozen NOAA scripts unchanged — 2026-11-19

`WeeklyHarmonicRegression` moved from `scripts/run_harmonic_example.py` to
`model_pipelining.harmonic_regression`, with its name and configuration
unchanged. It also joined the adapter catalog with a fixed `from_configuration`.
Re-running the example reproduced the saved
`runs/weekly-harmonic-example-20261119` input, predictions, evaluations,
scores and case schedule byte for byte. Only the recorded class path and source
filename changed. `scripts/__init__.py` now makes the `python -m scripts.…`
commands use a regular package.

`TemperatureRegression` and the NOAA helpers (`daily_maxima`, `MODELS`,
`_evaluated`, `_metrics`) stay in `study_noaa_seasonality.py` and
`run_noaa_temperature.py`. The frozen 2025 plan records SHA-256 hashes of those
scripts and of `check_noaa_later_period.py`. `evaluate` refuses to run if any of
them changes, so moving code out of them would block replaying the frozen
evaluation. Move them into the package when the next NOAA experiment freezes its
own plan. That plan can then pin the package modules.

## Check the Python 3.12 workplace profile in CI — 2026-11-19

The 3.12 profile had been verified only by hand with targeted tests. CI now runs
a second job that installs CPython 3.12.16 with exactly the pinned
`requirements-python312.txt` packages and runs the whole suite. Tests that need
the research-only packages `emcee`, `arviz` or `openpyxl` use
`tests/optional_dependencies.requires` and are skipped there. The job asserts
that those packages are absent, so the skips are the intended ones. The research
job sets `REQUIRE_OPTIONAL_TEST_DEPENDENCIES=1`, which turns any such skip into
a failure, so no test silently disappears from the full environment. The cost
is that shared code must keep running on Python 3.12. A dependency change also
needs both `uv.lock` and `requirements-python312.txt` updated.

## Package temperature code and preserve the frozen NOAA replay

The paired study imports the scalar `TemperatureRegression` adapter, metric
NOAA parser and saved-forecast checks from the package. The parser takes a
station, inclusive dates and either `TMIN` or `TMAX` from the caller. The
adapter restores each of the three historical seasonal configurations from
JSON. New paired designs record the package module digests alongside the
script digest. The frozen maximum workflow still runs its original three
scripts with their recorded hashes. The next range study can pin the package
modules when its design is confirmed.

Validate benchmark entries as strings before consulting the shared benchmark
dictionary. Invalid types or names in a loaded experiment now follow the
same `ValueError` path with the allowed benchmark names.

## Compare empirical range probabilities from the same marginal errors

Use the full 2025 matched minimum and maximum residual series as calibration.
The same-day forecast gives each matched pair equal weight. The independent
forecast gives every cross-pair equal weight. Both use the same saved scalar
point forecasts. Score the inclusive range event on matching later dates with
Brier loss, and retain the probability of a strictly negative range. Use
decimal comparisons of the saved Celsius values at the event boundary.
Freeze the inputs, study sources and evaluation settings before retrieving
later observations. The [study script and workflow](noaa-range-probability.md)
keep event scoring at the study level.

The frozen January–June 2026 evaluation scored 181 targets, of which 64
reached the 10 °C threshold. Independent pairing scored 0.22415 and same-day
pairing scored 0.23935. Mean negative-range probability fell from 5.01% to
0.91% with same-day pairing. The evidence supports keeping this comparison at
the study level. See the [result and audit](noaa-range-probability.md#januaryjune-2026-results).

## Evaluate seasonal residual pools alongside the full-year controls

Use 2025 residuals from the target’s meteorological season as a study-level
variant. Compare both pairing methods within each pool choice on identical
point forecasts, with paired Brier scores and negative-range probability.
Preserve the verified January–June archive as eligible training history.
The ten-day July pilot found worse Brier scores with seasonal calibration
for both pairing methods. Retain the full-year methods and the frozen 174-date
follow-up design. See the [diagnostic and pilot](noaa-range-probability.md#seasonal-calibration-follow-up).

