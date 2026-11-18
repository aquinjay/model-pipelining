# Decisions and open questions

This is a chronological design history. A dated entry records what was decided
at that time; later entries can supersede it. Use [the README](../README.md),
[the API reference](../src/model_pipelining/README.md), and [the roadmap](roadmap.md)
for current behavior and review priorities.

## Current choices at a glance

- Preserve one local file per snapshot, identified by dataset plus SHA-256.
- Keep acquisition, registration, tag assignment, and execution separate calls.
- Run point models, Bayesian local-level adapters with fixed or learned observation variance, or constant-mean Gaussian adapters with known or learned observation variance through shared daily experiments, with adapter-specific timing limits.
- Save a predictive mean and each declared observation-uncertainty kind. Prefer highest-density prediction regions in reporting.
- Keep failed cases visible, human interpretation separate, and data/run evidence local.
- Treat prior/noise sensitivity as descriptive influence evidence; strong influence does not fail computation. Save explicit local-level diagnostics separately from backtest results.
- Declare target/timing, release/revision selection, benchmarks, and evaluation roles in versioned experiment specs; defer application selection. Unknown-mean parameter integration and optional independent-chain MCMC audits are implemented.
- Use model-supplied evidence calculations and experiment-owned fit names/separate workflow requirements; execute separate evidence explicitly and retain failed or unavailable requirements.
- Use CPython 3.14 with pandas, openpyxl, matplotlib, NumPy, SciPy, emcee and pre-1.0 ArviZ; installable packaging, arbitrary-stage checks, and release automation remain open.

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
