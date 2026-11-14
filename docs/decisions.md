# Decisions and open questions

This is a chronological design history. A dated entry records what was decided
at that time; later entries can supersede it. Use [the README](../README.md),
[the API reference](../src/model_pipelining/README.md), and [the roadmap](roadmap.md)
for current behavior and review priorities.

## Current choices at a glance

- Preserve one local file per snapshot, identified by dataset plus SHA-256.
- Keep acquisition, registration, tag assignment, and execution separate calls.
- Run the training mean or fixed-variance Bayesian local-level adapter through one daily backtest.
- Save a predictive mean and each declared observation-uncertainty kind. Prefer highest-density prediction regions in reporting.
- Keep failed cases visible, human interpretation separate, and data/run evidence local.
- Use CPython 3.14 with pandas, openpyxl, and matplotlib; installable packaging, a configurable check plan, and release automation remain open.

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

## Working proposals

- Choose installable package/build configuration when distribution is needed; current imports use `PYTHONPATH=src`.
- Extend the first JSON/CSV run artifacts only as another model or task requires it.
- Extend the implemented local-level adapter's prior sensitivity, fitted-model criticism, and specialized check evidence.
- One market series as the first real-data application, followed later by a separate operational application.

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
