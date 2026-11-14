# Decisions and open questions

This log distinguishes current direction from proposals. Update it when a choice affects observable behavior or a responsibility boundary.

## Current direction — 2026-11-01

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
- **No selected packages yet:** begin with an empty dependency list. Runtime libraries and development tools will be selected together before the synthetic experiment.
- **Packaging deferred:** this is currently an environment-managed project with no installable local package. The uv-specific `package = false` setting prevents attempts to install the placeholder source directory. Standard pip setup installs exported dependencies only.
- **Work portability:** a different interpreter or platform must be checked deliberately. The current range records our development target, not a guarantee about an unknown work deployment runtime.

## Data versioning decisions — 2026-11-03

The agreed direction, alternatives, open details, and ordered continuation plan were saved in [Data versioning: decisions and next steps](data-versioning.md). No versioning code existed at that point. The narrower implementation selected later is recorded below.

The selected approach preserves snapshots of one or more related files, compares SHA-256 file hashes before reusing contents, and records every acquisition. Dataset-scoped tags can move only through explicit requests, including automation. Runs pin permanent snapshot IDs. Processing preserves lineage; quality failures remain visible; required failures block model runs unless an explicit experimental override is recorded. Storage starts locally behind a replaceable destination boundary.

## Ingestion first; preservation and execution later — 2026-11-09

- **Earlier preservation tool direction (superseded below):** use git-annex for preserving and retrieving versioned file contents. git-annex has not been installed or configured for this project.
- **Execution tooling deferred:** consider DVC or another tool for dependencies, stages, execution, and caching when the runnable workflow needs it. DVC is a candidate, not a selected dependency. Avoid duplicating execution responsibilities between tools and the project runner.
- **Immediate priority:** begin a small library of ingestion functions with one function for a local file and a companion CSV/XLSX table reader with worksheet selection by name or index. Functions for Snowflake, S3, SFTP, and other sources can follow as concrete needs arise. Acquisition remains separate from parsing, snapshot registration, validation, and model execution.
- **Local implementation:** [Local ingestion: first slice and later enhancements](ingestion.md) describes the implemented interface and behavioral checks. pandas and openpyxl are declared, locked, and exported for table reading. Standard-library unittest is sufficient for this slice. The package is imported through `PYTHONPATH=src`; installable packaging is deferred.
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
- **Evidence:** all 57 behavioral tests pass, including retained context after source removal and store relocation, distinct registration/acquisition times, receipt-independent identity, UTC normalization, invalid receipts, and older records.

## Build shared execution around a training-mean model — 2026-11-13

- **Question and decision:** after tagging, the user selected building the first modeling infrastructure around a working simple forecast, on a new branch (`feature/first-forecast-workflow`). Start with the existing synthetic daily series and a training-mean baseline rather than designing a larger unused interface first.
- **Task:** predict tomorrow's `value` immediately after today's observation; 30 initial observations by default, then expanding history. Validate ordered/unique daily UTC timestamps and finite numeric values. Immediate availability without revisions is an explicit synthetic-data assumption, not inferred from ingestion timestamps. All forecast dates are development evidence; no untouched final evaluation period is reserved.
- **Boundary:** `models.py` supplies immutable observation/request records, a point-forecast protocol, and the baseline. A model receives eligible history, issuance, and target time, and returns a finite predictive mean. The runner owns scheduling, coordinates, validation, evaluation, and artifacts. Models refit from each request; retained training state, probabilistic capabilities, and adapter diagnostics are deferred. The interface assumes trusted Python code.
- **Data and evidence:** resolve any tag once, verify its snapshot, and save the permanent dataset/ID. Save JSON manifest/configuration/checks/scores/review and CSV predictions/evaluation. Close predictions before joining actuals for scoring. Per-origin model failures/invalid outputs and scoring overflow are retained; execution continues and coverage is explicit. A failed case fails the run. Invalid input/configuration raises before run creation. Fixed checks belong to the runner; configurable plans and release rules remain open.
- **Reporting:** draw a PNG using matplotlib from saved results, with no model calls or input-store access. The library makes reporting explicit; the baseline script calls it automatically. matplotlib is a runtime dependency, declared, locked, and exported alongside the existing table readers.
- **Limits and alternatives:** a general adapter/plugin registry, execution service, larger storage layer, tuning system, real-data task, and Bayesian adapter can follow demonstrated needs. Local run files remain writable; existing output directories/charts are rejected. Interrupted runs may retain `running` status. Source hashes and environment versions identify the implementation but do not restore it; callers preserve matching code, environment, and inputs. Human review stays pending even when checks pass.
- **Evidence:** known forecasts and scores, future-value isolation, tag/store changes, failed cases, overflow, and reporting from saved artifacts are covered by behavioral tests. See [the executable workflow and reproduction instructions](forecasting.md).

## Working proposals

- Start the Python implementation with a few files in `src/model_pipelining/`; package/build configuration follows the first importable code.
- Extend the first JSON/CSV run artifacts only as another model or task requires it.
- A local-level BDLM with fixed variance settings as the first substantive model.
- One market series as the first real-data application, followed later by a separate operational application.

These are starting choices to revisit deliberately, rather than implemented interfaces or completed experiments.

## Open questions

- Which additional synthetic processes would expose Bayesian or feature-model failure cases?
- Which initial runtime modules, development tools, and compatible versions should we select?
+- Which standard build backend should package the first importable implementation?
- How should the working point-forecast request grow for Bayesian outputs and specialized checks?
- Which checks stop execution, and which allow exploratory evidence collection while blocking release?
- What requires explicit human interpretation in the first review?
- Which additional artifacts are needed beyond the first inspectable JSON/CSV records?
- Which instrument/source/target would make the later real-data exercise informative?
- Which GitHub visibility and license should be selected before publication?

## Record a consequential change

Use a dated entry with the question, decision, rationale, alternatives considered, consequences, and evidence or issue/PR links. Preserve superseded choices so later changes remain understandable.
