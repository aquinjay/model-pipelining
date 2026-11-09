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

The agreed direction, alternatives, open details, and ordered continuation plan are saved in [Data versioning: decisions and next steps](data-versioning.md). Review that document before implementation. No versioning code has been written yet.

The selected approach preserves snapshots of one or more related files, compares SHA-256 file hashes before reusing contents, and records every acquisition. Dataset-scoped tags can move only through explicit requests, including automation. Runs pin permanent snapshot IDs. Processing preserves lineage; quality failures remain visible; required failures block model runs unless an explicit experimental override is recorded. Storage starts locally behind a replaceable destination boundary.

## Ingestion first; preservation and execution later — 2026-11-09

- **Preservation tool direction:** use git-annex for preserving and retrieving versioned file contents. Integration, repository layout, and portability checks are deferred. git-annex has not been installed or configured for this project.
- **Execution tooling deferred:** consider DVC or another tool for dependencies, stages, execution, and caching when the runnable workflow needs it. DVC is a candidate, not a selected dependency. Avoid duplicating execution responsibilities between tools and the project runner.
- **Immediate priority:** begin a small library of ingestion functions with one function for a local file and a companion CSV/XLSX table reader with worksheet selection by name or index. Functions for Snowflake, S3, SFTP, and other sources can follow as concrete needs arise. Acquisition remains separate from parsing, snapshot registration, validation, and model execution.
- **Local implementation:** [Local ingestion: first slice and later enhancements](ingestion.md) describes the implemented interface and behavioral checks. pandas and openpyxl are declared, locked, and exported for table reading. Standard-library unittest is sufficient for this slice. The package is imported through `PYTHONPATH=src`; installable packaging is deferred.
- **Sequence change:** this supersedes the earlier instruction to implement the full local snapshot/tag system before the next code slice. The snapshot identity, provenance, lineage, and validation requirements above remain future requirements; the first ingestion receipt does not satisfy them by itself.

## Working proposals

- Start the Python implementation with a few files in `src/model_pipelining/`; package/build configuration follows the first importable code.
- Plain local files for configuration and artifacts; select exact formats during the first slice.
- A local-level BDLM with fixed variance settings as the first substantive model.
- One market series as the first real-data application, followed later by a separate operational application.

These are starting choices to revisit deliberately, rather than implemented interfaces or completed experiments.

## Open questions

- What synthetic process best exposes the first failure cases?
- Which initial runtime modules, development tools, and compatible versions should we select?
+- Which standard build backend should package the first importable implementation?
- What is the smallest useful request/prediction/check result contract?
- Which checks stop execution, and which allow exploratory evidence collection while blocking release?
- What requires explicit human interpretation in the first review?
- Which artifact formats balance inspectability and reproduction?
- Which instrument/source/target would make the later real-data exercise informative?
- Which GitHub visibility and license should be selected before publication?

## Record a consequential change

Use a dated entry with the question, decision, rationale, alternatives considered, consequences, and evidence or issue/PR links. Preserve superseded choices so later changes remain understandable.
