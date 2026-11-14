# Python API reference

Use this page to look up arguments, return types, and records. For a complete
example with explanations, start with [the forecasting walkthrough](../../docs/forecasting.md).
All relative paths below are relative to the repository root. Launch Python with:

```bash
PYTHONPATH=src .venv/bin/python
```

A `*` in a signature means the following arguments must be named. Public
functions are listed below; names starting with `_` are implementation helpers.

| Module | Responsibility |
|---|---|
| [ingest.py](ingest.py) | Copy a completed file and read a CSV/XLSX table |
| [snapshots.py](snapshots.py) | Preserve exact file bytes, verify retrieval, inspect registrations |
| [tags.py](tags.py) | Assign readable dataset references and retain assignment history |
| [models.py](models.py) | Shared requests/results, capabilities, and the training mean |
| [local_level.py](local_level.py) | Gaussian local-level adapter, filtering, prior simulation |
| [uncertainty.py](uncertainty.py) | Shared output validation and compatible uncertainty evaluation |
| [backtest.py](backtest.py) | Daily history selection, model execution, frozen forecasts, scoring, checks |
| [reporting.py](reporting.py) | Read saved artifacts and draw/compare results without fitting |

## Terms and record boundaries

| Term | Meaning |
|---|---|
| Dataset | A named history of input versions, such as `synthetic-mean-shift` |
| Snapshot | One preserved file version; referenced by dataset plus snapshot ID |
| Snapshot ID | Lowercase 64-character SHA-256 of exact file bytes; identifies but cannot recreate missing bytes |
| Tag | A movable readable reference within one dataset |
| Store | Preserved files and their metadata; defaults to `data/versioned` |
| Registration/acquisition record | One successful `register_snapshot` event; may carry an earlier ingestion receipt |
| Snapshot manifest | File identity/layout record, format version 1 |
| Run manifest | Input/task/model/evaluation/environment/status record, format version 2 for new runs |
| Run | One model/configuration executed over selected historical forecast cases |

The different manifest versions apply to different record types. Adding run
format version 2 did not change snapshot, acquisition, or tag format version 1.
Acquisition time, registration time, and historical observation availability are
separate concepts. The first two do not establish the third.

## Copying and table reading

```text
ingest_local_file(source, destination) -> IngestResult
read_local_table(path, *, sheet=None, header=0) -> pandas.DataFrame
```

Source/destination/path arguments accept strings or `Path` objects.
`ingest_local_file` copies one regular file, creates missing parent directories,
and refuses existing destinations (including symlinks). A handled copy failure
removes only its partial output. The source must remain unchanged during copying.

`IngestResult` is a frozen dataclass:

| Field | Value |
|---|---|
| `source` | Resolved original local `Path` |
| `destination` | Resolved completed-copy `Path` |
| `acquired_at` | Timezone-aware UTC copy-completion time |
| `size_bytes` | Copied byte count |

The reader infers CSV/XLSX from the case-insensitive extension and returns one
DataFrame. For XLSX, `sheet` is a worksheet name or nonnegative zero-based index;
omission selects the first sheet. CSV rejects `sheet`. `header` is a nonnegative
zero-based row index or `None` for no header. pandas infers types/missing values;
the function does not clean data, convert application timestamps, or calculate
Excel formulas. Reading XLSX is supported here; the forecasting runner accepts
CSV only. [The ingestion guide](../../docs/ingestion.md) covers detailed behavior.

## One complete workflow

**This example covers data preparation only:** copy → preserve → tag → select →
read. Continue with [the forecasting walkthrough](../../docs/forecasting.md) to
run models. Generate the source first if absent with
`.venv/bin/python scripts/generate_synthetic.py`.

```python
from model_pipelining.ingest import ingest_local_file, read_local_table
from model_pipelining.snapshots import register_snapshot, retrieve_snapshot
from model_pipelining.tags import assign_tag, resolve_tag

receipt = ingest_local_file(
    "data/synthetic/mean_shift.csv",
    "data/ingested/api-reference-001/mean_shift.csv",
)
snapshot = register_snapshot(
    receipt.destination,
    dataset="synthetic-mean-shift",
    ingestion_receipt=receipt,
    note="Input prepared using the API reference",
)
assign_tag(
    dataset=snapshot.dataset, tag="baseline-input",
    snapshot_id=snapshot.snapshot_id, changed_by="api-example",
)
selected_snapshot_id = resolve_tag(dataset=snapshot.dataset, tag="baseline-input")
preserved_file = retrieve_snapshot(selected_snapshot_id, dataset=snapshot.dataset)
table = read_local_table(preserved_file)
```

Choose an unused ingestion destination if repeating this example. Copying does
not register/tag a snapshot. Passing `ingestion_receipt=receipt` makes registration
save that context automatically; no separate receipt-saving call is needed.
You can also register an existing completed file directly without a copy/receipt.

## Preserving and inspecting versions

All functions below accept optional keyword `store="data/versioned"`:

```text
register_snapshot(path, *, dataset, note=None, ingestion_receipt=None, store=...) -> Snapshot
retrieve_snapshot(snapshot_id, *, dataset, store=...) -> Path
list_snapshots(*, dataset, store=...) -> list[dict]
list_acquisitions(*, dataset, store=...) -> list[dict]
```

`Snapshot` is a frozen dataclass with `dataset`, `snapshot_id`, and `path`.
Keep the dataset/ID as the permanent reference. The path is the current local
location; retrieve again after moving the store. Retrieval checks the manifest,
size, and hash and raises on missing/tampered contents.

Identical bytes in the same dataset reuse a snapshot, even if renamed; the first
preserved filename stays. Every successful registration adds a separate event.
Changed bytes create a new snapshot. Different datasets have separate copies and
histories even when the content hash matches.

`note` accepts a string or `None` and belongs to the event, not snapshot identity.
An optional ingestion receipt must describe the registered destination/byte count
and have an aware acquisition time. Its original source need not still exist.
The receipt has no hash; the caller must keep the acquired copy unchanged until
registration, including against same-size edits.

| Listing | Contents and order |
|---|---|
| `list_snapshots` | `dataset`, `snapshot_id`, `filename`, `size_bytes`; sorted by ID |
| `list_acquisitions` | `format_version`, `acquisition_id`, `dataset`, `snapshot_id`, `source_path`, `registered_at`, `note`, `ingestion_receipt`; oldest UTC registration first, then ID for ties |

Listings return `[]` for absent data without creating directories. They inspect
metadata, not preserved bytes. Older optional note/receipt fields return `None`
without rewriting files. Historical receipt paths remain context after moving
the store. [Snapshot files and record fields](../../docs/snapshots.md#files-on-disk)
are described in the storage reference.

## Understanding tags

All tag functions accept optional keyword `store="data/versioned"`:

```text
assign_tag(*, dataset, tag, snapshot_id, changed_by, note=None, store=...) -> dict
resolve_tag(*, dataset, tag, store=...) -> str
list_tags(*, dataset, store=...) -> list[dict]
list_tag_history(*, dataset, tag, store=...) -> list[dict]
```

Assignment verifies the target in the same dataset, then records every explicit
assignment, including repeats. Registration never creates/moves tags. Multiple
tags may select one snapshot; different datasets may use the same tag name.
Moving a tag does not alter a previously resolved permanent ID or a saved run.

`changed_by` is required, nonblank, caller-supplied context rather than
authenticated identity. Optional `note` accepts a string or `None`. The returned
event contains `change_id`, `previous_snapshot_id`, `snapshot_id`, UTC
`changed_at`, `changed_by`, and `note`.

Resolution reads the current ID and raises `FileNotFoundError` for an unassigned
tag. Retrieval separately verifies its contents. `list_tags` returns current
target/last-assignment metadata sorted by tag name. `list_tag_history` returns
all events in saved sequence, including when the clock moves backwards. Empty
listings return `[]`; history for an unassigned tag is also `[]`. See
[the tag reference](../../docs/snapshots.md#assign-resolve-and-inspect-tags).

## Backtesting and reporting

```text
run_backtest(
    *, dataset, output_directory, snapshot_id=None, tag=None,
    minimum_history=30, model=None, interval_levels=(0.8, 0.95),
    store="data/versioned",
) -> Path

plot_run(run_directory, *, interval_level=None, interval_method="auto") -> Path
compare_runs(first_run, second_run, *, output_directory) -> Path
```

Import the runner from `model_pipelining.backtest` and reports from
`model_pipelining.reporting`. Supply exactly one of `snapshot_id` and `tag`; the
runner resolves a tag once and saves its permanent ID. `model=None` selects
`TrainingMean`. For the Bayesian adapter, pass `LocalLevel(...)` from
`model_pipelining.local_level`. Every output directory must be unused.

`run_backtest` automatically verifies/parses the CSV, validates the daily series,
selects historical training data, executes the model, freezes outputs, scores,
and saves checks/review. It returns an absolute run-directory `Path`, **including
when required cases failed**. Input/configuration errors raise before output
creation. Per-case forecast failures are retained and later origins continue.
Filesystem errors propagate and can leave a `running` directory.

Read `manifest.json.status` and `checks.json` before interpreting results. Point
`scores.complete` is not the overall result: required uncertainty scoring can
fail independently. The [walkthrough](../../docs/forecasting.md#read-the-results)
explains status and [run-file reference](../../runs/README.md) lists artifacts.
The library does not acquire/register/tag data, simulate priors, or draw charts
inside `run_backtest`. The two run scripts call reporting automatically; the
local-level script also handles prior simulation/files.

`plot_run` reads saved evidence and returns the chart path. `"auto"` prefers the
widest saved highest-density region, then equal-tail intervals. Explicit
`interval_method` is `"highest_density"` or `"equal_tail"`; explicit levels must
have been saved. A point-only run has no band. Existing charts are rejected.

`compare_runs` reads completed runs with matching permanent inputs, task,
training coordinates, target dates, and actuals. It saves JSON/PNG comparison
files and returns the absolute comparison-directory path. It does not fit,
tune, or select a model. Uncertainty comparisons use common supplied kinds and
levels only; a point-only baseline has no uncertainty comparison.

## Model requests and results

`Forecaster` is a Python protocol: adapters supply a nonempty `name`, JSON
`configuration` dictionary, and `forecast(request)`. Boolean `supports_quantiles`
and `supports_highest_density` flags declare available uncertainty. Absent flags
are treated as false for legacy adapters. `PointForecaster` remains an alias.
Models refit from each request, retaining no prior request's training data.

| Immutable record | Fields |
|---|---|
| `Observation` | `timestamp`, `value` |
| `ForecastRequest` | `history` (tuple of observations), `issued_at`, `target_time`, `quantile_levels=()`, `highest_density_levels=()` |
| `Forecast` | `mean`, `quantiles=()`, `highest_density_regions=()` |
| `Quantile` | `probability`, `value` |
| `HighestDensityRegion` | `level`, `segments` (tuple of `Interval` records) |
| `Interval` | `lower`, `upper` |

The runner requests all declared uncertainty kinds at `interval_levels`.
`LocalLevel.forecast` returns `Forecast`; direct callers use `.mean` for the point
value. `TrainingMean` returns a numerical mean and declares no uncertainty.
Numeric legacy outputs are accepted for point-only adapters; declaring uncertainty
while returning only a number fails the required-output check. Direct callers constructing
a request must specify desired levels; the empty request defaults request none.

Record construction itself does not validate numerical contents; the runner
performs output validation before accepting/saving a forecast.

All shared uncertainty targets the future **observation**, not a hidden state
or parameter. [The uncertainty contract](../../docs/uncertainty.md) defines exact
validation/scoring behavior, including HPD segments and failure handling.

`LocalLevel` accepts fixed `initial_mean=0`, `initial_variance=100`,
`observation_variance=1`, and `evolution_variance=0.1` settings. Its methods also
include `predictive_distribution(request) -> NormalDist`,
`filter_history(history) -> tuple[FilterStep, ...]`, and
`simulate_prior(*, count, seed=0) -> tuple[PriorDraw, ...]`.
[The model guide](../../docs/local-level.md#library-interface-and-diagnostics)
defines the diagnostic fields, timing, and separate simulation/reporting calls.

## Naming, errors, and limits

Dataset/tag names are case-sensitive: start with an ASCII letter/digit, followed
by ASCII letters, digits, underscores, or hyphens. Invalid options/checks generally raise
`ValueError`; wrong note/receipt types raise `TypeError`; filesystem failures
retain their usual types. Malformed saved records raise rather than being
silently skipped. Individual forecast/evaluation failures are saved as described
above instead of propagating to stop all later origins.

Use completed, stable inputs and one writer. Preserved/run files remain writable;
retrieval verifies input integrity but does not recover lost bytes. Move the
whole snapshot store, including manifests/events/tags. Git excludes local data
and run evidence. Remote storage, crash recovery, retention, and backups remain
caller concerns or future work. The Python model boundary is trusted code, not a
security sandbox. [Tests](../../tests/README.md) map claims to supporting evidence.
