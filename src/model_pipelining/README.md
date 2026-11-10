# Working with ingestion, snapshots, and tags

This is the everyday reference for the functions in this directory: what to call, what comes back, and where the results are saved.

| Module | Responsibility |
|---|---|
| [ingest.py](ingest.py) | Copy local files and read CSV/XLSX tables |
| [snapshots.py](snapshots.py) | Preserve exact file versions, verify them, and inspect registrations |
| [tags.py](tags.py) | Give snapshots readable names and retain assignment history |

## The terms we use

| Term | Meaning |
|---|---|
| Dataset | A named collection of versions of an input, such as `synthetic-mean-shift` |
| Snapshot | One preserved version of a file |
| Snapshot ID | The SHA-256 fingerprint of the file's bytes |
| Acquisition record | One successful registration, including its local source path, registration time, and optional note |
| Manifest | The record describing a snapshot's filename, byte count, hash, and format version |
| Tag | A readable name within a dataset pointing to a snapshot, such as `baseline-input` |
| Store | The directory containing preserved files and records; defaults to `data/versioned` |

A snapshot describes which contents we have. An acquisition record describes a registration of those contents. A tag describes which version a readable name currently selects.

## Run from the project root

After following the [environment setup](../../README.md#development-environment), launch Python from the repository root, rather than this source directory:

```bash
PYTHONPATH=src .venv/bin/python
```

`PYTHONPATH=src` makes the local package importable. Relative data paths in the examples are relative to the project root.

For the complete example, generate the synthetic inputs if they are absent:

```bash
.venv/bin/python scripts/generate_synthetic.py
```

The generator refuses to overwrite its output, so do not rerun it if those files already exist.

## One complete workflow

Copy a file, preserve it as a version, give that version a name, then select and read it:

```python
from model_pipelining.ingest import ingest_local_file, read_local_table
from model_pipelining.snapshots import register_snapshot, retrieve_snapshot
from model_pipelining.tags import assign_tag, resolve_tag

# 1. Acquire a local copy. This destination must not already exist.
receipt = ingest_local_file(
    "data/synthetic/mean_shift.csv",
    "data/ingested/readme-example/mean_shift.csv",
)

# 2. Preserve the copy and record why we registered it.
snapshot = register_snapshot(
    receipt.destination,
    dataset="synthetic-mean-shift",
    note="Initial input for the baseline",
)

# 3. Explicitly give this version a readable name.
assign_tag(
    dataset=snapshot.dataset,
    tag="baseline-input",
    snapshot_id=snapshot.snapshot_id,
    changed_by="manual-review",
    note="Selected this version for development",
)

# 4. Resolve once. Save this dataset and ID for reproduction.
selected_snapshot_id = resolve_tag(
    dataset=snapshot.dataset,
    tag="baseline-input",
)

# 5. Verify the selected version and read its preserved contents.
preserved_file_path = retrieve_snapshot(
    selected_snapshot_id,
    dataset=snapshot.dataset,
)
table = read_local_table(preserved_file_path)
print(table.head())
```

To repeat the full example, choose a new ingestion destination. Repeating registration or tag assignment is allowed and records another event. You can also register an existing completed file directly; the separate ingestion copy is optional.

## Copying and table reading

| Function | What it does | Returns |
|---|---|---|
| `ingest_local_file(source, destination)` | Copies one file and creates missing destination directories; refuses to overwrite existing files | `IngestResult` receipt |
| `read_local_table(path, *, sheet=None, header=0)` | Reads a CSV or one XLSX worksheet without changing the file | pandas DataFrame: a table in memory |

`IngestResult` contains `source` and `destination` as absolute paths, UTC `acquired_at`, and `size_bytes`. Access fields with dots, such as `receipt.destination`.

Copying handles raw file bytes, including whole XLSX workbooks. It does not parse them. The source must remain unchanged while copying; a handled failure removes the partial output created by that call.

For files you have acquired yourself, table reading can look like this:

```python
table = read_local_table("data/ingested/observations.csv")
table = read_local_table("data/ingested/sales.xlsx", sheet="Sales")
table = read_local_table("data/ingested/sales.xlsx", sheet=1, header=2)
```

Options use zero-based indexes: `sheet=1` selects the second worksheet; `header=2` uses the third row as column names. Omit `sheet` for the first XLSX worksheet. Use `header=None` when every row contains data. CSV does not accept a sheet argument.

Extensions are case-insensitive. pandas infers types and missing values. Excel formulas use cached results rather than being calculated by the reader. The `*` in a signature means subsequent options must be named, such as `sheet="Sales"`.

## Preserving and inspecting versions

All snapshot functions accept `store="data/versioned"` as an optional keyword argument.

| Function | What it does | Returns |
|---|---|---|
| `register_snapshot(path, *, dataset, note=None, store=...)` | Preserves a completed file or reuses matching contents; records every successful registration | `Snapshot` receipt |
| `retrieve_snapshot(snapshot_id, *, dataset, store=...)` | Checks the manifest, byte count, and hash before locating the preserved file | `Path` |
| `list_snapshots(*, dataset, store=...)` | Lists unique versions sorted by ID | List of dictionaries |
| `list_acquisitions(*, dataset, store=...)` | Lists registrations oldest first, then by acquisition ID when times tie | List of dictionaries |

`Snapshot` contains `snapshot_id`, `dataset`, and `path`. Save the dataset and ID as the permanent reference. The path is local to the current store; retrieve again after moving it.

Identical bytes in one dataset reuse a snapshot, including renamed inputs. The first saved filename remains. Changed bytes create another version while keeping the previous one. Each successful registration gets its own acquisition record. The optional note belongs to that record and does not affect snapshot identity.

```python
from model_pipelining.snapshots import list_snapshots, list_acquisitions

versions = list_snapshots(dataset="synthetic-mean-shift")
registrations = list_acquisitions(dataset="synthetic-mean-shift")

for registration in registrations:
    print(registration["registered_at"], registration["note"])
```

Snapshot summaries contain `dataset`, `snapshot_id`, `filename`, and `size_bytes`. Acquisition records also describe the supplied `source_path`, `registered_at`, optional `note`, and unique `acquisition_id`. Older records without notes return `note=None`.

Both listing functions return `[]` for absent data without creating directories. Listing reads metadata; retrieval checks actual file integrity. Registration time does not establish when the original source was acquired or historically available.

## Understanding tags

A tag is a movable reference within one dataset:

```text
sales / baseline-input → snapshot A
```

After explicitly assigning the same tag to a revised version:

```text
sales / baseline-input → snapshot B
```

Snapshot A remains preserved. An ID resolved before the move still selects A. Another dataset, such as `weather`, can have an independent `baseline-input` tag. Multiple tags within one dataset can also point to the same snapshot.

All tag functions accept the optional `store` keyword.

| Function | What it does | Returns |
|---|---|---|
| `assign_tag(*, dataset, tag, snapshot_id, changed_by, note=None, store=...)` | Verifies the target in that dataset, then creates or moves the tag and records the assignment | Assignment dictionary |
| `resolve_tag(*, dataset, tag, store=...)` | Reads the tag's current target | Snapshot ID string |
| `list_tags(*, dataset, store=...)` | Lists current targets and last-assignment details, sorted by tag name | List of dictionaries |
| `list_tag_history(*, dataset, tag, store=...)` | Lists all assignments in saved sequence | List of dictionaries |

To move a tag, call `assign_tag` again with the revised snapshot's ID. Registering new data never automatically moves tags. Every explicit assignment adds history, including reassigning the same ID. Assignment does not change data or run a model.

`changed_by` is required: supply the person or process requesting the assignment. `note` optionally explains the selection. These are caller-supplied details rather than authenticated identity.

Each assignment dictionary contains:

| Field | Meaning |
|---|---|
| `change_id` | Unique ID for the assignment |
| `previous_snapshot_id` | Previous target, or `None` initially |
| `snapshot_id` | Assigned target |
| `changed_at` | UTC assignment time |
| `changed_by` | Supplied person/process name |
| `note` | Explanation, or `None` |

Resolution reads the reference; retrieval verifies its contents. Resolve once and save the dataset plus ID with a future experiment, so subsequent tag moves cannot change the selected input.

```python
from model_pipelining.tags import list_tags, list_tag_history

current_tags = list_tags(dataset="synthetic-mean-shift")
assignments = list_tag_history(
    dataset="synthetic-mean-shift",
    tag="baseline-input",
)

for assignment in assignments:
    print(assignment["previous_snapshot_id"], "→", assignment["snapshot_id"])
    print(assignment["changed_by"], assignment["note"])
```

Current tag summaries contain `dataset`, `tag`, `snapshot_id`, and the last assignment's `changed_at`, `changed_by`, and `note`. Listing reads metadata without verifying data files. History retains saved order even if the clock moves backwards or timestamps tie.

A never-assigned tag has an empty history and raises `FileNotFoundError` when resolved. A dataset with no tags returns `[]` from `list_tags`. Deletion, renaming, and automatic tag movement have not been implemented.

## Files saved on disk

```text
data/versioned/
  synthetic-mean-shift/
    snapshots/
      <snapshot-id>/
        manifest.json
        content/
          mean_shift.csv
    acquisitions/
      <acquisition-id>.json
    tags/
      baseline-input.json
```

Each tag file contains `format_version`, `dataset`, `tag`, and its complete `history`. The final entry determines its current target. Updating writes a completed temporary file, then replaces the previous file, keeping the target and history together.

To use another store, pass the same `store` argument to the snapshot and tag functions. Move the entire tree to preserve data and records together. The default store is ignored by Git; pushing code to GitHub does not upload these data files.

## Naming, errors, and current limits

- Dataset and tag names are case-sensitive, start with a letter or digit, and otherwise contain letters, digits, underscores, or hyphens.
- Notes accept strings or `None`. Other types raise `TypeError`.
- Missing files raise filesystem errors. An occupied ingestion destination raises `FileExistsError`; invalid options or failed integrity checks raise `ValueError`. Malformed JSON also raises an error rather than being silently skipped.
- Use one writer at a time. Inputs must remain unchanged during copying or registration. Crash recovery, remote storage, and durable backups are future work.
- Files and records remain writable. Integrity checks detect changed data but do not recover lost bytes. Assignment history is a local record rather than a tamper-proof log.
- Successful copying, registration, or tagging does not establish data quality or historical availability. No model runner has been implemented.

## Finding your way through the code

Names ending in `_directory` identify folders; `_file_path` identifies a file's location. `source_file`, `destination_file`, and `file_stream` are open file objects. A `staged_` item is being prepared in temporary space. In `Path` expressions, `/ dataset` joins the dataset name onto a directory path.

Start with `ingest.py`, then `snapshots.py`, then `tags.py`. Public functions above are the normal entry points. Helpers beginning with `_`, such as `_get_dataset_directory`, support them internally.

For tests and executable examples, see [test_ingest.py](../../tests/test_ingest.py), [test_snapshots.py](../../tests/test_snapshots.py), and [test_tags.py](../../tests/test_tags.py). Run them from the project root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

For implementation decisions and saved enhancements, see [the ingestion notes](../../docs/ingestion.md) and [snapshot/tag notes](../../docs/snapshots.md). For what is complete and where to resume next, use [the roadmap](../../docs/roadmap.md#resume-here).
