# Small local snapshots

This is the reference for preserving and tagging a **single completed file**.
A snapshot keeps the file bytes; a registration event explains one preservation
call; a tag gives a dataset's version a readable name. Start with
[the walkthrough](forecasting.md) for model execution or
[the API reference](../src/model_pipelining/README.md) for signatures.

The implementation uses local files, SHA-256, and JSON. Snapshot/registration/tag
records use format version 1. Run-manifest version 2 belongs to the separate
forecasting layer and does not change these storage formats. The earlier
multi-file design is retained as a [historical proposal](data-versioning.md).

## Use it

After installing the existing project dependencies, launch Python from the repository root with `PYTHONPATH=src .venv/bin/python`:

```python
from model_pipelining.ingest import read_local_table
from model_pipelining.snapshots import register_snapshot, retrieve_snapshot

snapshot = register_snapshot(
    "data/synthetic/mean_shift.csv",
    dataset="synthetic-mean-shift",
    note="Initial synthetic input",
)
print(snapshot.snapshot_id)

# Save the dataset and ID as the permanent reference. Later:
path = retrieve_snapshot(snapshot.snapshot_id, dataset=snapshot.dataset)
table = read_local_table(path)
```

Generate the synthetic source first if needed using `scripts/generate_synthetic.py`. An already ingested file is equally valid. Pass its receipt to preserve the original acquisition details:

```python
from model_pipelining.ingest import ingest_local_file

receipt = ingest_local_file(
    "data/synthetic/mean_shift.csv",
    "data/ingested/snapshot-example/mean_shift.csv",
)
snapshot = register_snapshot(
    receipt.destination,
    dataset="synthetic-mean-shift",
    ingestion_receipt=receipt,
    note="Preserve the original source and copy time",
)
```

Choose an unused ingestion destination when repeating this example. `ingestion_receipt` is optional and accepts an `IngestResult` from `ingest_local_file`. Registration checks that its destination resolves to the supplied file and its byte count matches, before creating store files. Its acquisition time must be timezone-aware and is saved in UTC. The original ingestion source may already have been removed. Registration's own staging copy never supplies original acquisition context automatically.

The receipt is caller-supplied historical context. It has no content hash, so the path and size checks cannot detect same-size edits after ingestion. Keep the ingested file unchanged until registration. Retrieval verifies the bytes preserved at registration.

`register_snapshot` returns a `Snapshot` dataclass containing `snapshot_id`, `dataset`, and the current local `path`. `retrieve_snapshot` returns a `Path` after checking the manifest, size, and SHA-256. It raises an error for missing or changed contents; it does not silently repair or replace them. Pass that path to the existing CSV/XLSX reader. Excel worksheet selection remains a reading option; the snapshot preserves the whole workbook.

The optional `note` is a string describing why this registration was made. It belongs to the registration record, so changing the note does not change the snapshot ID, manifest, or preserved file. Omission records `None`; empty strings, Unicode, and line breaks are preserved. Other value types are rejected before creating store files.

## Inspect saved versions and history

```python
from model_pipelining.snapshots import list_snapshots, list_acquisitions

versions = list_snapshots(dataset="synthetic-mean-shift")
history = list_acquisitions(dataset="synthetic-mean-shift")

for version in versions:
    print(version["snapshot_id"], version["filename"], version["size_bytes"])

for event in history:
    print(event["registered_at"], event["snapshot_id"], event["note"])
```

Both functions return a list of dictionaries for the named dataset and accept the same `store` argument as registration and retrieval. An absent dataset returns `[]` without creating directories.

- `list_snapshots` returns one summary per preserved snapshot: `dataset`, `snapshot_id`, `filename`, and `size_bytes`. Results are sorted by snapshot ID, not registration time.
- `list_acquisitions` returns the complete registration records, including repeat registrations of the same snapshot. Results are sorted by the recorded UTC registration timestamp, oldest first, then by acquisition ID for ties. Missing optional `note` and `ingestion_receipt` fields return `None`, including for older records; files are not rewritten.

Listing reads saved metadata without hashing or checking data-file availability. Use `retrieve_snapshot` when you need to verify the contents before use. Missing manifests and malformed JSON raise errors rather than silently hiding records. You can also inspect either list using `pandas.DataFrame(versions)` or `pandas.DataFrame(history)`.

## Assign, resolve, and inspect tags

A tag is a readable name such as `baseline-input` that belongs to one dataset. Assignment is always explicit; registering or reading data never creates or moves a tag.

```python
from model_pipelining.tags import assign_tag, resolve_tag, list_tags, list_tag_history

assignment = assign_tag(
    dataset=snapshot.dataset,
    tag="baseline-input",
    snapshot_id=snapshot.snapshot_id,
    changed_by="manual-review",
    note="Selected the initial baseline input",
)

# Resolve once and save both the dataset and this ID with a model run.
selected_snapshot_id = resolve_tag(dataset=snapshot.dataset, tag="baseline-input")
selected_file_path = retrieve_snapshot(selected_snapshot_id, dataset=snapshot.dataset)
table = read_local_table(selected_file_path)

current_tags = list_tags(dataset=snapshot.dataset)
tag_history = list_tag_history(dataset=snapshot.dataset, tag="baseline-input")
```

To move the tag, call `assign_tag` again with a different registered snapshot ID in the same dataset. The previously resolved ID still refers to the original snapshot. Multiple tags can point to the same version, and different datasets can each have a tag named `baseline-input`.

- `assign_tag` verifies the target snapshot's existence and contents before writing. It returns the saved assignment dictionary, including a unique `change_id`, `previous_snapshot_id` (`None` initially), `snapshot_id`, UTC `changed_at`, `changed_by`, and optional `note`. Every explicit assignment is recorded, including repeated assignment to the same ID.
- `changed_by` is required and names the person or process requesting the assignment. It is caller-supplied context, not authenticated identity. Notes follow the same string-or-None rule as registration notes.
- `resolve_tag` returns the current snapshot ID, or raises `FileNotFoundError` if the tag has never been set. Resolution itself reads the reference; use `retrieve_snapshot` to verify the file before use.
- `list_tags` returns current targets and last-assignment details as dictionaries sorted by tag name. A dataset with no tags returns `[]`.
- `list_tag_history` returns all assignments in their saved sequence. It returns `[]` for a never-assigned tag. Saved sequence is authoritative even when timestamps tie or the clock moves backwards.

All four functions accept `store`. Tag names follow the same letter/digit/underscore/hyphen rules as dataset names. Assignment does not alter snapshot manifests or contents, execute models, or establish data-quality eligibility. Tag deletion, renaming, and automatic advancement are outside this slice.

## Identity and repeated registration

- The ID is the lowercase SHA-256 hash of the preserved file's exact bytes. No row hashing or table parsing occurs.
- A permanent reference consists of the dataset name and snapshot ID. A local absolute path is just the file's current location.
- Identical bytes in the same dataset reuse a snapshot, including after a rename. The first preserved filename stays in its manifest. Every successful call still creates a separate registration/acquisition record with the supplied source path.
- Identical bytes in different datasets have the same content hash but separate copies and histories. There is no cross-dataset deduplication.
- Changed bytes produce another snapshot. Earlier copies remain available. XLSX formatting or changes to another worksheet can also change the workbook's bytes and ID.
- A registration verifies an existing snapshot before reusing it. A new registration copies into staging, calculates the copied file's hash, writes its manifest, and then moves the completed directory into place.

Dataset names start with an ASCII letter or digit and otherwise contain ASCII letters, digits, underscores, or hyphens. Names are case-sensitive. The store defaults to `data/versioned`; pass the same `store=...` to all snapshot/tag calls that use another location.

The byte-only identity rule is a deliberate simplification for this single-file slice. The older multi-file proposal includes filenames and inventory membership in identity. That future collection contract has not been implemented; do not assume these single-file IDs already identify collections.

## Files on disk

```text
data/versioned/
  synthetic-mean-shift/
    snapshots/
      <sha256>/
        manifest.json
        content/
          mean_shift.csv
    acquisitions/
      <unique-registration-id>.json
    tags/
      baseline-input.json
```

The immutable-by-convention manifest has only `format_version`, `filename`, `sha256`, and `size_bytes`. Registration times and source paths are outside it, so repeated registration does not change the snapshot manifest or identity.

Each acquisition record has `format_version`, a unique `acquisition_id`, `dataset`, `snapshot_id`, `source_path`, `registered_at` in UTC, an optional `note`, and an optional `ingestion_receipt`. `source_path` identifies the local file supplied to registration. `registered_at` describes registration, separately from the earlier ingestion copy's completion time. Neither time establishes historical availability. The note is user-provided text; no credentials or arbitrary request details are automatically collected.

When supplied, `ingestion_receipt` is a dictionary with the receipt's four fields:

| Field | Saved value |
|---|---|
| `source` | Original ingestion source path as text |
| `destination` | Completed ingestion copy's path as text |
| `acquired_at` | UTC ISO timestamp when that copy finished |
| `size_bytes` | Byte count recorded by ingestion |

Omission saves `ingestion_receipt: null`. Different receipts for identical bytes belong to separate registration records; the snapshot ID and manifest remain unchanged. These optional fields extend format version 1; older records remain readable without migration. Moving the store does not rewrite historical receipt paths, and inspecting receipts does not require those paths to exist.

Each tag JSON file has `format_version`, `dataset`, `tag`, and `history`. Its final history entry identifies the current target. Assignment writes the updated file into temporary space on the same filesystem and replaces the previous tag file in one operation. This keeps the target and its history together without a separate mutable pointer file. Existing histories are retained by the API; they remain editable files, not a tamper-proof audit log. Malformed JSON or mismatched tag metadata raises an error instead of being silently overwritten.

Move or copy the whole store to preserve manifests and histories together with data. References continue to resolve through the new `store` argument; historical `source_path` entries remain contextual records of where registration read from.

Everything under `data/versioned` is ignored by Git. Committing the implementation does not preserve or back up this local store.

## Limits and later enhancements

The caller supplies a completed source file that remains unchanged during copying and uses one writer at a time. Store files remain writable: preservation is a convention, with changes detected by retrieval verification. Hashes do not restore lost bytes.

Staging prevents a handled copy failure from publishing a partial snapshot, and acquisition records are moved into place only after writing completes. This is not a crash-durable transaction across snapshot and event publication. If recording an event fails after a snapshot is published, the function raises and may leave a complete snapshot without that event; retrying verifies and reuses the snapshot. Machine-crash recovery and cleanup of abandoned staging directories are deferred.

Tag publication also uses a completed staged file: a handled failure before its replacement leaves the previous tag record unchanged. Tag updates still require one writer at a time and do not promise durability after a power loss. Move the tags directory with the rest of the store to preserve readable references and their histories.

Keep these enhancements for later review:

- More tag operations only when needed, such as deletion, renaming, or configurable experiment-selection rules. Basic dataset-scoped tags and assignment history are implemented.
- Multi-file inventories, supporting documents, processing lineage, and an explicit format migration policy.
- Source-specific acquisition context beyond the implemented local ingestion receipt: remote source/version, query parameters, and meaningful availability assumptions.
- Broader data-quality evidence and runner eligibility, separate from content integrity. The [first backtest](forecasting.md) now verifies its selected snapshot and separately validates a daily CSV; reusable quality-evidence records remain future work.
- Durable backup/remote storage and an integration with git-annex if its benefits become useful. It is no longer a required next step.
- Concurrent writers, crash recovery, retention, and richer inspection APIs.
- DVC or another dependency/stage/execution tool, independently of this storage slice.

The broader requirements and earlier discussion remain in [data-versioning.md](data-versioning.md).

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`tests/test_snapshots.py` covers repeated and renamed inputs, separate dataset histories, revised old observations, tampered data and manifests, failed copying, store relocation, invalid references, and reading a preserved XLSX worksheet. It also checks dataset-scoped listings, time ordering and ties, empty stores, malformed records, notes and ingestion receipts independent of snapshot identity, and older records without optional fields. Receipt tests cover source removal, separate copy/registration times, UTC normalization, wrong destinations, changed byte counts, invalid types, and missing timezones. Fixtures use temporary directories.

`tests/test_tags.py` covers assignment and movement, permanent IDs resolved before a move, explicit-only changes, multiple tags and separate datasets, invalid or corrupt targets, invalid arguments, failed publication, malformed records, store relocation, and history order when clock time moves backwards. It also uses temporary fixtures.
