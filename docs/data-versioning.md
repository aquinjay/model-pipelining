# Data versioning: decisions and next steps

Recorded: 2026-11-03.

Current update: 2026-11-09. A [small single-file snapshot implementation](snapshots.md) now registers, preserves, and verifies local files with standard-library hashing and JSON records. [Explicit dataset-scoped tags](snapshots.md#assign-resolve-and-inspect-tags) support assignment, resolution, listing, and history. This supersedes git-annex as the next required integration. Single-file IDs hash bytes only; renamed identical inputs reuse a snapshot, whereas this older collection proposal includes filenames in identity. The remaining broader requirements below are saved for later, not claims about the first implementation.

Earlier update: 2026-11-09. The [ingestion-first decision](decisions.md#ingestion-first-preservation-and-execution-later--2026-11-09) changed the immediate sequence to start with local-file ingestion and defer dependency/stage/execution tooling. Its git-annex direction has since been superseded by the current update above. The requirements below remain saved for later; their original ordered implementation plan is historical.

Historical status at the original stopping point: agreed direction from the design discussion, saved for review before implementation. At that point, the registration system, tag registry, storage interface, and model integration did not exist. See the current update above for the narrower implementation that followed.

## Start here next time

Use [the roadmap's current resume section](roadmap.md#resume-here) as the continuation plan. This document preserves the broader design and its earlier alternatives; its original implementation sequence below is historical.

Ingestion and CSV/XLSX reading were first published in `685b7fb` on `main`. The subsequent snapshot/tagging slice implements single-file snapshot registration and verified retrieval, inventory/history listing, registration notes, dataset-scoped tags, and tag assignment history. The user requested publication of this slice and the updated documentation on 2026-11-09. Its behavior and examples are documented in [the source-directory function reference](../src/model_pipelining/README.md) and [snapshot/tag notes](snapshots.md). All 50 tests passed again before publication; the README workflow was also exercised with synthetic data in a temporary store.

Resume from those implemented functions. A proposed following slice is preserving original ingestion source and acquisition-time context during registration. Its interface remains undecided; do not treat it as approved implementation work. Additional versioning requirements and the baseline experiment design remain future work. Keep future changes small and understandable to the user.

The purpose still holds: every future model output should identify its exact input data even when history is revised or tags move. Current single-file references use the dataset plus SHA-256 of bytes, and registration time is distinct from acquisition time and historical availability. The original `feature/data-versioning` branch suggestion is part of the earlier plan, not an outstanding instruction to restart the completed implementation.

## Agreed decisions

### 1. Acquisition and registration are separate

A person or automated process can download files, call an API, run a query, or generate synthetic data using its own tools. Registration accepts the completed result. A caller can make registration the next step of an automated acquisition, but acquisition connectors are not required for the first version.

Initial flow:

```text
acquire data → register snapshot → explicitly assign/select a tag
            → explicitly start processing or a model run
```

Assigning a tag does not itself start processing or model execution.

### 2. A snapshot can contain one or more related files

A snapshot preserves a collection that should be selected together. It can contain one data file or several related data files, plus supporting information such as a saved query and a manifest.

For example, a query and its single result table fit this design, as do several tables that must be joined together. Sharing a download date alone is not a reason to group unrelated data.

### 3. Compare actual contents before reusing a snapshot

Calculate SHA-256 hashes of the completed data files. Compare the complete collection of data filenames and hashes, rather than assuming that the same query, URL, or requested dates means the same data.

- Matching contents reuse an existing snapshot.
- Added, removed, renamed, or changed data files produce a new snapshot.
- Every acquisition gets a separate provenance record, including repeated acquisitions of identical content.
- A different query returning identical data retains its own acquisition record.

The first version uses exact file contents. Formatting, row order, or serialization changes can therefore create a new snapshot even when the table has the same meaning. Semantic table comparison can come later.

This detects retroactive source changes only when the affected history is fetched again. Fetching only new dates does not check older observations for revisions. Hashes detect differences; they do not explain which rows changed or recover a lost file.

### 4. Snapshot IDs are permanent; tags can move

A registered snapshot has a permanent identity and preserved contents. Tags are readable references that can be moved to another snapshot, with a history of those changes.

Every model run resolves its selected tag to a permanent snapshot ID and saves that ID. Reproducing a run uses the saved ID, regardless of where the tag points later. Multiple tags can point to the same snapshot.

The exact snapshot ID format has not been chosen. SHA-256 file checksums are agreed; using a combined content hash as the snapshot ID remains a design detail to review.

### 5. Tags belong to a named dataset

A dataset groups successive snapshots of an input. Tags are scoped to that dataset, so unrelated inputs can each have a tag such as `latest-snapshot-processed` without competing for one global name.

Conceptually, selection uses both a dataset name and a tag:

```text
synthetic-constant-mean / latest-snapshot-processed → snapshot A
synthetic-mean-shift    / latest-snapshot-processed → snapshot B
synthetic-mean-shift    / model-test-004            → snapshot B
```

These names illustrate the behavior. Exact naming rules and command syntax are not settled.

### 6. Tag updates are explicit and support automation

Registration does not automatically advance a `latest` tag. A caller explicitly requests a tag update. That caller can be a person or an automated workflow; this does not require an interactive confirmation step.

An acquisition can request registration and a tag update in one operation. Another acquisition can preserve a snapshot without changing any tags. Record who or what changed a tag and when.

Model workflows can have explicit selection rules, such as accepting only tags matching `model-test-<number>`. This makes tagging useful for choosing experimental inputs without making the tag itself execute a model. The exact pattern is configurable and has not been fixed.

### 7. Processing creates a derived snapshot when contents change

Preserve the raw snapshot. Processing that changes the data produces a derived snapshot linked to its input snapshot or snapshots. Do not relabel changed data as though it were the original immutable snapshot.

The intended workflow looks like this:

```text
dataset: synthetic-mean-shift

raw snapshot A
  latest-snapshot-unprocessed → A

processed snapshot B, derived from A
  latest-snapshot-processed → B

explicit experiment selection
  model-test-004 → B

model run
  records permanent snapshot ID B
```

The raw and processed tags can coexist. Later tag moves do not change earlier run inputs.

### 8. Preserve data that fails quality checks

Registration records what was received. Validation records whether it meets the requirements of a workflow. Preserve completed acquisitions with quality problems, together with their check results, so they can be inspected or used for deliberate experiments.

A preserved snapshot is not automatically suitable for training. Passing, failing, and not-yet-performed checks must remain distinguishable. A test tag does not erase failed checks.

An interrupted copy or failed integrity verification is a registration failure, not a successfully preserved snapshot with a data-quality warning. The implementation must publish only complete, verified snapshots.

### 9. Required check failures block runs by default

Allow an explicit experimental override. Record the failed checks, the override, and the reason with the run, and make its experimental status clear.

Selecting a test tag alone does not override validation. Automation can explicitly request an experimental override. An override permits an experiment; it does not turn failed evidence into passed evidence or establish release readiness.

Experiments still own their required checks and failure policies, as described in [the architecture](architecture.md).

### 10. Start locally with a replaceable storage destination

The first storage implementation is a local directory ignored by Git. Keep snapshot identity independent of storage location. Runs reference snapshot IDs rather than machine-specific paths.

Use a small storage boundary for saving a complete snapshot, retrieving it, and checking whether it exists. Future implementations can send the same snapshot to another destination without changing its identity. Verify retrieved contents against the manifest.

The versioning system owns snapshot identities, manifests, hashes, provenance, lineage, tag history, and validation evidence. A storage implementation handles destination-specific access and transfers. Recovery, replication, and access across machines depend on the chosen destination and its configuration; the project is not implementing its own replication service.

Support additional destinations when needed. A universal storage plugin framework and multiple remote implementations are outside the first slice.

## Information we need to preserve

These are record responsibilities, not a finalized schema:

| Record | Purpose |
|---|---|
| Dataset | Stable context for related snapshots and scoped tags |
| Snapshot manifest | Permanent ID, named file inventory, sizes/checksums, and format version |
| Acquisition | When and how data was obtained, with available source/query or generator details; repeated retrievals remain separate events |
| Processing lineage | Parent snapshot references and the transformation needed to explain the derived output |
| Validation evidence | Checks performed, outcomes, requirements, and the versions/context used |
| Tag history | Dataset, tag, old and new snapshot references, time, and initiating person or process |
| Run input record | Resolved snapshot IDs, selected files and training window, selection context, and any experimental override |

Do not make acquisition timestamps or later tag changes modify immutable snapshot contents. The precise separation between immutable manifests and additional event records needs to be designed.

For forecasting, distinguish observation time, acquisition time, and historical availability. A historical series downloaded today does not prove what was available at a past forecast origin. Preserve availability assumptions and unknowns rather than treating acquisition time as historical evidence.

## Open details to review together

These have not been selected yet:

1. **Identity and matching scope:** the snapshot ID format; deterministic ordering of file inventories; whether identical contents in different named datasets share storage or remain separate registrations. Source context must remain available either way.
2. **Supporting files:** how saved queries and other provenance documents are retained without turning every acquisition timestamp or request change into a different data identity.
3. **Dataset and tag conventions:** names, how processed outputs relate to named datasets, and whether the two existing synthetic scenarios should be separate datasets or one bundle. The examples above are illustrative.
4. **Required provenance:** which fields are mandatory for manual imports versus automation, how unknown details are represented, and how to keep credentials out of saved requests.
5. **Validation contract:** the first concrete checks, who defines requirements, and how missing or errored required evidence affects eligibility. Define the override record alongside the future runner.
6. **Persistence and portability:** local layout and record formats, including where dataset metadata, acquisitions, tags, history, and validation evidence live. Moving the data alone is insufficient if the records needed to interpret it are lost.
7. **Interrupted and competing operations:** complete publication, safe tag updates, and behavior when two callers register the same content. Choose the simplest correct local behavior before designing remote concurrency.

Local snapshots remain vulnerable to machine loss or deletion. Before depending on real data that cannot be reconstructed, choose a durable second copy or an appropriate storage destination. No external storage service has been selected.

## Original ordered plan — historical

This was the sequence saved before implementation. Use [the current roadmap](roadmap.md#resume-here) to resume; do not interpret the completed steps here as still pending. Multi-file collections, processing lineage, validation evidence, and runner integration in this original scope remain deferred.

1. **Review this document together.** Confirm it captures the decisions and resolve the open details needed for the first slice. Do not treat illustrative names or proposed schemas as already approved.
2. **Create the data-versioning branch.** Keep this work separate from the committed synthetic generator. Select packaging and tools only as needed for the first importable implementation.
3. **Write the smallest record and storage contract.** Specify dataset identity, snapshot inventory and matching, acquisition records, tag updates, validation evidence, and a local destination. Establish what travels with a snapshot when storage changes.
4. **Implement local registration and verification.** Copy completed inputs, preserve them, calculate hashes, detect repeated or changed contents, record acquisitions, and publish only complete snapshots.
5. **Add dataset-scoped tag operations.** Explicit assignment/movement, resolution to permanent IDs, and an inspectable history. Assigning a tag must not execute models.
6. **Exercise the workflow with our synthetic data.** Register identical data twice, modify an old observation and register again, create a derived snapshot, move tags, and inspect the preserved history and validation results.
7. **Review the evidence before building the baseline runner.** The future runner must pin input IDs and enforce required checks with explicit experimental overrides. Do not claim this integration exists before executable model code is added.

## Original acceptance evidence — broader than the current slice

The single-file implementation covers content reuse and revisions, explicit tag movement and fixed references, failed-copy handling, integrity verification, store relocation, and Git exclusion. Collection inventories, processing lineage, data-quality evidence, and a replaceable storage-backend interface are still future requirements; the list below preserves the original broader scope.

- Repeated identical inputs reuse a snapshot and create another acquisition record.
- An old-value revision creates a new snapshot, while the previous contents remain retrievable.
- Related files are verified as a complete collection; changed membership is detected.
- Tag movement is explicit, scoped to a dataset, and recorded. Resolving a tag produces a permanent ID that survives later moves.
- Processing preserves parent references and the original data.
- Data-quality failures remain inspectable; incomplete writes never appear as successful registrations.
- Tampered stored contents fail integrity verification.
- Saving and retrieving through the local storage boundary preserves identity and contents without requiring a fixed absolute path.
- Generated and registered data remain ignored by Git; scripts and documentation remain eligible for version control.

Runner acceptance evidence comes later: default blocking on required failures, recorded experimental overrides, and outputs that identify their exact input snapshots and training selection.

## Save for later

API/database acquisition connectors, file watchers, automatic execution triggered by tags, dashboards and searchable catalogs, semantic table comparison, row-level revision reports, remote storage integrations, coordinated writes across machines, retention/deletion automation, and custom recovery or replication systems.

The small local registration and reference system is now implemented for review. Extending it to another storage destination is future work. This remains a saved list of possibilities rather than a commitment to build every extension.
