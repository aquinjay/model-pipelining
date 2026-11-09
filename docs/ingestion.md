# Local ingestion: first slice and later enhancements

Recorded: 2026-11-09.

Status: implemented, with behavioral tests. The local-file copy function and CSV/XLSX table reader support worksheet selection; snapshot registration, git-annex integration, and execution tooling are deferred.

## Purpose and boundary

An ingestion function obtains data from a source and produces completed local files plus acquisition context. Begin with one local file. Later functions can download from S3 or SFTP, export a Snowflake query, or use another source-specific library while returning the same small kind of result.

The acquisition function copies the source bytes to an explicit destination, including an entire XLSX workbook. A companion table reader parses a CSV or a selected XLSX worksheet into the same table type. Application-specific quality checks happen separately. A downloaded file can use the same reader later, regardless of its acquisition source.

Keep source acquisition independent of the destination used to preserve registered snapshots. Fetching from S3 and storing preserved snapshots in S3 are separate operations with potentially different accounts, locations, and credentials.

## Starting interface

The module `src/model_pipelining/ingest.py` contains the `IngestResult` dataclass, `ingest_local_file`, and `read_local_table`. From the repository root, launch Python with `PYTHONPATH=src .venv/bin/python`, then:

```python
from pathlib import Path
from model_pipelining.ingest import ingest_local_file, read_local_table

result = ingest_local_file(
    source=Path("data/synthetic/mean_shift.csv"),
    destination=Path("data/ingested/example-001/observations.csv"),
)
table = read_local_table(result.destination)
```

The example destination must not already exist. The source can be generated using `.venv/bin/python scripts/generate_synthetic.py` if it is not already present. Dependency installation is described in the main README. `PYTHONPATH=src` makes the local package importable without installing the project itself.

The result contains:

| Field | Meaning |
|---|---|
| `source` | Resolved local source path identifying where this acquisition read from |
| `destination` | Resolved path to the completed local copy |
| `acquired_at` | UTC time when this copy completed; not evidence of historical availability |
| `size_bytes` | Size of the completed copy |

These are acquisition details, not a permanent snapshot ID or a portable provenance record. A later result contract can support multiple files and source-specific context when the second source requires it.

## First-function behavior

1. Accept string or `Path` source and destination arguments. Copy one regular file, preserving its bytes and leaving the original unchanged. Resolve symlinks to a regular source file; do not ingest a directory recursively.
2. Create missing destination parent directories for convenience. The caller selects the destination; there is no automatic dataset naming, tag assignment, or unique-name generator.
3. Refuse to overwrite an existing destination, including a symlink or the source itself. Enforce this when creating the destination rather than relying only on an earlier existence check.
4. Copy incrementally using standard-library file operations so the full file need not fit in memory.
5. Return the receipt only after the copy completes and the output is closed. On a handled copy failure, remove the partial output created by this call and raise the error. Do not delete or modify pre-existing files. Empty parent directories may remain.
6. Expose useful filesystem errors, such as missing source, non-file source, existing destination, or denied access. Avoid wrapping every failure in a generic error.

The source is expected to be a completed acquisition and to remain unchanged during copying. This first slice does not promise a consistent view of a file being edited concurrently, crash-safe publication, integrity verification, immutable storage, or successful registration. Copy completion does not imply data quality checks have passed.

## CSV and XLSX table reading

Use the companion `read_local_table` function to parse a file:

```python
# Read a CSV, including a copy returned by ingest_local_file.
table = read_local_table("data/ingested/example-001/observations.csv")

# Select an Excel worksheet by name or zero-based position.
table = read_local_table("data/ingested/example-002/sales.xlsx", sheet="Sales")
table = read_local_table("data/ingested/example-002/sales.xlsx", sheet=0)
```

Implemented behavior:

- Infer CSV or XLSX from the case-insensitive file extension. Reject unsupported formats clearly.
- Return one pandas DataFrame for either format. Use `pandas.read_csv` for CSV and `pandas.read_excel` with the openpyxl engine for XLSX. pandas and openpyxl are declared in `pyproject.toml`, with exact versions in `uv.lock` and `requirements.txt`.
- For XLSX, allow `sheet` to be a worksheet name or a nonnegative, zero-based index. When omitted, read the first worksheet. Raise a useful error for a missing name or invalid index; never silently fall back to another sheet.
- Reject a sheet argument for CSV. Read one table per call; reading all worksheets at once can follow later.
- Use the first row as column headers by default and offer a small `header` option: a nonnegative, zero-based row number or `None` for no header. Keep other parsing options deferred until needed.
- Read the completed local copy so parsing and later use refer to the acquired workbook. Keep the original workbook bytes intact; selecting a worksheet does not rewrite or discard other sheets.
- Use ordinary library type/missing-value inference initially and document it. Do not add automatic column renaming, filling, deduplication, or application-specific timestamp conversions.
- Excel formula values depend on values cached by the workbook producer; the reader does not calculate formulas. Formula-dependent datasets need an explicit policy before they are used for a model run.

Acquisition and parsing have separate outcomes: a workbook can be copied successfully while a worksheet selection fails. Preserve the completed copy on a parsing error. Later run/provenance records must identify the acquired file, selected sheet, and parsing options in addition to the permanent snapshot reference.

References: [pandas CSV reader](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [pandas Excel reader and worksheet selection](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html), and [openpyxl workbook options and cached formula values](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html).

## Build and verification

Copying and the unittest test harness use the Python standard library. Table reading uses pandas and openpyxl; their dependency declarations, lockfile, and pip export are updated together. The local package has an `__init__.py` and is imported using `PYTHONPATH=src`; installable packaging remains deferred.

Run the behavioral checks from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Verify that:

- Ingesting the synthetic CSV preserves exact bytes, leaves the source unchanged, and returns the correct destination, byte count, and UTC acquisition time.
- Missing or directory sources fail without producing an output file.
- An existing destination and a source/destination referring to the same file are rejected without changing their contents.
- An injected copy failure propagates and removes the partial file created by the call.
- A completed copy can be read without relying on the original source path.
- Equivalent simple CSV and XLSX fixtures produce matching table values and column names.
- A two-sheet workbook returns the requested sheet by name and index, and defaults to its first worksheet.
- Missing sheets, out-of-range indexes, sheet arguments on CSV, and unsupported extensions fail clearly without altering the acquired file.
- A fixture with an introductory row exercises the `header` option.

Exercise the real example in an ignored directory under `data/`. Keep fixtures created by tests in temporary directories. Document one invocation; a CLI, connector registry, configuration format, and plugin framework are unnecessary for this slice.

## Save for later

- Snowflake query export, S3, SFTP, and user-supplied acquisition functions. Reuse source-specific libraries rather than implementing transport protocols.
- A shared multi-file result with source kind, query and parameters, source object versions, acquisition timing, connector/code version, and explicit unknowns. Keep credentials out of saved records.
- A clear consistency policy for files changing during acquisition and multi-file/database reads.
- Deterministic query export ordering and serialization to avoid accidental byte differences.
- Checksums and snapshot registration, stable IDs, repeat-acquisition records, lineage, dataset tags, and tag history.
- git-annex integration for preservation and retrieval. Preserve the accompanying records as well as data; define how permanent snapshot references resolve through the tool.
- DVC or another dependency/stage/execution tool. Evaluate overlap with the forecasting runner before adopting its execution features.
- CSV/schema checks, timestamp availability, and experiment-specific eligibility rules, with outcomes separate from acquisition success.
- Multiple-sheet results, sheet listing, cell ranges, explicit dtypes/date/missing-value policies, CSV dialect/encoding options, formula-cache checks, and additional spreadsheet formats as concrete needs arise. Google Sheets would be a separate acquisition source; worksheet selection here means sheets inside an XLSX workbook.
- Retries, timeouts, progress reporting, batch/directory ingestion, durable receipts, atomic publication, recovery, and concurrency, when actual use requires them.

The first deliverable is one callable local-file acquisition function, a CSV/XLSX table reader with worksheet selection, and evidence for their behavior. It is not a versioning service or a model runner.
