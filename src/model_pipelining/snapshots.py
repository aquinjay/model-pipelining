"""Preserve file versions and explain when each file was registered.

A snapshot is one preserved version of a file, identified by its SHA-256 hash.
An acquisition record describes one successful registration of that version.
Registering identical bytes twice keeps one snapshot and two acquisition records.

The public functions register, retrieve, or list records. Functions whose names
start with an underscore are internal helpers used by this module.

Names ending in _directory refer to folders; names ending in _file_path refer
to files. A staged file or directory is a temporary copy being prepared before
it becomes a preserved snapshot or a saved acquisition record.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from model_pipelining.ingest import IngestResult, ingest_local_file


@dataclass(frozen=True)
class Snapshot:
    """Receipt returned by register_snapshot.

    Keep dataset and snapshot_id as the permanent reference. The path points to
    the current local copy. After moving the store, retrieve again to get its
    new path. frozen=True prevents field reassignment; it does not make the
    file on disk read-only.
    """

    snapshot_id: str
    dataset: str
    path: Path


def _get_dataset_directory(store: str | Path, dataset: str) -> Path:
    """Validate the dataset name and return its absolute directory path.

    This calculates a path without creating the directory. The name must start
    with a letter or digit; subsequent characters can also be _ or -.
    """
    # fullmatch checks the entire name, excluding separators such as ../ or /.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
        raise ValueError("dataset must use letters, digits, underscores, or hyphens")
    return Path(store).expanduser().resolve() / dataset


def _calculate_file_sha256(file_path: Path) -> str:
    """Return a file's fingerprint as 64 lowercase hexadecimal characters."""
    # Hash raw bytes in chunks, so CSV and XLSX files use the same operation
    # and the whole file does not need to fit in memory. hexdigest returns text.
    with file_path.open("rb") as file_stream:
        return hashlib.file_digest(file_stream, "sha256").hexdigest()


def list_snapshots(
    *,
    dataset: str,
    store: str | Path = "data/versioned",
) -> list[dict]:
    """List snapshot metadata in ID order; use retrieval to verify file contents.

    Each dictionary contains dataset, snapshot_id, filename, and size_bytes.
    An absent dataset returns an empty list without creating directories.
    Missing or malformed manifests raise an error rather than hiding a version.
    """
    snapshots_directory = _get_dataset_directory(store, dataset) / "snapshots"
    if not snapshots_directory.exists():
        return []

    snapshot_summaries = []
    # Sorting directory names gives a repeatable order because each name is an ID.
    for snapshot_directory in sorted(snapshots_directory.iterdir()):
        manifest_file_path = snapshot_directory / "manifest.json"
        manifest = json.loads(manifest_file_path.read_text(encoding="utf-8"))
        if manifest["format_version"] != 1 or manifest["sha256"] != snapshot_directory.name:
            raise ValueError(f"Invalid snapshot manifest: {snapshot_directory}")
        snapshot_summary = {
            "dataset": dataset,
            "snapshot_id": snapshot_directory.name,
            "filename": manifest["filename"],
            "size_bytes": manifest["size_bytes"],
        }
        snapshot_summaries.append(snapshot_summary)
    return snapshot_summaries


def list_acquisitions(
    *,
    dataset: str,
    store: str | Path = "data/versioned",
) -> list[dict]:
    """List registration records oldest first, including repeated acquisitions.

    The dictionaries include source_path, registered_at, snapshot_id, note,
    and an optional ingestion_receipt, along with the saved format version,
    dataset name, and acquisition ID. Equal timestamps are ordered by acquisition
    ID. Missing optional fields return None without modifying stored records.
    """
    acquisitions_directory = _get_dataset_directory(store, dataset) / "acquisitions"
    acquisition_records = []
    # glob finds registration JSON files; an absent directory yields no matches.
    for acquisition_record_path in acquisitions_directory.glob("*.json"):
        acquisition_record = json.loads(acquisition_record_path.read_text(encoding="utf-8"))
        # Supply the newer optional field in memory only. Do not rewrite history.
        acquisition_record.setdefault("note", None)
        acquisition_record.setdefault("ingestion_receipt", None)
        acquisition_records.append(acquisition_record)

    # Writer-generated UTC ISO timestamps sort chronologically. The event ID
    # makes the order repeatable when two registrations share a timestamp.
    return sorted(
        acquisition_records,
        key=lambda acquisition_record: (
            acquisition_record["registered_at"],
            acquisition_record["acquisition_id"],
        ),
    )


def retrieve_snapshot(
    snapshot_id: str,
    *,
    dataset: str,
    store: str | Path = "data/versioned",
) -> Path:
    """Find a preserved file and verify its manifest, byte count, and hash.

    Keep the dataset and snapshot ID to identify an input permanently. The
    returned path depends on the current store location. This does not parse
    the file or establish that its contents are suitable for modeling.
    """
    dataset_directory = _get_dataset_directory(store, dataset)
    # A SHA-256 hex ID contains exactly 64 characters, each 0-9 or a-f.
    if not re.fullmatch(r"[0-9a-f]{64}", snapshot_id):
        raise ValueError("snapshot_id must be a lowercase SHA-256 hexadecimal string")

    snapshot_directory = dataset_directory / "snapshots" / snapshot_id
    manifest_file_path = snapshot_directory / "manifest.json"
    manifest = json.loads(manifest_file_path.read_text(encoding="utf-8"))
    # The manifest must use our supported layout and describe the requested ID.
    if manifest["format_version"] != 1 or manifest["sha256"] != snapshot_id:
        raise ValueError("Snapshot manifest does not match the requested identity")
    filename = manifest["filename"]
    # Only a filename belongs here; a relative path could point outside content/.
    if not filename or Path(filename).name != filename or filename in (".", ".."):
        raise ValueError("Snapshot manifest must contain a single filename")

    preserved_file_path = snapshot_directory / "content" / filename
    # Size catches truncation quickly. Hashing also catches same-size edits.
    if (
        preserved_file_path.stat().st_size != manifest["size_bytes"]
        or _calculate_file_sha256(preserved_file_path) != snapshot_id
    ):
        raise ValueError(f"Snapshot integrity check failed: {snapshot_id}")
    return preserved_file_path


def register_snapshot(
    path: str | Path,
    *,
    dataset: str,
    note: str | None = None,
    ingestion_receipt: IngestResult | None = None,
    store: str | Path = "data/versioned",
) -> Snapshot:
    """Preserve a file, reuse matching bytes, and record this registration.

    path is a completed input file. dataset names its version history. store is
    the local root directory (data/versioned by default). note explains this
    particular registration and may be omitted. ingestion_receipt optionally
    carries the earlier copy's source, destination, UTC completion time, and
    size into this registration record. Its destination must resolve to path
    and its byte count must match. Return a Snapshot receipt.

    The original must remain unchanged during this call. Identity is based on
    bytes within a dataset; repeated or renamed inputs retain the first saved
    filename. Local paths, registration times, and optional notes live in
    separate event files. Notes and receipts do not affect the content ID or
    manifest. The receipt is caller-supplied context, not proof that the file
    has remained unchanged since ingestion; it contains no content hash.
    Use one writer at a time; remote storage and crash recovery are deferred.
    """
    # Check inputs before creating directories or copying data.
    if note is not None and not isinstance(note, str):
        raise TypeError("note must be a string or None")
    source_file_path = Path(path).expanduser().resolve(strict=True)
    if not source_file_path.is_file():
        raise ValueError(f"Source must be a regular file: {source_file_path}")
    # Keep the original ingestion separate from our internal staging copy.
    # The original source need not still exist: this receipt describes history.
    ingestion_context = None
    if ingestion_receipt is not None:
        if not isinstance(ingestion_receipt, IngestResult):
            raise TypeError("ingestion_receipt must be an IngestResult or None")
        if ingestion_receipt.destination.resolve() != source_file_path:
            raise ValueError("ingestion_receipt destination must match the registered path")
        if ingestion_receipt.size_bytes != source_file_path.stat().st_size:
            raise ValueError("ingestion_receipt size_bytes must match the registered file")
        if ingestion_receipt.acquired_at.utcoffset() is None:
            raise ValueError("ingestion_receipt acquired_at must be timezone-aware")
        ingestion_context = {
            "source": str(ingestion_receipt.source),
            "destination": str(ingestion_receipt.destination),
            "acquired_at": ingestion_receipt.acquired_at.astimezone(UTC).isoformat(),
            "size_bytes": ingestion_receipt.size_bytes,
        }
    dataset_directory = _get_dataset_directory(store, dataset)
    dataset_directory.mkdir(parents=True, exist_ok=True)

    # Prepare a complete snapshot in temporary space on this filesystem. The
    # context manager removes anything left here after success or a handled error.
    with TemporaryDirectory(prefix=".register-", dir=dataset_directory) as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        staged_snapshot_directory = temporary_directory_path / "snapshot"
        staged_file_path = staged_snapshot_directory / "content" / source_file_path.name
        staging_receipt = ingest_local_file(source_file_path, staged_file_path)

        # Hash the completed copy we will preserve, then describe it in a manifest.
        snapshot_id = _calculate_file_sha256(staging_receipt.destination)
        snapshot_manifest = {
            "format_version": 1,
            "filename": source_file_path.name,
            "sha256": snapshot_id,
            "size_bytes": staging_receipt.size_bytes,
        }
        manifest_file_path = staged_snapshot_directory / "manifest.json"
        manifest_file_path.write_text(
            json.dumps(snapshot_manifest, indent=2) + "\n", encoding="utf-8",
        )

        snapshot_directory = dataset_directory / "snapshots" / snapshot_id
        snapshot_directory.parent.mkdir(exist_ok=True)
        if snapshot_directory.exists():
            # Reuse only after checking that the preserved copy is still intact.
            preserved_file_path = retrieve_snapshot(snapshot_id, dataset=dataset, store=store)
        else:
            # Moving the completed directory makes the manifest and data appear
            # together. Readers never need to use a half-copied staged snapshot.
            staged_snapshot_directory.rename(snapshot_directory)
            preserved_file_path = snapshot_directory / "content" / source_file_path.name

        # A random event ID distinguishes repeated registrations of the same
        # content ID. Time and notes belong to the event, outside the manifest.
        acquisition_id = uuid4().hex
        acquisition_record = {
            "format_version": 1,
            "acquisition_id": acquisition_id,
            "dataset": dataset,
            "snapshot_id": snapshot_id,
            "source_path": str(source_file_path),
            "registered_at": datetime.now(UTC).isoformat(),
            "note": note,
            "ingestion_receipt": ingestion_context,
        }
        # Finish writing the event in temporary space before publishing it, too.
        staged_acquisition_record_path = temporary_directory_path / "acquisition.json"
        staged_acquisition_record_path.write_text(
            json.dumps(acquisition_record, indent=2) + "\n", encoding="utf-8",
        )
        acquisitions_directory = dataset_directory / "acquisitions"
        acquisitions_directory.mkdir(exist_ok=True)
        acquisition_record_path = acquisitions_directory / f"{acquisition_id}.json"
        staged_acquisition_record_path.rename(acquisition_record_path)

    return Snapshot(snapshot_id=snapshot_id, dataset=dataset, path=preserved_file_path)
