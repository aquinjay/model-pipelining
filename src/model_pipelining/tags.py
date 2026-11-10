"""Give preserved snapshots readable names with an explicit assignment history.

A tag belongs to a dataset and can move between that dataset's snapshots.
Save the snapshot ID returned by resolve_tag when an experiment selects inputs;
that permanent reference continues to work after the tag moves.

Each tag's history lives in one JSON file. Its final assignment is the current
target, so publishing that file updates the target and history together.
Use one writer at a time, as with snapshot registration.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from model_pipelining.snapshots import _get_dataset_directory, retrieve_snapshot


def _get_tag_file_path(store: str | Path, dataset: str, tag: str) -> Path:
    """Validate names and locate a tag record without creating directories."""
    dataset_directory = _get_dataset_directory(store, dataset)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
        raise ValueError("tag must use letters, digits, underscores, or hyphens")
    return dataset_directory / "tags" / f"{tag}.json"


def list_tag_history(
    *,
    dataset: str,
    tag: str,
    store: str | Path = "data/versioned",
) -> list[dict]:
    """Return assignments in saved order, or [] if the tag has never been set.

    Each assignment records previous_snapshot_id, snapshot_id, changed_at in
    UTC, changed_by, note, and a unique change_id. Stored order preserves the
    sequence even when timestamps tie or the clock changes.
    """
    tag_file_path = _get_tag_file_path(store, dataset, tag)
    if not tag_file_path.exists():
        return []

    tag_record = json.loads(tag_file_path.read_text(encoding="utf-8"))
    if (
        tag_record["format_version"] != 1
        or tag_record["dataset"] != dataset
        or tag_record["tag"] != tag
    ):
        raise ValueError(f"Tag record does not match the requested tag: {tag_file_path}")
    assignment_history = tag_record["history"]
    if not isinstance(assignment_history, list) or not assignment_history:
        raise ValueError(f"Tag record must contain a nonempty history: {tag_file_path}")
    return assignment_history


def resolve_tag(
    *,
    dataset: str,
    tag: str,
    store: str | Path = "data/versioned",
) -> str:
    """Return the current permanent snapshot ID; a missing tag raises an error.

    This reads the reference only. Call retrieve_snapshot with the returned ID
    and dataset to check integrity and obtain the file before using its data.
    """
    assignment_history = list_tag_history(dataset=dataset, tag=tag, store=store)
    if not assignment_history:
        raise FileNotFoundError(f"Tag {tag!r} has not been assigned in dataset {dataset!r}")
    return assignment_history[-1]["snapshot_id"]


def list_tags(
    *,
    dataset: str,
    store: str | Path = "data/versioned",
) -> list[dict]:
    """List current targets and last-assignment details, sorted by tag name.

    An absent dataset or tags directory returns []. Like snapshot listing,
    this reads metadata without checking the preserved files' contents.
    """
    tags_directory = _get_dataset_directory(store, dataset) / "tags"
    tag_summaries = []
    for tag_file_path in sorted(tags_directory.glob("*.json")):
        tag = tag_file_path.stem
        assignment_history = list_tag_history(dataset=dataset, tag=tag, store=store)
        latest_assignment = assignment_history[-1]
        tag_summaries.append({
            "dataset": dataset,
            "tag": tag,
            "snapshot_id": latest_assignment["snapshot_id"],
            "changed_at": latest_assignment["changed_at"],
            "changed_by": latest_assignment["changed_by"],
            "note": latest_assignment["note"],
        })
    return tag_summaries


def assign_tag(
    *,
    dataset: str,
    tag: str,
    snapshot_id: str,
    changed_by: str,
    note: str | None = None,
    store: str | Path = "data/versioned",
) -> dict:
    """Assign or move a tag after verifying its target snapshot; return the event.

    changed_by is a caller-supplied person or process name, not authenticated
    identity. Every explicit assignment is recorded, even if the target is
    unchanged. This operation does not change data, run models, or establish
    data quality. Concurrent writers and crash durability remain unsupported.
    """
    tag_file_path = _get_tag_file_path(store, dataset, tag)
    if not isinstance(changed_by, str) or not changed_by.strip():
        raise ValueError("changed_by must be a nonempty person or process name")
    if note is not None and not isinstance(note, str):
        raise TypeError("note must be a string or None")

    # Check the same dataset's preserved bytes before creating or moving a tag.
    # Failed verification leaves any current tag and its history untouched.
    retrieve_snapshot(snapshot_id, dataset=dataset, store=store)
    assignment_history = list_tag_history(dataset=dataset, tag=tag, store=store)
    previous_snapshot_id = assignment_history[-1]["snapshot_id"] if assignment_history else None
    assignment_record = {
        "change_id": uuid4().hex,
        "previous_snapshot_id": previous_snapshot_id,
        "snapshot_id": snapshot_id,
        "changed_at": datetime.now(UTC).isoformat(),
        "changed_by": changed_by,
        "note": note,
    }
    assignment_history.append(assignment_record)
    tag_record = {
        "format_version": 1,
        "dataset": dataset,
        "tag": tag,
        "history": assignment_history,
    }

    tags_directory = tag_file_path.parent
    tags_directory.mkdir(parents=True, exist_ok=True)
    # Write the complete updated history beside the final file. Replacing it
    # publishes the new target and history in one filesystem operation.
    with TemporaryDirectory(prefix=".tag-", dir=tags_directory) as temporary_directory:
        staged_tag_file_path = Path(temporary_directory) / "tag.json"
        staged_tag_file_path.write_text(
            json.dumps(tag_record, indent=2) + "\n", encoding="utf-8",
        )
        staged_tag_file_path.replace(tag_file_path)

    return assignment_record
