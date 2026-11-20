"""Shared writers for saved JSON and CSV artifacts.

Every workflow writes its manifests, checks and tables through these helpers,
so saved files use one encoding and layout. New files are created exclusively
unless the caller explicitly replaces a manifest it owns.
"""

import csv
import json
from pathlib import Path


def write_json(path: Path, value: object, *, overwrite: bool = False) -> None:
    """Write indented, finite JSON plus a newline; refuse an existing file unless overwrite=True."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(text)


def write_csv(path: Path, fields, rows) -> None:
    """Create a new CSV with the given header order; refuse an existing file."""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
