"""Copy local input files and read CSV or XLSX tables.

Copying and parsing are separate: a workbook remains available even if reading
one of its worksheets fails. Neither operation registers a versioned snapshot.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil

import pandas as pd


@dataclass(frozen=True)
class IngestResult:
    """Receipt for a completed copy; acquisition time is not data availability."""

    source: Path
    destination: Path
    acquired_at: datetime
    size_bytes: int


def ingest_local_file(source: str | Path, destination: str | Path) -> IngestResult:
    """Copy one completed local file without overwriting an existing destination.

    Parent directories are created as needed. The source must remain unchanged
    during copying. A handled copy failure removes the partial destination;
    this function does not provide crash recovery or integrity verification.
    """
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"Source must be a regular file: {source_path}")

    destination_path = Path(destination).expanduser()
    with source_path.open("rb") as source_file:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # Resolve the parent only: resolving the final name would follow an
        # existing symlink instead of rejecting it as an occupied destination.
        destination_path = destination_path.parent.resolve() / destination_path.name
        destination_file = destination_path.open("xb")
        try:
            with destination_file:
                shutil.copyfileobj(source_file, destination_file)
                size_bytes = destination_file.tell()
        except BaseException:
            # Exclusive creation succeeded, so this call owns the partial file.
            destination_path.unlink(missing_ok=True)
            raise

    return IngestResult(
        source=source_path,
        destination=destination_path,
        acquired_at=datetime.now(UTC),
        size_bytes=size_bytes,
    )


def read_local_table(
    path: str | Path,
    *,
    sheet: str | int | None = None,
    header: int | None = 0,
) -> pd.DataFrame:
    """Read a CSV or one XLSX worksheet, using pandas' ordinary type inference.

    XLSX sheets are selected by name or zero-based index; omission selects the
    first worksheet. CSV files do not accept a sheet. ``header`` identifies the
    zero-based header row, or is None for data with no header. Excel formulas
    are not evaluated; their values depend on the workbook's cached results.
    """
    if header is not None and (type(header) is not int or header < 0):
        raise ValueError("header must be a nonnegative integer or None")

    table_path = Path(path).expanduser()
    match table_path.suffix.lower():
        case ".csv":
            if sheet is not None:
                raise ValueError("sheet is only supported for XLSX files")
            return pd.read_csv(table_path, header=header)
        case ".xlsx":
            if sheet is not None and not (
                isinstance(sheet, str) or (type(sheet) is int and sheet >= 0)
            ):
                raise ValueError("sheet must be a worksheet name or nonnegative integer")
            return pd.read_excel(
                table_path,
                sheet_name=0 if sheet is None else sheet,
                header=header,
                engine="openpyxl",
            )
        case _:
            raise ValueError("Unsupported table format; expected .csv or .xlsx")
