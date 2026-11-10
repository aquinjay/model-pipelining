"""Copy local input files and read CSV or XLSX tables.

Copying and parsing are separate: a workbook remains available even if reading
one of its worksheets fails. Neither operation registers a versioned snapshot.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil

import pandas


@dataclass(frozen=True)
class IngestResult:
    """Receipt for a completed copy.

    source and destination are absolute file paths. acquired_at records when
    this copy finished in UTC; it does not establish historical availability.
    size_bytes is the byte count. frozen=True prevents field reassignment.
    """

    source: Path
    destination: Path
    acquired_at: datetime
    size_bytes: int


def ingest_local_file(source: str | Path, destination: str | Path) -> IngestResult:
    """Copy one completed local file without overwriting an existing destination.

    source and destination accept strings or Path objects. Return an IngestResult
    receipt describing the completed copy. Filesystem failures propagate to the
    caller, including a missing source or an already occupied destination.

    Parent directories are created as needed. The source must remain unchanged
    during copying. A handled copy failure removes the partial destination;
    this function does not provide crash recovery or integrity verification.
    """
    # Expand ~ and resolve an absolute source path. strict=True requires it to
    # exist; the next check excludes directories and other non-regular files.
    source_file_path = Path(source).expanduser().resolve(strict=True)
    if not source_file_path.is_file():
        raise ValueError(f"Source must be a regular file: {source_file_path}")

    destination_file_path = Path(destination).expanduser()
    # Binary mode preserves bytes for any file format. The with block closes
    # the source automatically, even if copying raises an error.
    with source_file_path.open("rb") as source_file:
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Resolve the parent only: resolving the final name would follow an
        # existing symlink instead of rejecting it as an occupied destination.
        destination_file_path = destination_file_path.parent.resolve() / destination_file_path.name
        # x creates a new file exclusively; b writes bytes. Existing files and
        # symlinks fail here, before entering the cleanup block below.
        destination_file = destination_file_path.open("xb")
        try:
            with destination_file:
                shutil.copyfileobj(source_file, destination_file)
                # Copying starts at byte zero, so the final position is the size.
                size_bytes = destination_file.tell()
        except BaseException:
            # Catch errors and interruptions to remove only our partial copy.
            # The bare raise reports the original failure after cleanup.
            destination_file_path.unlink(missing_ok=True)
            raise

    # Both files are closed before we report success and record the UTC time.
    return IngestResult(
        source=source_file_path,
        destination=destination_file_path,
        acquired_at=datetime.now(UTC),
        size_bytes=size_bytes,
    )


def read_local_table(
    path: str | Path,
    *,
    sheet: str | int | None = None,
    header: int | None = 0,
) -> pandas.DataFrame:
    """Read a CSV or one XLSX worksheet, using pandas' ordinary type inference.

    XLSX sheets are selected by name or zero-based index; omission selects the
    first worksheet. CSV files do not accept a sheet. ``header`` identifies the
    zero-based header row, or is None for data with no header. Excel formulas
    are not evaluated; their values depend on the workbook's cached results.
    The returned DataFrame is a table in memory; the input file is unchanged.
    """
    # Reject booleans as row indexes, even though bool is a subclass of int.
    if header is not None and (type(header) is not int or header < 0):
        raise ValueError("header must be a nonnegative integer or None")

    table_file_path = Path(path).expanduser()
    # Choose a parser from the extension, accepting .CSV and .XLSX as well.
    file_extension = table_file_path.suffix.lower()
    match file_extension:
        case ".csv":
            if sheet is not None:
                raise ValueError("sheet is only supported for XLSX files")
            return pandas.read_csv(table_file_path, header=header)
        case ".xlsx":
            if sheet is not None and not (
                isinstance(sheet, str) or (type(sheet) is int and sheet >= 0)
            ):
                raise ValueError("sheet must be a worksheet name or nonnegative integer")
            # Select exactly one sheet. Index 0 means the first worksheet.
            selected_sheet = 0 if sheet is None else sheet
            return pandas.read_excel(
                table_file_path,
                sheet_name=selected_sheet,
                header=header,
                engine="openpyxl",
            )
        case _:
            raise ValueError("Unsupported table format; expected .csv or .xlsx")
