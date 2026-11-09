"""Behavioral checks for local acquisition and CSV/XLSX table reading."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openpyxl import Workbook
from pandas.testing import assert_frame_equal

from model_pipelining.ingest import ingest_local_file, read_local_table


class LocalIngestTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.csv"
        self.contents = b"timestamp,value\n2020-01-01,10\n2020-01-02,11\n"
        self.source.write_bytes(self.contents)
        self.destination = self.root / "acquired" / "observations.csv"

    def test_copy_preserves_bytes_and_returns_receipt(self):
        before = datetime.now(UTC)
        result = ingest_local_file(str(self.source), str(self.destination))
        after = datetime.now(UTC)

        self.assertEqual(self.destination.read_bytes(), self.contents)
        self.assertEqual(self.source.read_bytes(), self.contents)
        self.assertEqual(result.source, self.source.resolve())
        self.assertEqual(result.destination, self.destination.resolve())
        self.assertEqual(result.size_bytes, len(self.contents))
        self.assertEqual(result.acquired_at.tzinfo, UTC)
        self.assertLessEqual(before, result.acquired_at)
        self.assertLessEqual(result.acquired_at, after)

    def test_missing_or_directory_source_produces_no_copy(self):
        for source, error in (
            (self.root / "missing.csv", FileNotFoundError),
            (self.root, ValueError),
        ):
            with self.subTest(source=source):
                with self.assertRaises(error):
                    ingest_local_file(source, self.destination)
                self.assertFalse(self.destination.exists())

    def test_existing_destination_is_preserved(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"existing data")
        with self.assertRaises(FileExistsError):
            ingest_local_file(self.source, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"existing data")

    def test_same_file_is_preserved(self):
        with self.assertRaises(FileExistsError):
            ingest_local_file(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), self.contents)

    def test_destination_symlink_is_not_followed(self):
        self.destination.parent.mkdir()
        for target in (self.source, self.root / "missing.csv"):
            with self.subTest(target=target):
                self.destination.symlink_to(target)
                try:
                    with self.assertRaises(FileExistsError):
                        ingest_local_file(self.source, self.destination)
                    self.assertTrue(self.destination.is_symlink())
                    self.assertEqual(self.source.read_bytes(), self.contents)
                    if target != self.source:
                        self.assertFalse(target.exists())
                finally:
                    self.destination.unlink()

    def test_source_symlink_copies_the_regular_file(self):
        link = self.root / "source-link.csv"
        link.symlink_to(self.source)
        result = ingest_local_file(link, self.destination)
        self.assertEqual(result.source, self.source.resolve())
        self.assertEqual(self.destination.read_bytes(), self.contents)

    def test_failed_copy_removes_its_partial_output(self):
        def interrupted_copy(source_file, destination_file):
            destination_file.write(source_file.read(5))
            raise OSError("simulated interrupted copy")

        with patch("model_pipelining.ingest.shutil.copyfileobj", interrupted_copy):
            with self.assertRaisesRegex(OSError, "simulated interrupted copy"):
                ingest_local_file(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.source.read_bytes(), self.contents)

    def test_copy_can_be_read_after_source_is_removed(self):
        result = ingest_local_file(self.source, self.destination)
        self.source.unlink()
        table = read_local_table(result.destination)
        self.assertEqual(table["value"].tolist(), [10, 11])


class LocalTableTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "observations.CSV"
        self.csv.write_text("day,value\nMonday,10\nTuesday,11\n", encoding="utf-8")
        self.xlsx = self.root / "observations.XLSX"

        workbook = Workbook()
        first = workbook.active
        first.title = "Sales"
        for row in (("day", "value"), ("Monday", 10), ("Tuesday", 11)):
            first.append(row)
        second = workbook.create_sheet("Forecast")
        second.append(("day", "value"))
        second.append(("Wednesday", 12))
        intro = workbook.create_sheet("With intro")
        intro.append(("Example report",))
        intro.append(("day", "value"))
        intro.append(("Thursday", 13))
        workbook.save(self.xlsx)
        workbook.close()

    def test_csv_and_default_xlsx_sheet_have_matching_tables(self):
        csv_table = read_local_table(str(self.csv))
        xlsx_table = read_local_table(self.xlsx)
        assert_frame_equal(csv_table, xlsx_table)
        self.assertEqual(csv_table.to_dict("list"), {
            "day": ["Monday", "Tuesday"], "value": [10, 11],
        })

    def test_sheet_selection_by_name_and_index(self):
        named = read_local_table(self.xlsx, sheet="Forecast")
        indexed = read_local_table(self.xlsx, sheet=1)
        assert_frame_equal(named, indexed)
        self.assertEqual(named.to_dict("list"), {"day": ["Wednesday"], "value": [12]})

    def test_header_row_can_follow_an_introduction(self):
        csv = self.root / "intro.csv"
        csv.write_text("Example report\nday,value\nThursday,13\n", encoding="utf-8")
        csv_table = read_local_table(csv, header=1)
        xlsx_table = read_local_table(self.xlsx, sheet="With intro", header=1)
        assert_frame_equal(csv_table, xlsx_table)
        self.assertEqual(csv_table.to_dict("list"), {"day": ["Thursday"], "value": [13]})

    def test_header_none_keeps_the_first_row_as_data(self):
        for path in (self.csv, self.xlsx):
            with self.subTest(path=path):
                table = read_local_table(path, header=None)
                self.assertEqual(table.columns.tolist(), [0, 1])
                self.assertEqual(table.iloc[0].tolist(), ["day", "value"])
                self.assertEqual(len(table), 3)

    def test_missing_sheet_does_not_change_the_acquired_workbook(self):
        destination = self.root / "acquired" / "copy.xlsx"
        result = ingest_local_file(self.xlsx, destination)
        original_bytes = self.xlsx.read_bytes()
        for sheet in ("Missing", 3):
            with self.subTest(sheet=sheet):
                with self.assertRaises(ValueError):
                    read_local_table(result.destination, sheet=sheet)
                self.assertEqual(result.destination.read_bytes(), original_bytes)

    def test_csv_rejects_sheet_arguments(self):
        for sheet in (0, "Sales"):
            with self.subTest(sheet=sheet):
                with self.assertRaisesRegex(ValueError, "only supported for XLSX"):
                    read_local_table(self.csv, sheet=sheet)

    def test_invalid_sheet_and_header_options_fail_clearly(self):
        for sheet in (-1, True, 1.5, ["Sales"]):
            with self.subTest(sheet=sheet):
                with self.assertRaisesRegex(ValueError, "sheet must be"):
                    read_local_table(self.xlsx, sheet=sheet)
        for header in (-1, True, 1.5, [0, 1]):
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, "header must be"):
                    read_local_table(self.csv, header=header)

    def test_unsupported_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported table format"):
            read_local_table(self.root / "observations.json")

    def test_missing_table_raises_a_filesystem_error(self):
        with self.assertRaises(FileNotFoundError):
            read_local_table(self.root / "missing.csv")


if __name__ == "__main__":
    unittest.main()
