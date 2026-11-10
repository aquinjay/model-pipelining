"""Check preserved versions, registration history, integrity, and portability."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openpyxl import Workbook

from model_pipelining.ingest import ingest_local_file, read_local_table
from model_pipelining.snapshots import (
    list_acquisitions,
    list_snapshots,
    register_snapshot,
    retrieve_snapshot,
)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        """Give every test its own input file and empty temporary store."""
        self.temporary_directory = TemporaryDirectory()
        # unittest runs this cleanup even when an assertion fails.
        self.addCleanup(self.temporary_directory.cleanup)
        self.test_directory = Path(self.temporary_directory.name)
        self.store_directory = self.test_directory / "store"
        self.source_file_path = self.test_directory / "observations.csv"
        self.original_file_bytes = b"timestamp,value\n2020-01-01,10\n2020-01-02,11\n"
        self.source_file_path.write_bytes(self.original_file_bytes)
        self.dataset_name = "synthetic-mean-shift"

    def register_source_file(self, source=None, dataset=None, note=None, ingestion_receipt=None):
        """Register a test input, using this test's defaults unless overridden."""
        return register_snapshot(
            source if source is not None else self.source_file_path,
            dataset=dataset if dataset is not None else self.dataset_name,
            note=note,
            ingestion_receipt=ingestion_receipt,
            store=self.store_directory,
        )

    def retrieve_preserved_file(self, snapshot_id, store=None):
        """Retrieve a test snapshot, optionally after its store has moved."""
        return retrieve_snapshot(
            snapshot_id, dataset=self.dataset_name,
            store=store if store is not None else self.store_directory,
        )

    def read_saved_acquisition_records(self):
        """Read files directly so assertions can inspect what registration saved."""
        return [
            json.loads(path.read_text())
            for path in (self.store_directory / self.dataset_name / "acquisitions").glob("*.json")
        ]

    def test_registration_preserves_bytes_manifest_and_source_context(self):
        snapshot = self.register_source_file()
        expected_id = hashlib.sha256(self.original_file_bytes).hexdigest()
        self.assertEqual(snapshot.snapshot_id, expected_id)
        self.assertEqual(snapshot.dataset, self.dataset_name)
        self.assertEqual(snapshot.path.read_bytes(), self.original_file_bytes)
        self.assertEqual(self.source_file_path.read_bytes(), self.original_file_bytes)
        manifest = json.loads((snapshot.path.parent.parent / "manifest.json").read_text())
        self.assertEqual(manifest, {
            "format_version": 1, "filename": self.source_file_path.name,
            "sha256": expected_id, "size_bytes": len(self.original_file_bytes),
        })
        event, = self.read_saved_acquisition_records()
        self.assertEqual(event["snapshot_id"], expected_id)
        self.assertEqual(event["dataset"], self.dataset_name)
        self.assertEqual(event["source_path"], str(self.source_file_path.resolve()))
        self.assertEqual(datetime.fromisoformat(event["registered_at"]).utcoffset().total_seconds(), 0)
        # Registration's internal staging copy is not an original ingestion.
        self.assertIsNone(event["ingestion_receipt"])

    def test_ingestion_receipt_survives_registration_source_removal_and_store_move(self):
        receipt = ingest_local_file(
            self.source_file_path, self.test_directory / "ingested" / "copy.csv",
        )
        self.source_file_path.unlink()
        before_registration = datetime.now(UTC)
        snapshot = self.register_source_file(receipt.destination, ingestion_receipt=receipt)
        after_registration = datetime.now(UTC)
        event, = self.read_saved_acquisition_records()
        self.assertEqual(event["source_path"], str(receipt.destination))
        self.assertEqual(event["ingestion_receipt"], {
            "source": str(receipt.source),
            "destination": str(receipt.destination),
            "acquired_at": receipt.acquired_at.isoformat(),
            "size_bytes": len(self.original_file_bytes),
        })
        registered_at = datetime.fromisoformat(event["registered_at"])
        self.assertLessEqual(before_registration, registered_at)
        self.assertLessEqual(registered_at, after_registration)
        self.assertLessEqual(receipt.acquired_at, before_registration)

        # Reading historical context does not require either original file.
        receipt.destination.unlink()
        moved_store = self.test_directory / "moved-store"
        shutil.move(self.store_directory, moved_store)
        listed, = list_acquisitions(dataset=self.dataset_name, store=moved_store)
        self.assertEqual(listed, event)
        self.assertEqual(
            self.retrieve_preserved_file(snapshot.snapshot_id, store=moved_store).read_bytes(),
            self.original_file_bytes,
        )

    def test_different_ingestion_receipts_reuse_snapshot_and_keep_separate_context(self):
        first = self.register_source_file()
        manifest_path = first.path.parent.parent / "manifest.json"
        manifest_before = manifest_path.read_bytes()
        expected_receipts = []
        for day in (1, 2):
            original = self.test_directory / f"original-{day}.csv"
            original.write_bytes(self.original_file_bytes)
            receipt = ingest_local_file(original, self.test_directory / f"copy-{day}.csv")
            receipt = replace(receipt, acquired_at=datetime(2026, 11, day, tzinfo=UTC))
            snapshot = self.register_source_file(receipt.destination, ingestion_receipt=receipt)
            self.assertEqual(snapshot, first)
            expected_receipts.append({
                "source": str(receipt.source), "destination": str(receipt.destination),
                "acquired_at": receipt.acquired_at.isoformat(), "size_bytes": receipt.size_bytes,
            })
        records = list_acquisitions(dataset=self.dataset_name, store=self.store_directory)
        self.assertCountEqual(
            [record["ingestion_receipt"] for record in records], [None, *expected_receipts],
        )
        self.assertEqual(len({record["acquisition_id"] for record in records}), 3)
        self.assertEqual(manifest_path.read_bytes(), manifest_before)

    def test_unrelated_ingestion_receipt_is_rejected_even_for_identical_bytes(self):
        receipt = ingest_local_file(self.source_file_path, self.test_directory / "copy.csv")
        with self.assertRaisesRegex(ValueError, "destination must match"):
            self.register_source_file(ingestion_receipt=receipt)
        self.assertFalse(self.store_directory.exists())

    def test_changed_ingested_file_size_is_rejected_before_creating_store(self):
        receipt = ingest_local_file(self.source_file_path, self.test_directory / "copy.csv")
        receipt.destination.write_bytes(b"changed size")
        with self.assertRaisesRegex(ValueError, "size_bytes must match"):
            self.register_source_file(receipt.destination, ingestion_receipt=receipt)
        self.assertFalse(self.store_directory.exists())

    def test_invalid_ingestion_receipt_type_is_rejected_before_creating_store(self):
        for receipt in ("receipt", {"source": "original.csv"}, True):
            with self.subTest(receipt=receipt):
                with self.assertRaisesRegex(TypeError, "ingestion_receipt must be"):
                    self.register_source_file(ingestion_receipt=receipt)
                self.assertFalse(self.store_directory.exists())

    def test_receipt_with_naive_acquisition_time_is_rejected_before_creating_store(self):
        receipt = ingest_local_file(self.source_file_path, self.test_directory / "copy.csv")
        receipt = replace(receipt, acquired_at=datetime(2026, 11, 1))
        with self.assertRaisesRegex(ValueError, "acquired_at must be timezone-aware"):
            self.register_source_file(receipt.destination, ingestion_receipt=receipt)
        self.assertFalse(self.store_directory.exists())

    def test_receipt_acquisition_time_is_saved_in_utc(self):
        receipt = ingest_local_file(self.source_file_path, self.test_directory / "copy.csv")
        local_time = datetime(2026, 11, 1, 10, tzinfo=timezone(timedelta(hours=-8)))
        receipt = replace(receipt, acquired_at=local_time)
        self.register_source_file(receipt.destination, ingestion_receipt=receipt)
        event, = self.read_saved_acquisition_records()
        self.assertEqual(event["ingestion_receipt"]["acquired_at"], "2026-11-01T18:00:00+00:00")

    def test_identical_or_renamed_input_reuses_snapshot_with_separate_records(self):
        first = self.register_source_file()
        renamed = self.test_directory / "renamed.csv"
        renamed.write_bytes(self.original_file_bytes)
        second = self.register_source_file(renamed)
        third = self.register_source_file()
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(len(list((self.store_directory / self.dataset_name / "snapshots").iterdir())), 1)
        events = self.read_saved_acquisition_records()
        self.assertEqual(len(events), 3)
        self.assertEqual(len({event["acquisition_id"] for event in events}), 3)
        self.assertEqual({event["source_path"] for event in events}, {
            str(self.source_file_path), str(renamed),
        })

    def test_old_observation_revision_keeps_both_versions_retrievable(self):
        first = self.register_source_file()
        revised = self.original_file_bytes.replace(b"2020-01-01,10", b"2020-01-01,99")
        self.source_file_path.write_bytes(revised)
        second = self.register_source_file()
        self.source_file_path.unlink()
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(
            self.retrieve_preserved_file(first.snapshot_id).read_bytes(),
            self.original_file_bytes,
        )
        self.assertEqual(self.retrieve_preserved_file(second.snapshot_id).read_bytes(), revised)

    def test_identical_data_in_different_datasets_have_separate_storage_and_records(self):
        first = self.register_source_file()
        other = self.register_source_file(dataset="another-dataset")
        self.assertEqual(first.snapshot_id, other.snapshot_id)
        self.assertNotEqual(first.path, other.path)
        self.assertEqual(len(self.read_saved_acquisition_records()), 1)
        other_acquisitions_directory = self.store_directory / "another-dataset" / "acquisitions"
        self.assertEqual(len(list(other_acquisitions_directory.glob("*.json"))), 1)

    def test_tampering_blocks_retrieval_and_reuse_without_recording_success(self):
        snapshot = self.register_source_file()
        # Same-size alteration ensures the checksum must detect the change.
        snapshot.path.write_bytes(self.original_file_bytes.replace(b",10", b",90"))
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            self.retrieve_preserved_file(snapshot.snapshot_id)
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            self.register_source_file()
        self.assertEqual(len(self.read_saved_acquisition_records()), 1)

    def test_manifest_identity_or_filename_changes_are_rejected(self):
        snapshot = self.register_source_file()
        manifest_path = snapshot.path.parent.parent / "manifest.json"
        original_manifest = json.loads(manifest_path.read_text())
        for field, value in (("sha256", "0" * 64), ("filename", "../outside.csv")):
            with self.subTest(field=field):
                manifest_path.write_text(json.dumps(original_manifest | {field: value}))
                with self.assertRaises(ValueError):
                    self.retrieve_preserved_file(snapshot.snapshot_id)

    def test_failed_copy_publishes_no_snapshot_or_acquisition(self):
        def failed_copy(source, destination):
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"partial")
            raise OSError("simulated copy failure")

        with patch("model_pipelining.snapshots.ingest_local_file", failed_copy):
            with self.assertRaisesRegex(OSError, "simulated copy failure"):
                self.register_source_file()
        self.assertEqual(list((self.store_directory / self.dataset_name).iterdir()), [])

    def test_store_can_move_without_changing_identity(self):
        snapshot = self.register_source_file()
        moved_store_directory = self.test_directory / "moved-store"
        shutil.move(self.store_directory, moved_store_directory)
        retrieved_file_path = self.retrieve_preserved_file(
            snapshot.snapshot_id, store=moved_store_directory,
        )
        self.assertEqual(retrieved_file_path.read_bytes(), self.original_file_bytes)
        self.assertTrue((moved_store_directory / self.dataset_name / "acquisitions").is_dir())

    def test_missing_or_directory_source_creates_no_store(self):
        invalid_sources = (
            (self.test_directory / "missing.csv", FileNotFoundError),
            (self.test_directory, ValueError),
        )
        for source, error in invalid_sources:
            with self.subTest(source=source):
                with self.assertRaises(error):
                    self.register_source_file(source)
                self.assertFalse(self.store_directory.exists())

    def test_invalid_dataset_or_snapshot_reference_is_rejected(self):
        for dataset in ("", "../outside", "a/b", ".", "two words"):
            with self.subTest(dataset=dataset):
                with self.assertRaisesRegex(ValueError, "dataset must"):
                    self.register_source_file(dataset=dataset)
        for snapshot_id in ("", "../outside", "f" * 63, "G" * 64):
            with self.subTest(snapshot_id=snapshot_id):
                with self.assertRaisesRegex(ValueError, "snapshot_id must"):
                    self.retrieve_preserved_file(snapshot_id)
        self.assertFalse(self.store_directory.exists())

    def test_missing_snapshot_raises_filesystem_error(self):
        with self.assertRaises(FileNotFoundError):
            self.retrieve_preserved_file("0" * 64)

    def test_preserved_xlsx_keeps_worksheets_readable(self):
        source = self.test_directory / "sales.xlsx"
        workbook = Workbook()
        workbook.active.title = "Sales"
        workbook.active.append(("value",))
        workbook.active.append((10,))
        forecast = workbook.create_sheet("Forecast")
        forecast.append(("value",))
        forecast.append((12,))
        workbook.save(source)
        workbook.close()
        snapshot = self.register_source_file(source)
        source.unlink()
        table = read_local_table(self.retrieve_preserved_file(snapshot.snapshot_id), sheet="Forecast")
        self.assertEqual(table["value"].tolist(), [12])

    def test_empty_lists_do_not_create_a_store(self):
        self.assertEqual(list_snapshots(dataset=self.dataset_name, store=self.store_directory), [])
        self.assertEqual(list_acquisitions(dataset=self.dataset_name, store=self.store_directory), [])
        self.assertFalse(self.store_directory.exists())

    def test_snapshot_listing_summarizes_unique_versions_within_one_dataset(self):
        first = self.register_source_file()
        self.register_source_file()
        self.source_file_path.write_bytes(b"timestamp,value\n2020-01-01,99\n")
        second = self.register_source_file()
        self.register_source_file(dataset="another-dataset")
        records = list_snapshots(dataset=self.dataset_name, store=self.store_directory)
        self.assertEqual([record["snapshot_id"] for record in records], sorted([
            first.snapshot_id, second.snapshot_id,
        ]))
        self.assertEqual({record["snapshot_id"]: record for record in records}, {
            snapshot.snapshot_id: {
                "dataset": self.dataset_name, "snapshot_id": snapshot.snapshot_id,
                "filename": self.source_file_path.name, "size_bytes": snapshot.path.stat().st_size,
            }
            for snapshot in (first, second)
        })

    def test_acquisition_listing_preserves_repeat_events_and_sorts_by_time_then_id(self):
        # Deliberately register out of timestamp order, including a tied time.
        registration_times = [datetime(2026, 11, day, tzinfo=UTC) for day in (3, 1, 1)]
        # Replace the clock temporarily so this test never depends on timing.
        with patch("model_pipelining.snapshots.datetime") as simulated_clock:
            simulated_clock.now.side_effect = registration_times
            self.register_source_file(note="Later")
            self.register_source_file(note="Earlier")
            self.register_source_file(note="Also earlier")
        self.register_source_file(dataset="another-dataset", note="Separate history")
        records = list_acquisitions(dataset=self.dataset_name, store=self.store_directory)
        self.assertEqual(len(records), 3)
        self.assertEqual(len({record["snapshot_id"] for record in records}), 1)
        self.assertEqual([record["note"] for record in records][-1], "Later")
        self.assertEqual([record["acquisition_id"] for record in records[:2]], sorted(
            record["acquisition_id"] for record in records[:2]
        ))
        self.assertEqual({record["note"] for record in records}, {
            "Later", "Earlier", "Also earlier",
        })

    def test_notes_belong_to_registrations_without_changing_the_snapshot(self):
        first = self.register_source_file()
        manifest_path = first.path.parent.parent / "manifest.json"
        manifest_before = manifest_path.read_bytes()
        note = "Corrected source description\nReviewed by José"
        second = self.register_source_file(note=note)
        third = self.register_source_file(note="")
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertEqual(first.path.read_bytes(), self.original_file_bytes)
        records = list_acquisitions(dataset=self.dataset_name, store=self.store_directory)
        self.assertCountEqual([record["note"] for record in records], [None, note, ""])

    def test_older_acquisitions_without_optional_fields_are_read_without_rewriting_them(self):
        self.register_source_file()
        record_path, = (self.store_directory / self.dataset_name / "acquisitions").glob("*.json")
        record = json.loads(record_path.read_text())
        del record["note"]
        del record["ingestion_receipt"]
        record_path.write_text(json.dumps(record))
        before = record_path.read_bytes()
        listed, = list_acquisitions(dataset=self.dataset_name, store=self.store_directory)
        self.assertIsNone(listed["note"])
        self.assertIsNone(listed["ingestion_receipt"])
        self.assertEqual(record_path.read_bytes(), before)

    def test_invalid_note_is_rejected_before_creating_the_store(self):
        for note in (123, True, ["a note"], {"reason": "a note"}):
            with self.subTest(note=note):
                with self.assertRaisesRegex(TypeError, "note must be"):
                    self.register_source_file(note=note)
                self.assertFalse(self.store_directory.exists())

    def test_snapshot_listing_can_inspect_metadata_without_available_contents(self):
        snapshot = self.register_source_file()
        snapshot.path.unlink()
        listed, = list_snapshots(dataset=self.dataset_name, store=self.store_directory)
        self.assertEqual(listed["snapshot_id"], snapshot.snapshot_id)
        self.assertEqual(listed["size_bytes"], len(self.original_file_bytes))
        with self.assertRaises(FileNotFoundError):
            self.retrieve_preserved_file(snapshot.snapshot_id)

    def test_listing_surfaces_malformed_records(self):
        snapshot = self.register_source_file()
        manifest_path = snapshot.path.parent.parent / "manifest.json"
        manifest_path.write_text("broken JSON")
        with self.assertRaises(json.JSONDecodeError):
            list_snapshots(dataset=self.dataset_name, store=self.store_directory)
        record_path, = (self.store_directory / self.dataset_name / "acquisitions").glob("*.json")
        record_path.write_text("broken JSON")
        with self.assertRaises(json.JSONDecodeError):
            list_acquisitions(dataset=self.dataset_name, store=self.store_directory)


if __name__ == "__main__":
    unittest.main()
