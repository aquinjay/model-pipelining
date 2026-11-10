"""Check explicit tag assignments, dataset boundaries, history, and failure safety."""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model_pipelining.snapshots import register_snapshot, retrieve_snapshot
from model_pipelining.tags import assign_tag, list_tag_history, list_tags, resolve_tag


class TagTests(unittest.TestCase):
    def setUp(self):
        """Keep all files and tag changes inside a fresh store for each test."""
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.test_directory = Path(self.temporary_directory.name)
        self.store_directory = self.test_directory / "store"
        self.source_file_path = self.test_directory / "observations.csv"
        self.source_file_path.write_bytes(b"value\n10\n")
        self.dataset_name = "synthetic"
        self.original_snapshot = register_snapshot(
            self.source_file_path, dataset=self.dataset_name, store=self.store_directory,
        )

    def assign_input_tag(self, snapshot_id=None, **options):
        """Use defaults for this test, allowing individual arguments to be varied."""
        arguments = {
            "dataset": self.dataset_name,
            "tag": "baseline-input",
            "snapshot_id": snapshot_id if snapshot_id is not None else self.original_snapshot.snapshot_id,
            "changed_by": "test-user",
            "store": self.store_directory,
        }
        return assign_tag(**(arguments | options))

    def read_input_history(self):
        return list_tag_history(
            dataset=self.dataset_name, tag="baseline-input", store=self.store_directory,
        )

    def resolve_input_tag(self, store=None):
        return resolve_tag(
            dataset=self.dataset_name, tag="baseline-input",
            store=store if store is not None else self.store_directory,
        )

    def register_revised_snapshot(self):
        self.source_file_path.write_bytes(b"value\n99\n")
        return register_snapshot(
            self.source_file_path, dataset=self.dataset_name, store=self.store_directory,
        )

    def test_initial_assignment_records_context_without_changing_the_snapshot(self):
        manifest_file_path = self.original_snapshot.path.parent.parent / "manifest.json"
        original_manifest = manifest_file_path.read_bytes()
        assignment = self.assign_input_tag(note="Selected for review\nJosé approved")
        self.assertEqual(self.resolve_input_tag(), self.original_snapshot.snapshot_id)
        self.assertEqual(self.read_input_history(), [assignment])
        self.assertIsNone(assignment["previous_snapshot_id"])
        self.assertEqual(assignment["changed_by"], "test-user")
        self.assertEqual(assignment["note"], "Selected for review\nJosé approved")
        self.assertEqual(datetime.fromisoformat(assignment["changed_at"]).utcoffset().total_seconds(), 0)
        self.assertEqual(manifest_file_path.read_bytes(), original_manifest)
        self.assertEqual(self.original_snapshot.path.read_bytes(), b"value\n10\n")

    def test_moving_tag_preserves_history_and_previously_resolved_input(self):
        self.assign_input_tag()
        pinned_snapshot_id = self.resolve_input_tag()
        revised_snapshot = self.register_revised_snapshot()
        assignment = self.assign_input_tag(
            revised_snapshot.snapshot_id, changed_by="refresh-process", note="Use corrected input",
        )
        self.assertEqual(self.resolve_input_tag(), revised_snapshot.snapshot_id)
        self.assertEqual(assignment["previous_snapshot_id"], pinned_snapshot_id)
        self.assertEqual(len(self.read_input_history()), 2)
        pinned_file_path = retrieve_snapshot(
            pinned_snapshot_id, dataset=self.dataset_name, store=self.store_directory,
        )
        self.assertEqual(pinned_file_path.read_bytes(), b"value\n10\n")

    def test_registration_does_not_create_or_advance_tags(self):
        self.assertEqual(list_tags(dataset=self.dataset_name, store=self.store_directory), [])
        self.assign_input_tag()
        self.register_revised_snapshot()
        self.assertEqual(self.resolve_input_tag(), self.original_snapshot.snapshot_id)
        self.assertEqual(len(self.read_input_history()), 1)

    def test_reassigning_same_snapshot_is_an_explicit_event(self):
        first_assignment = self.assign_input_tag()
        second_assignment = self.assign_input_tag(note="Reconfirmed")
        self.assertEqual(second_assignment["previous_snapshot_id"], self.original_snapshot.snapshot_id)
        self.assertNotEqual(first_assignment["change_id"], second_assignment["change_id"])
        self.assertEqual(self.read_input_history(), [first_assignment, second_assignment])

    def test_tags_are_dataset_scoped_and_multiple_tags_can_share_a_target(self):
        self.assign_input_tag()
        self.assign_input_tag(tag="another-input")
        other_snapshot = register_snapshot(
            self.source_file_path, dataset="other-dataset", store=self.store_directory,
        )
        self.assign_input_tag(other_snapshot.snapshot_id, dataset="other-dataset")
        tags = list_tags(dataset=self.dataset_name, store=self.store_directory)
        self.assertEqual([record["tag"] for record in tags], ["another-input", "baseline-input"])
        self.assertEqual({record["snapshot_id"] for record in tags}, {self.original_snapshot.snapshot_id})
        self.assertEqual(len(self.read_input_history()), 1)
        other_tags = list_tags(dataset="other-dataset", store=self.store_directory)
        self.assertEqual(len(other_tags), 1)
        self.assertEqual(other_tags[0]["dataset"], "other-dataset")

    def test_missing_tags_are_read_without_creating_directories(self):
        self.assertEqual(self.read_input_history(), [])
        self.assertEqual(list_tags(dataset="missing", store=self.store_directory), [])
        with self.assertRaises(FileNotFoundError):
            self.resolve_input_tag()
        self.assertFalse((self.store_directory / self.dataset_name / "tags").exists())
        self.assertFalse((self.store_directory / "missing").exists())

    def test_invalid_arguments_do_not_create_tag_files(self):
        invalid_arguments = [
            {"tag": ""}, {"tag": "../outside"}, {"tag": "two words"},
            {"dataset": "../outside"}, {"snapshot_id": "bad-id"},
            {"changed_by": ""}, {"changed_by": "   "}, {"changed_by": None},
            {"note": ["not a string"]},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((ValueError, TypeError)):
                    self.assign_input_tag(**arguments)
        self.assertFalse((self.store_directory / self.dataset_name / "tags").exists())

    def test_missing_or_wrong_dataset_target_cannot_move_a_tag(self):
        self.assign_input_tag()
        original_history = self.read_input_history()
        with self.assertRaises(FileNotFoundError):
            self.assign_input_tag("0" * 64)
        with self.assertRaises(FileNotFoundError):
            self.assign_input_tag(dataset="another-dataset")
        self.assertEqual(self.read_input_history(), original_history)

    def test_corrupted_target_cannot_move_a_tag(self):
        self.assign_input_tag()
        revised_snapshot = self.register_revised_snapshot()
        revised_snapshot.path.write_bytes(b"value\n88\n")
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            self.assign_input_tag(revised_snapshot.snapshot_id)
        self.assertEqual(self.resolve_input_tag(), self.original_snapshot.snapshot_id)
        self.assertEqual(len(self.read_input_history()), 1)

    def test_failed_publication_preserves_current_target_and_history(self):
        self.assign_input_tag()
        revised_snapshot = self.register_revised_snapshot()
        original_history = self.read_input_history()
        # Fail the final publication, after the updated JSON has been written.
        with patch("model_pipelining.tags.Path.replace", side_effect=OSError("simulated publication failure")):
            with self.assertRaisesRegex(OSError, "simulated publication failure"):
                self.assign_input_tag(revised_snapshot.snapshot_id)
        self.assertEqual(self.resolve_input_tag(), self.original_snapshot.snapshot_id)
        self.assertEqual(self.read_input_history(), original_history)
        tags_directory = self.store_directory / self.dataset_name / "tags"
        self.assertEqual([path.name for path in tags_directory.iterdir()], ["baseline-input.json"])

    def test_malformed_tag_records_are_not_silently_replaced(self):
        self.assign_input_tag()
        tag_file_path = self.store_directory / self.dataset_name / "tags" / "baseline-input.json"
        original_record = json.loads(tag_file_path.read_text())
        malformed_records = [
            "broken JSON",
            json.dumps(original_record | {"dataset": "wrong-dataset"}),
            json.dumps(original_record | {"history": []}),
        ]
        for malformed_record in malformed_records:
            with self.subTest(record=malformed_record):
                tag_file_path.write_text(malformed_record)
                with self.assertRaises(ValueError):
                    self.resolve_input_tag()
                with self.assertRaises(ValueError):
                    self.assign_input_tag()
                self.assertEqual(tag_file_path.read_text(), malformed_record)

    def test_whole_store_can_move_with_tags_and_history_intact(self):
        self.assign_input_tag(note="Initial selection")
        original_history = self.read_input_history()
        moved_store_directory = self.test_directory / "moved-store"
        shutil.move(self.store_directory, moved_store_directory)
        self.assertEqual(self.resolve_input_tag(store=moved_store_directory), self.original_snapshot.snapshot_id)
        moved_history = list_tag_history(
            dataset=self.dataset_name, tag="baseline-input", store=moved_store_directory,
        )
        self.assertEqual(moved_history, original_history)

    def test_history_keeps_assignment_order_when_clock_moves_backwards(self):
        assignment_times = [datetime(2026, 11, day, tzinfo=UTC) for day in (3, 1)]
        with patch("model_pipelining.tags.datetime") as simulated_clock:
            simulated_clock.now.side_effect = assignment_times
            first_assignment = self.assign_input_tag()
            second_assignment = self.assign_input_tag(note="Later action with earlier clock time")
        self.assertEqual(self.read_input_history(), [first_assignment, second_assignment])


if __name__ == "__main__":
    unittest.main()
