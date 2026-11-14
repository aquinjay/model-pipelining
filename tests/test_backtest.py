"""Behavioral checks for historical selection, failures, and saved evidence."""

import csv
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model_pipelining.backtest import run_backtest
from model_pipelining.models import TrainingMean
from model_pipelining.reporting import plot_run
from model_pipelining.snapshots import register_snapshot
from model_pipelining.tags import assign_tag


class RecordingMean(TrainingMean):
    """Retain requests for inspection without changing the baseline's behavior."""

    def __init__(self):
        self.requests = []

    def forecast(self, request):
        self.requests.append(request)
        return super().forecast(request)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        self.counter = 0

    def snapshot(self, values=(1, 3, 5, 7, 9), *, timestamps=None, columns=("timestamp", "value")):
        self.counter += 1
        source = self.root / f"input-{self.counter}.csv"
        if timestamps is None:
            timestamps = [(datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat()
                          for index in range(len(values))]
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(zip(timestamps, values))
        return register_snapshot(source, dataset="example", store=self.store)

    def run_snapshot(self, snapshot, **options):
        output = options.pop("output_directory", self.root / f"run-{self.counter}")
        return run_backtest(dataset="example", snapshot_id=snapshot.snapshot_id,
                            store=self.store, minimum_history=2, output_directory=output, **options)

    def read_json(self, run, name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    def read_csv(self, run, name="predictions.csv"):
        with (run / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_known_predictions_alignment_and_scores(self):
        snapshot = self.snapshot()
        run = self.run_snapshot(snapshot)
        predictions = self.read_csv(run)
        self.assertEqual([float(row["prediction"]) for row in predictions], [2, 3, 4])
        self.assertEqual([int(row["training_count"]) for row in predictions], [2, 3, 4])
        self.assertEqual(predictions[0]["issued_at"], "2020-01-02T00:00:00+00:00")
        self.assertEqual(predictions[0]["target_time"], "2020-01-03T00:00:00+00:00")
        self.assertNotIn("actual", predictions[0])
        scores = self.read_json(run, "scores.json")
        self.assertEqual(scores["mae"], 4)
        self.assertAlmostEqual(scores["rmse"], math.sqrt(50 / 3))
        self.assertEqual(scores["scored_count"], 3)
        self.assertTrue(scores["complete"])
        self.assertTrue(all(check["status"] == "passed"
                            for check in self.read_json(run, "checks.json") if check["required"]))
        manifest = self.read_json(run, "manifest.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["input"]["snapshot_id"], snapshot.snapshot_id)
        self.assertEqual(manifest["task"]["evaluation_role"], "development")
        self.assertEqual(self.read_json(run, "review.json")["status"], "pending")

    def test_future_changes_cannot_change_an_earlier_prediction(self):
        first = self.run_snapshot(self.snapshot((1, 3, 5, 7, 9)))
        second = self.run_snapshot(self.snapshot((1, 3, 500, 700, 900)))
        self.assertEqual(self.read_csv(first)[0], self.read_csv(second)[0])
        self.assertNotEqual(self.read_csv(first)[1]["prediction"], self.read_csv(second)[1]["prediction"])

    def test_model_receives_only_immutable_eligible_history(self):
        model = RecordingMean()
        self.run_snapshot(self.snapshot(), model=model)
        self.assertEqual([len(request.history) for request in model.requests], [2, 3, 4])
        for request in model.requests:
            self.assertIsInstance(request.history, tuple)
            self.assertEqual(request.issued_at, request.history[-1].timestamp)
            self.assertTrue(all(observation.timestamp <= request.issued_at for observation in request.history))
            self.assertEqual(request.target_time - request.issued_at, timedelta(days=1))
            with self.assertRaises(FrozenInstanceError):
                request.history[0].value = 999

    def test_resolves_tag_once_and_reproduces_after_tag_moves(self):
        first = self.snapshot()
        other = self.snapshot((11, 13, 15, 17, 19))
        assign_tag(dataset="example", tag="baseline", snapshot_id=first.snapshot_id,
                   changed_by="test", store=self.store)

        class MovingTagMean(TrainingMean):
            def forecast(model, request):
                assign_tag(dataset="example", tag="baseline", snapshot_id=other.snapshot_id,
                           changed_by="test", store=self.store)
                return super().forecast(request)

        run = run_backtest(dataset="example", tag="baseline", minimum_history=2,
                           model=MovingTagMean(), store=self.store, output_directory=self.root / "tag-run")
        self.assertEqual(self.read_json(run, "manifest.json")["input"]["snapshot_id"], first.snapshot_id)
        replay = self.run_snapshot(first, output_directory=self.root / "replay")
        self.assertEqual((run / "predictions.csv").read_bytes(), (replay / "predictions.csv").read_bytes())
        self.assertEqual(self.read_json(run, "scores.json"), self.read_json(replay, "scores.json"))

    def test_replay_works_after_store_relocation_and_source_removal(self):
        snapshot = self.snapshot()
        run = self.run_snapshot(snapshot)
        manifest = self.read_json(run, "manifest.json")
        relocated = self.root / "relocated"
        shutil.move(self.store, relocated)
        for path in self.root.glob("input-*.csv"):
            path.unlink()
        replay = run_backtest(**{key: manifest["input"][key] for key in ("dataset", "snapshot_id")},
                              minimum_history=manifest["task"]["minimum_history"],
                              output_directory=self.root / "replay", store=relocated)
        for filename in ("predictions.csv", "evaluation.csv", "scores.json"):
            self.assertEqual((run / filename).read_bytes(), (replay / filename).read_bytes())

    def test_failed_fold_is_saved_and_other_folds_continue(self):
        class FailingMean(TrainingMean):
            def forecast(self, request):
                if len(request.history) == 3:
                    raise RuntimeError("example fitting failure")
                return super().forecast(request)

        run = self.run_snapshot(self.snapshot(), model=FailingMean())
        rows = self.read_csv(run)
        self.assertEqual([row["status"] for row in rows], ["passed", "failed", "passed"])
        self.assertIn("example fitting failure", rows[1]["error"])
        self.assertEqual(rows[1]["prediction"], "")
        scores = self.read_json(run, "scores.json")
        self.assertEqual((scores["expected_count"], scores["scored_count"], scores["failed_count"]), (3, 2, 1))
        self.assertFalse(scores["complete"])
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")

    def test_invalid_outputs_are_failed_evidence_instead_of_nan_scores(self):
        snapshot = self.snapshot()
        for index, value in enumerate((float("nan"), float("inf"), True, "4", [4], None)):
            with self.subTest(value=value):
                class InvalidModel(TrainingMean):
                    def forecast(self, request):
                        return value

                run = self.run_snapshot(snapshot, model=InvalidModel(), output_directory=self.root / f"invalid-{index}")
                scores = self.read_json(run, "scores.json")
                self.assertEqual(scores["scored_count"], 0)
                self.assertIsNone(scores["mae"])
                self.assertIsNone(scores["rmse"])
                self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
                self.assertIn("failed", [check["status"] for check in self.read_json(run, "checks.json")])

    def test_scoring_overflow_is_visible(self):
        class LargePoint(TrainingMean):
            def forecast(self, request):
                return 1e308

        run = self.run_snapshot(self.snapshot(), model=LargePoint())
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertEqual(self.read_json(run, "scores.json")["scored_count"], 0)
        self.assertTrue(all(row["status"] == "failed" for row in self.read_csv(run, "evaluation.csv")))

    def test_predictions_are_closed_before_scoring(self):
        from model_pipelining import backtest

        snapshot = self.snapshot()
        original_write = backtest._write_csv

        def inspect_write(path, rows):
            if path.name == "evaluation.csv":
                with (path.parent / "predictions.csv").open(encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
            return original_write(path, rows)

        with patch.object(backtest, "_write_csv", side_effect=inspect_write):
            self.run_snapshot(snapshot)

    def test_finite_large_errors_can_be_aggregated_without_overflow(self):
        class LargePoint(TrainingMean):
            def forecast(self, request):
                return 9e153

        run = self.run_snapshot(self.snapshot((0, 0, 0, 0, 0)), model=LargePoint())
        scores = self.read_json(run, "scores.json")
        self.assertTrue(scores["complete"])
        self.assertAlmostEqual(scores["rmse"] / 9e153, 1)

    def test_invalid_values_raise_before_output_is_created(self):
        for bad_value in ("", "not-a-number", "NaN", "inf", "-inf", "True"):
            with self.subTest(value=bad_value):
                snapshot = self.snapshot((1, 3, bad_value))
                output = self.root / f"bad-{self.counter}"
                with self.assertRaises(ValueError):
                    self.run_snapshot(snapshot, output_directory=output)
                self.assertFalse(output.exists())

    def test_invalid_timestamp_calendars_raise(self):
        cases = (
            ("2020-01-01", "2020-01-02", "2020-01-03"),
            ("invalid", "2020-01-02Z", "2020-01-03Z"),
            ("2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"),
            ("2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z"),
            ("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", "2020-01-04T00:00:00Z"),
        )
        for times in cases:
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.run_snapshot(self.snapshot((1, 3, 5), timestamps=times))

    def test_timezone_offsets_are_normalized_to_utc(self):
        snapshot = self.snapshot((1, 3, 5), timestamps=("2020-01-01T01:00:00+01:00",
                                 "2020-01-02T00:00:00Z", "2020-01-03T02:00:00+02:00"))
        run = self.run_snapshot(snapshot)
        self.assertEqual(self.read_csv(run)[0]["training_start"], "2020-01-01T00:00:00+00:00")

    def test_missing_schema_and_insufficient_history_raise(self):
        for snapshot in (self.snapshot(columns=("date", "value")), self.snapshot((1, 2)), self.snapshot(())):
            with self.subTest(snapshot=snapshot.snapshot_id), self.assertRaises(ValueError):
                self.run_snapshot(snapshot)

    def test_invalid_selection_and_minimum_history_raise(self):
        snapshot = self.snapshot()
        for selection in ({}, {"snapshot_id": snapshot.snapshot_id, "tag": "baseline"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                run_backtest(dataset="example", store=self.store, output_directory=self.root / "unused", **selection)
        for minimum in (0, -1, True, 1.5):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                run_backtest(dataset="example", snapshot_id=snapshot.snapshot_id, minimum_history=minimum,
                             store=self.store, output_directory=self.root / "unused")

    def test_integrity_check_happens_before_model_execution(self):
        snapshot = self.snapshot()
        snapshot.path.write_text("changed", encoding="utf-8")
        model = RecordingMean()
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.run_snapshot(snapshot, model=model)
        self.assertEqual(model.requests, [])

    def test_existing_outputs_are_preserved(self):
        snapshot = self.snapshot()
        run = self.run_snapshot(snapshot)
        original = {path.name: path.read_bytes() for path in run.iterdir()}
        with self.assertRaises(FileExistsError):
            self.run_snapshot(snapshot)
        self.assertEqual(original, {path.name: path.read_bytes() for path in run.iterdir()})

    def test_invalid_model_configuration_prevents_execution(self):
        class InvalidConfig(TrainingMean):
            @property
            def configuration(self):
                return {"bad": float("nan")}

        with self.assertRaises(ValueError):
            self.run_snapshot(self.snapshot(), model=InvalidConfig())

    def test_chart_reads_saved_results_without_data_or_fitting(self):
        run = self.run_snapshot(self.snapshot())
        original = {path.name: path.read_bytes() for path in run.iterdir()}
        shutil.rmtree(self.store)
        with patch.object(TrainingMean, "forecast", side_effect=AssertionError("No fitting in reports")):
            chart = plot_run(run)
        self.assertTrue(chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        for name, contents in original.items():
            self.assertEqual((run / name).read_bytes(), contents)
        with self.assertRaises(FileExistsError):
            plot_run(run)

    def test_chart_handles_a_run_with_no_successful_predictions(self):
        class BrokenModel(TrainingMean):
            def forecast(self, request):
                raise RuntimeError("broken")

        run = self.run_snapshot(self.snapshot(), model=BrokenModel())
        self.assertTrue(plot_run(run).exists())

    def test_chart_refuses_an_incomplete_run(self):
        run = self.run_snapshot(self.snapshot())
        manifest = self.read_json(run, "manifest.json")
        manifest["status"] = "running"
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finished"):
            plot_run(run)


if __name__ == "__main__":
    unittest.main()
