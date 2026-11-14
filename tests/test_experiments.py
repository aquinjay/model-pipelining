"""Verify experiment semantics with release/vintage and holdout counterexamples."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model_pipelining import backtest
from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.local_level import LocalLevel
from model_pipelining.models import LastValue, TrainingMean
from model_pipelining.reporting import compare_runs, plot_run
from model_pipelining.snapshots import register_snapshot


def timestamp(day, hours=0):
    return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day, hours=hours)


class RecordingMean(TrainingMean):
    def __init__(self):
        self.requests = []

    def forecast(self, request):
        self.requests.append(request)
        return super().forecast(request)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        self.sequence = 0
        self.experiment = ExperimentSpec(task=ForecastTask(minimum_history=2),
                                         evaluation=EvaluationPlan(benchmarks=("training-mean", "last-value")))

    def register(self, rows=None, columns=("timestamp", "value")):
        self.sequence += 1
        source = self.root / f"source-{self.sequence}.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows if rows is not None else [(timestamp(index).isoformat(), value) for index, value in enumerate((1, 3, 5, 7, 9))])
        return register_snapshot(source, dataset="example", store=self.store)

    def run_experiment(self, snapshot, name="run", **options):
        return run_backtest(dataset="example", snapshot_id=snapshot.snapshot_id, store=self.store,
                            experiment=options.pop("experiment", self.experiment),
                            output_directory=self.root / name, **options)

    def read_json(self, run, name):
        return json.loads((run / name).read_text())

    def read_csv(self, run, name="predictions.csv"):
        with (run / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def vintage_fixture(self, revision=101):
        rows = [(timestamp(0).isoformat(), 1, timestamp(0).isoformat()),
                (timestamp(0).isoformat(), revision, timestamp(3).isoformat()),
                (timestamp(1).isoformat(), 3, timestamp(1, 12).isoformat()),
                (timestamp(1).isoformat(), 303, timestamp(5).isoformat())]
        rows.extend((timestamp(index).isoformat(), 2 * index + 1, timestamp(index, 12).isoformat()) for index in range(2, 6))
        snapshot = self.register(rows, ("timestamp", "value", "released"))
        experiment = replace(self.experiment, task=ForecastTask(minimum_history=1, availability_column="released",
                                                               revision_policy="latest_available"))
        return snapshot, experiment

    def test_named_target_and_units_do_not_require_an_application(self):
        snapshot = self.register(columns=("period", "measurement"))
        experiment = replace(self.experiment, task=ForecastTask(name="generic-series", timestamp_column="period",
                                                               target_column="measurement", units="test units", minimum_history=2))
        run = self.run_experiment(snapshot, experiment=experiment)
        self.assertEqual([float(row["prediction"]) for row in self.read_csv(run)], [2, 3, 4])
        manifest = self.read_json(run, "manifest.json")
        self.assertEqual(manifest["task"]["target_column"], "measurement")
        self.assertEqual(manifest["task"]["units"], "test units")
        self.assertEqual(manifest["experiment"], experiment.to_dict())

    def test_delayed_releases_and_revisions_are_selected_as_of_each_issuance(self):
        snapshot, experiment = self.vintage_fixture()
        model = RecordingMean()
        run = self.run_experiment(snapshot, experiment=experiment, model=model)
        self.assertEqual([[item.value for item in request.history] for request in model.requests],
                         [[1], [1], [1, 3], [101, 3, 5], [101, 3, 5, 7]])
        predictions = self.read_csv(run)
        self.assertEqual(predictions[1]["training_end"], timestamp(0).isoformat())
        self.assertEqual(predictions[1]["issued_at"], timestamp(1).isoformat())
        self.assertNotEqual(predictions[2]["history_sha256"], predictions[3]["history_sha256"])
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "completed")

    def test_release_at_issuance_is_included_and_later_revision_cannot_change_earlier_forecasts(self):
        snapshot, experiment = self.vintage_fixture()
        experiment = replace(experiment, task=replace(experiment.task, issuance_delay_seconds=43200))
        model = RecordingMean()
        first = self.run_experiment(snapshot, "first", experiment=experiment, model=model)
        self.assertEqual([item.value for item in model.requests[1].history], [1, 3])
        self.assertEqual(model.requests[1].issued_at, timestamp(1, 12))
        other, _ = self.vintage_fixture(revision=10001)
        changed = self.run_experiment(other, "changed", experiment=experiment)
        self.assertEqual(self.read_csv(first)[:3], self.read_csv(changed)[:3])
        self.assertNotEqual(self.read_csv(first)[3]["prediction"], self.read_csv(changed)[3]["prediction"])

    def test_scoring_vintage_is_explicit_and_does_not_change_model_history(self):
        snapshot, experiment = self.vintage_fixture()
        first = self.run_experiment(snapshot, "first", experiment=experiment)
        latest = self.run_experiment(snapshot, "latest", experiment=replace(experiment, task=replace(experiment.task, actual_policy="latest_in_snapshot")))
        self.assertEqual((first / "predictions.csv").read_bytes(), (latest / "predictions.csv").read_bytes())
        self.assertEqual(self.read_csv(first, "evaluation.csv")[0]["actual"], "3.0")
        self.assertEqual(self.read_csv(latest, "evaluation.csv")[0]["actual"], "303.0")
        self.assertEqual(self.read_csv(first, "evaluation.csv")[0]["actual_available_at"], timestamp(1, 12).isoformat())
        with self.assertRaisesRegex(ValueError, "must match"):
            compare_runs(first, latest, output_directory=self.root / "invalid-comparison")

    def test_minimum_history_counts_available_periods_and_exclusions_are_saved(self):
        snapshot, experiment = self.vintage_fixture()
        experiment = replace(experiment, task=replace(experiment.task, minimum_history=2))
        run = self.run_experiment(snapshot, experiment=experiment)
        schedule = self.read_csv(run, "case_schedule.csv")
        self.assertEqual([row["selection"] for row in schedule[:3]],
                         ["insufficient_available_history", "insufficient_available_history", "selected"])
        self.assertEqual(self.read_csv(run)[0]["target_time"], timestamp(3).isoformat())
        self.assertEqual(self.read_json(run, "scores.json")["expected_count"], 3)

    def test_development_and_final_target_periods_are_disjoint_and_sequential(self):
        snapshot = self.register([(timestamp(index).isoformat(), index + 1) for index in range(8)])
        evaluation = EvaluationPlan(final_start=timestamp(5), final_end=timestamp(7), benchmarks=("last-value",))
        experiment = replace(self.experiment, evaluation=evaluation)
        development = self.run_experiment(snapshot, "development", experiment=experiment)
        model = RecordingMean()
        final = self.run_experiment(snapshot, "final", experiment=replace(experiment, evaluation=replace(evaluation, role="final")), model=model)
        self.assertEqual([row["target_time"] for row in self.read_csv(development)], [timestamp(index).isoformat() for index in (2, 3, 4)])
        self.assertEqual([row["target_time"] for row in self.read_csv(final)], [timestamp(index).isoformat() for index in (5, 6)])
        self.assertEqual([item.value for item in model.requests[0].history], [1, 2, 3, 4, 5])
        self.assertEqual(model.requests[1].history[-1].value, 6)
        self.assertEqual(self.read_json(final, "manifest.json")["task"]["evaluation_role"], "final")
        self.assertEqual(self.read_json(final, "review.json")["status"], "pending")
        with self.assertRaises(ValueError):
            compare_runs(development, final, output_directory=self.root / "invalid-comparison")

    def test_changed_final_values_cannot_change_development_artifacts(self):
        experiment = replace(self.experiment, evaluation=EvaluationPlan(final_start=timestamp(3), final_end=timestamp(5), benchmarks=("last-value",)))
        first = self.run_experiment(self.register(), "first", experiment=experiment)
        changed = self.run_experiment(self.register([(timestamp(index).isoformat(), value) for index, value in enumerate((1, 3, 5, 700, 900))]),
                                      "changed", experiment=experiment)
        for name in ("predictions.csv", "evaluation.csv", "benchmark_predictions.csv", "benchmark_evaluation.csv", "scores.json"):
            self.assertEqual((first / name).read_bytes(), (changed / name).read_bytes())

    def test_benchmarks_use_identical_cases_and_known_point_scores(self):
        run = self.run_experiment(self.register())
        predictions = self.read_csv(run)
        benchmarks = self.read_csv(run, "benchmark_predictions.csv")
        scores = {item["name"]: item for item in self.read_json(run, "scores.json")["benchmarks"]["runs"]}
        self.assertEqual(scores["training-mean"]["mae"], 4)
        self.assertEqual(scores["last-value"]["mae"], 2)
        self.assertEqual(scores["last-value"]["rmse"], 2)
        for name in ("training-mean", "last-value"):
            rows = [row for row in benchmarks if row["benchmark"] == name]
            for primary, benchmark in zip(predictions, rows):
                for key in ("issued_at", "target_time", "history_sha256", "training_count"):
                    self.assertEqual(primary[key], benchmark[key])
                self.assertNotIn("actual", benchmark)

    def test_case_plan_and_all_forecasts_are_saved_before_actual_selection(self):
        selected_actuals = backtest.select_actuals
        calls = []
        def inspect_actuals(*arguments):
            run = self.root / "run"
            originals = {name: (run / name).read_bytes() for name in ("case_schedule.csv", "predictions.csv", "benchmark_predictions.csv")}
            calls.append(originals)
            self.assertFalse((run / "evaluation.csv").exists())
            return selected_actuals(*arguments)
        with patch.object(backtest, "select_actuals", side_effect=inspect_actuals):
            run = self.run_experiment(self.register())
        self.assertEqual(len(calls), 1)
        for name, original in calls[0].items():
            self.assertEqual((run / name).read_bytes(), original)

    def test_required_benchmark_failure_fails_run_even_with_complete_primary_scores(self):
        with patch.object(LastValue, "forecast", side_effect=(3, float("nan"), 7)):
            run = self.run_experiment(self.register())
        scores = self.read_json(run, "scores.json")
        self.assertTrue(scores["complete"])
        self.assertEqual(scores["benchmarks"]["status"], "failed")
        last = next(item for item in scores["benchmarks"]["runs"] if item["name"] == "last-value")
        self.assertEqual((last["expected_count"], last["scored_count"], last["failed_count"]), (3, 2, 1))
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")

    def test_horizon_is_runner_owned_and_unsupported_adapter_requests_fail_visibly(self):
        experiment = replace(self.experiment, task=replace(self.experiment.task, horizon_days=2))
        model = RecordingMean()
        snapshot = self.register()
        self.run_experiment(snapshot, experiment=experiment, model=model)
        self.assertTrue(all(request.target_time - request.issued_at == timedelta(days=2) for request in model.requests))
        failed = self.run_experiment(snapshot, "local-level", experiment=experiment, model=LocalLevel())
        self.assertEqual(self.read_json(failed, "manifest.json")["status"], "failed")
        self.assertTrue(all("one-day target" in row["error"] for row in self.read_csv(failed)))

    def test_invalid_calendars_releases_and_ambiguous_revisions_raise_before_output(self):
        task = ForecastTask(minimum_history=1, availability_column="released", revision_policy="latest_available")
        examples = (
            [(timestamp(0).isoformat(), 1, timestamp(0).isoformat()), (timestamp(2).isoformat(), 2, timestamp(2).isoformat())],
            [(timestamp(0).isoformat(), 1, timestamp(0).isoformat()), (timestamp(0).isoformat(), 2, timestamp(0).isoformat())],
            [(timestamp(0).isoformat(), 1, timestamp(-1).isoformat())],
            [(timestamp(0).isoformat(), 1, "2020-01-01")],
        )
        for index, rows in enumerate(examples):
            snapshot = self.register(rows, ("timestamp", "value", "released"))
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.run_experiment(snapshot, str(index), experiment=replace(self.experiment, task=task))
            self.assertFalse((self.root / str(index)).exists())
        with self.assertRaises(ValueError):
            ForecastTask(calendar="market_days")
        with self.assertRaises(ValueError):
            ForecastTask(missing_policy="impute")
        with self.assertRaises(ValueError):
            ForecastTask(revision_policy="latest_available")

    def test_invalid_split_or_conflicting_configuration_cannot_start_run(self):
        for options in ({"role": "final"}, {"final_start": timestamp(2)}, {"final_start": timestamp(3), "final_end": timestamp(2)},
                        {"final_start": datetime(2020, 1, 1), "final_end": timestamp(2)}, {"benchmarks": ("unknown",)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EvaluationPlan(**options)
        snapshot = self.register()
        with self.assertRaisesRegex(ValueError, "both interfaces"):
            self.run_experiment(snapshot, minimum_history=2)
        empty = replace(self.experiment, evaluation=EvaluationPlan(role="final", final_start=timestamp(10), final_end=timestamp(11)))
        with self.assertRaises(ValueError):
            self.run_experiment(snapshot, experiment=empty)
        self.assertFalse((self.root / "run").exists())

    def test_saved_experiment_replays_and_comparison_checks_history_fingerprints(self):
        snapshot = self.register()
        first = self.run_experiment(snapshot, "first")
        spec = ExperimentSpec.from_dict(self.read_json(first, "manifest.json")["experiment"])
        second = self.run_experiment(snapshot, "second", experiment=spec)
        for name in ("case_schedule.csv", "predictions.csv", "evaluation.csv", "benchmark_predictions.csv", "benchmark_evaluation.csv", "scores.json", "checks.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        original = {path: path.read_bytes() for run in (first, second) for path in run.iterdir()}
        with patch.object(TrainingMean, "forecast", side_effect=AssertionError("No fitting in reports")), \
             patch.object(LastValue, "forecast", side_effect=AssertionError("No benchmarks in reports")):
            plot_run(first)
            compare_runs(first, second, output_directory=self.root / "comparison")
        self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))
        contents = (second / "predictions.csv").read_text()
        history_hash = self.read_csv(second)[0]["history_sha256"]
        (second / "predictions.csv").write_text(contents.replace(history_hash, "0" * 64))
        with self.assertRaisesRegex(ValueError, "must match"):
            compare_runs(first, second, output_directory=self.root / "changed-history")
        # Legacy optional fields may be absent; original required coordinates
        # must not silently disappear from both sides of a comparison.
        for run in (first, second):
            rows = [{key: value for key, value in row.items() if key != "training_count"} for row in self.read_csv(run)]
            with (run / "predictions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with self.assertRaises((KeyError, ValueError)):
            compare_runs(first, second, output_directory=self.root / "missing-coordinates")

    def test_version_two_point_artifacts_still_render_and_compare(self):
        snapshot = self.register()
        for name in ("old-a", "old-b"):
            run = self.run_experiment(snapshot, name, experiment=replace(self.experiment, evaluation=EvaluationPlan()))
            manifest = self.read_json(run, "manifest.json")
            manifest["format_version"] = 2
            manifest.pop("experiment")
            manifest.pop("experiment_sha256")
            (run / "manifest.json").write_text(json.dumps(manifest))
            for filename, removed in (("predictions.csv", {"training_end", "history_sha256"}), ("evaluation.csv", {"actual_available_at"})):
                rows = [{key: value for key, value in row.items() if key not in removed} for row in self.read_csv(run, filename)]
                with (run / filename).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            plot_run(run)
        comparison = compare_runs(self.root / "old-a", self.root / "old-b", output_directory=self.root / "old-comparison")
        self.assertEqual(self.read_json(comparison, "comparison.json")["case_count"], 3)

    def test_both_clis_load_spec_and_require_explicit_final_invocation(self):
        snapshot = self.register()
        spec = replace(self.experiment, evaluation=replace(self.experiment.evaluation, final_start=timestamp(3), final_end=timestamp(5)))
        configuration = self.root / "experiment.json"
        configuration.write_text(json.dumps(spec.to_dict()))
        for script, name in (("run_baseline.py", "baseline"), ("run_local_level.py", "local")):
            output = self.root / name
            result = subprocess.run([sys.executable, f"scripts/{script}", "--dataset", "example", "--snapshot-id", snapshot.snapshot_id,
                                     "--store", str(self.store), "--experiment", str(configuration), "--evaluation-role", "final",
                                     "--output-directory", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.read_json(output, "scores.json")["expected_count"], 2)
            self.assertEqual(self.read_json(output, "manifest.json")["experiment"]["evaluation"]["role"], "final")
            self.assertTrue((output / "forecast.png").is_file())


if __name__ == "__main__":
    unittest.main()
