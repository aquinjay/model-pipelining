"""Information/cohort boundaries in the explicit development review exercise."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from scipy.stats import t


spec = importlib.util.spec_from_file_location(
    "review_model_study", Path(__file__).resolve().parents[1] / "scripts/review_model_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    continuation_spec = importlib.util.spec_from_file_location(
        "continue_model_review", Path(__file__).resolve().parents[1] / "scripts/continue_model_review.py")
    continuation = importlib.util.module_from_spec(continuation_spec)
    continuation_spec.loader.exec_module(continuation)
finally:
    sys.path.pop(0)


class ModelReviewStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.snapshots = study.generate_sources(cls.root)
        experiment = study.ExperimentSpec(task=study.ForecastTask(minimum_history=30),
            evaluation=study.EvaluationPlan(final_start=study.START + study.timedelta(days=160),
                                            final_end=study.START + study.timedelta(days=200)))
        cls.forecast = study.run_backtest(dataset="mean-shift", snapshot_id=cls.snapshots["mean-shift"],
            store=cls.root / "store", model=study.NormalMeanVariance(), experiment=experiment,
            interval_levels=(.8, .95), output_directory=cls.root / "forecast")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy(self, directory):
        return Path(shutil.copytree(self.forecast, Path(directory) / "forecast"))

    def test_original_fixture_bytes_and_shared_noise_are_preserved(self):
        self.assertEqual(self.snapshots["mean-shift"], "c7e9fbed4871a4e77166afc8e91b03de38881731bca8a8a370bdc91ebfa5cdf5")
        self.assertEqual(self.snapshots["stable"], "2510db2af47e4b61c8f356ddc41df922d8992611afaa94cb75def102bec6dc68")
        stable, ramp = [study.read_csv(self.root / "source" / f"{name}.csv") for name in ("stable", "gradual")]
        for index, (a, b) in enumerate(zip(stable, ramp)):
            self.assertAlmostEqual(float(b["value"]) - float(a["value"]), .05 * max(0, index - 99), places=12)

    def test_saved_only_metrics_use_full_cdf_and_exclude_final_targets(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.forecast.rglob("*") if p.is_file()}
        with patch.object(study.NormalMeanVariance, "posterior", side_effect=AssertionError("No fits")):
            rows = study.case_metrics(self.forecast)
        self.assertEqual([row["day"] for row in rows], list(range(30, 160)))
        law = study.read_json(self.forecast / "predictive_distributions.json")[70]["distribution"]
        self.assertAlmostEqual(rows[70]["pit"], t.cdf(rows[70]["actual"], law["df"], loc=law["location"], scale=law["scale"]), places=14)
        self.assertEqual(study.summarize_window(rows, (100, 120))["cases"], 20)
        self.assertEqual(sum(row["covered_95"] for row in rows[70:90]), 4)
        self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items()))

    def test_matching_requires_history_and_actual_vintage_not_only_target(self):
        rows = study.case_metrics(self.forecast)
        for key in ("target_time", "issued_at", "history_sha256", "actual", "actual_available_at"):
            changed = [dict(row) for row in rows]
            changed[0][key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "histories or actual vintages"):
                study.matched(rows, changed)
        with self.assertRaises(ValueError):
            study.matched(rows, rows[:-1])

    def test_partial_duplicate_and_mismatched_hpd_cohorts_are_rejected(self):
        for change in ("missing", "duplicate", "actual", "failed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                directory = self.copy(temporary)
                rows = study.read_csv(directory / "highest_density_evaluation.csv")
                if change == "missing":
                    rows = [row for row in rows if not (float(row["level"]) == .95 and row["target_time"] == rows[0]["target_time"])]
                elif change == "duplicate":
                    rows.append(next(row for row in rows if float(row["level"]) == .95))
                else:
                    row = next(row for row in rows if float(row["level"]) == .95)
                    row["actual" if change == "actual" else "status"] = "999" if change == "actual" else "failed"
                study.write_csv(directory / "highest_density_evaluation.csv", rows)
                with self.assertRaises(ValueError):
                    study.case_metrics(directory)

    def test_failed_forecasts_and_incomplete_distribution_scores_are_rejected(self):
        for filename in ("manifest.json", "distribution_evaluation.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = self.copy(temporary)
                if filename.endswith("json"):
                    manifest = study.read_json(directory / filename)
                    manifest["status"] = "failed"
                    study.write_json(directory / filename, manifest)
                else:
                    rows = study.read_csv(directory / filename)
                    study.write_csv(directory / filename, rows[:-1])
                with self.assertRaisesRegex(ValueError, "complete|incomplete"):
                    study.case_metrics(directory)

    def test_window_boundaries_do_not_include_neighbors_or_skip_missing_days(self):
        rows = study.case_metrics(self.forecast)
        onset = study.summarize_window(rows, (100, 120))
        # Extreme neighboring observations must not alter this window's summary.
        changed = [dict(row) for row in rows]
        for row in changed:
            if row["day"] in (99, 120):
                row.update(crps=1e9, log_score=1e9, standardized_innovation=1e9)
        self.assertEqual(study.summarize_window(changed, (100, 120)), onset)
        with self.assertRaisesRegex(ValueError, "every declared daily target"):
            study.summarize_window([row for row in rows if row["day"] != 110], (100, 120))

    def test_continuation_rejects_changed_sources_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.copy(temporary)
            self.assertTrue(continuation.completed(directory, "manifest.json"))
            manifest = study.read_json(directory / "manifest.json")
            manifest["source_sha256"]["normal_mean_variance.py"] = "0" * 64
            study.write_json(directory / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "incompatible source"):
                continuation.completed(directory, "manifest.json")
            manifest["status"] = "failed"
            study.write_json(directory / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "retained failed artifact"):
                continuation.completed(directory, "manifest.json")
            self.assertTrue(directory.exists())
            manifest["status"] = "running"
            study.write_json(directory / "manifest.json", manifest)
            before = (directory / "manifest.json").read_bytes()
            self.assertFalse(continuation.completed(directory, "manifest.json"))
            preserved = list(Path(temporary).glob("forecast-interrupted-*"))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / "manifest.json").read_bytes(), before)

    def test_continuation_rejects_forecast_that_differs_from_saved_design(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "mean-shift/constant/reference"
            shutil.copytree(self.forecast, target)
            manifest = study.read_json(target / "manifest.json")
            changed = {**manifest["model"]["configuration"], "prior_mean": 999}
            study.write_json(root / "study.json", {"experiment": manifest["experiment"],
                "models": {"constant": {"reference": changed, "alternatives": {}}},
                "snapshots": {"mean-shift": self.snapshots["mean-shift"]}, "interval_levels": [.8, .95]})
            with patch.object(study.NormalMeanVariance, "fit", side_effect=AssertionError("No refit")), \
                    self.assertRaisesRegex(ValueError, "differs from saved study design"):
                continuation.run_job(str(root), "mean-shift", "constant")

    def test_continuation_accepts_json_lists_for_joint_configuration(self):
        # Joint settings use tuples in Python and lists in saved JSON. A real
        # short forecast must be reused rather than falsely rejected or refit.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            study.write_csv(source, [{"timestamp": (study.START + study.timedelta(days=i)).isoformat(),
                                      "value": float(i)} for i in range(4)])
            snapshot = study.register_snapshot(source, dataset="joint", store=root / "store")
            experiment = study.ExperimentSpec(task=study.ForecastTask(minimum_history=2),
                evaluation=study.EvaluationPlan(final_start=study.START + study.timedelta(days=4),
                                                final_end=study.START + study.timedelta(days=5)))
            model = study.LocalLevelJointVariance(evolution_variances=(0., .1))
            target = root / "joint/joint/reference"
            study.run_backtest(dataset="joint", snapshot_id=snapshot.snapshot_id, store=root / "store",
                experiment=experiment, model=model, interval_levels=(.8, .95), output_directory=target)
            configuration = json.loads(json.dumps(model.configuration))
            self.assertIsInstance(configuration["evolution_variances"], list)
            self.assertIsInstance(model.configuration["evolution_variances"], tuple)
            study.write_json(root / "study.json", {"experiment": experiment.to_dict(),
                "models": {"joint": {"reference": configuration, "alternatives": {}}},
                "snapshots": {"joint": snapshot.snapshot_id}, "interval_levels": [.8, .95]})
            with patch.object(continuation, "run_backtest", side_effect=AssertionError("No refit")), \
                    patch.object(continuation, "plot_run"), \
                    patch.object(continuation, "run_experiment_evidence", side_effect=RuntimeError("Passed JSON guard")), \
                    self.assertRaisesRegex(RuntimeError, "Passed JSON guard"):
                continuation.run_job(str(root), "joint", "joint")


if __name__ == "__main__":
    unittest.main()
