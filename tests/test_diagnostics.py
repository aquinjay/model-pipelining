"""Check influence meaning and saved evidence independently of agreement goals."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy

from model_pipelining.diagnostics import plot_local_level_diagnostics, run_local_level_diagnostics
from model_pipelining.local_level import LocalLevel
from model_pipelining.snapshots import register_snapshot
from model_pipelining.tags import assign_tag


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        self.snapshot = self.register((2, 4, 3, 6, 5))
        self.reference = LocalLevel(initial_mean=0, initial_variance=1, observation_variance=1, evolution_variance=1)

    def register(self, values):
        source = self.root / "input.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows(((datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat(), value)
                             for index, value in enumerate(values))
        return register_snapshot(source, dataset="example", store=self.store)

    def run_diagnostics(self, name="diagnostics", **options):
        return run_local_level_diagnostics(
            dataset="example", snapshot_id=self.snapshot.snapshot_id, store=self.store,
            reference=options.pop("reference", self.reference),
            scenarios=options.pop("scenarios", {"shifted-mean": replace(self.reference, initial_mean=10)}),
            output_directory=self.root / name, **options,
        )

    def read_json(self, output, filename):
        return json.loads((output / filename).read_text())

    def rows(self, output, name="reference"):
        with (output / "update_paths.csv").open(newline="") as stream:
            return [row for row in csv.DictReader(stream) if row["scenario"] == name]

    def test_hand_calculated_update_weights_and_prediction_errors(self):
        self.snapshot = self.register((2, 4))
        output = self.run_diagnostics()
        first, second = self.rows(output)
        self.assertAlmostEqual(float(first["level_mean"]), 4 / 3)
        self.assertAlmostEqual(float(first["update_gain"]), 2 / 3)
        self.assertAlmostEqual(float(first["initial_mean_weight"]), 1 / 3)
        self.assertAlmostEqual(float(second["level_mean"]), 3)
        self.assertAlmostEqual(float(second["level_sd"]) ** 2, 5 / 8)
        self.assertAlmostEqual(float(second["next_observation_sd"]) ** 2, 21 / 8)
        self.assertAlmostEqual(float(second["initial_mean_weight"]), 1 / 8)
        self.assertAlmostEqual(float(first["standardized_innovation"]), 2 / math.sqrt(3))
        self.assertAlmostEqual(float(first["pit"]), (1 + math.erf(2 / math.sqrt(6))) / 2)
        criticism = self.read_json(output, "scenario_summaries.json")[0]["criticism"]
        self.assertAlmostEqual(criticism["mean_squared_standardized_innovation"], 2)
        self.assertIsNone(criticism["lag_one_correlation"])
        self.assertIsNotNone(criticism["lag_one_unavailable_reason"])

    def test_initial_mean_derivative_matches_independent_batch_conditioning(self):
        output = self.run_diagnostics()
        for index, row in enumerate(self.rows(output), start=1):
            times = numpy.arange(1, index + 1)
            covariance = 1 + numpy.minimum.outer(times, times) + numpy.eye(index)
            cross_covariance = 1 + times
            derivative = 1 - cross_covariance @ numpy.linalg.solve(covariance, numpy.ones(index))
            self.assertAlmostEqual(float(row["initial_mean_weight"]), derivative, places=12)
        shifted = self.rows(output, "shifted-mean")
        for baseline, alternative in zip(self.rows(output), shifted):
            self.assertAlmostEqual(float(alternative["level_mean_difference"]), 10 * float(baseline["initial_mean_weight"]))
            self.assertEqual(alternative["level_sd"], baseline["level_sd"])

    def test_strong_persistent_prior_influence_completes_without_approval(self):
        reference = LocalLevel(initial_mean=0, initial_variance=0, evolution_variance=0)
        output = self.run_diagnostics(reference=reference, scenarios={"known-high-level": replace(reference, initial_mean=100)})
        self.assertEqual(self.read_json(output, "diagnostics.json")["status"], "completed")
        for row in self.rows(output, "known-high-level"):
            self.assertEqual(float(row["initial_mean_weight"]), 1)
            self.assertEqual(float(row["level_mean_difference"]), 100)
        review = self.read_json(output, "review.json")
        self.assertEqual((review["status"], review["decision"]), ("pending", None))

    def test_prior_variance_and_noise_changes_have_distinct_saved_meaning(self):
        output = self.run_diagnostics(scenarios={
            "tighter-prior": replace(self.reference, initial_variance=0),
            "more-noise": replace(self.reference, observation_variance=4),
            "more-movement": replace(self.reference, evolution_variance=4),
        })
        summaries = {item["name"]: item for item in self.read_json(output, "scenario_summaries.json")}
        self.assertEqual(summaries["tighter-prior"]["changed_prior_settings"], ["initial_variance"])
        self.assertEqual(summaries["more-noise"]["changed_prior_settings"], [])
        self.assertEqual(summaries["more-noise"]["changed_noise_settings"], ["observation_variance"])
        baseline = self.rows(output)[0]
        self.assertLess(float(self.rows(output, "tighter-prior")[0]["update_gain"]), float(baseline["update_gain"]))
        self.assertLess(float(self.rows(output, "more-noise")[0]["update_gain"]), float(baseline["update_gain"]))
        self.assertGreater(float(self.rows(output, "more-movement")[0]["update_gain"]), float(baseline["update_gain"]))
        self.assertNotEqual(self.rows(output, "more-noise")[-1]["next_observation_sd"], self.rows(output)[-1]["next_observation_sd"])

    def test_future_changes_do_not_change_any_earlier_update_or_criticism_row(self):
        first = self.run_diagnostics("first")
        self.snapshot = self.register((2, 4, 300, 600, 500))
        changed = self.run_diagnostics("changed")
        for name in ("reference", "shifted-mean"):
            self.assertEqual(self.rows(first, name)[:2], self.rows(changed, name)[:2])
            self.assertNotEqual(self.rows(first, name)[2:], self.rows(changed, name)[2:])

    def test_configuration_replay_and_tag_pin_preserve_evidence(self):
        first = self.run_diagnostics("first")
        manifest = self.read_json(first, "diagnostics.json")
        summaries = self.read_json(first, "scenario_summaries.json")
        models = {item["name"]: LocalLevel(**item["configuration"]) for item in summaries}
        replay = self.run_diagnostics("replay", reference=models.pop("reference"), scenarios=models)
        for filename in ("update_paths.csv", "scenario_summaries.json", "review.json"):
            self.assertEqual((first / filename).read_bytes(), (replay / filename).read_bytes())
        original_id = self.snapshot.snapshot_id
        assign_tag(dataset="example", tag="selected", snapshot_id=original_id, changed_by="test", store=self.store)
        tagged = run_local_level_diagnostics(dataset="example", tag="selected", store=self.store,
                                             reference=self.reference, scenarios=models, output_directory=self.root / "tagged")
        self.snapshot = self.register((9, 8, 7, 6, 5))
        assign_tag(dataset="example", tag="selected", snapshot_id=self.snapshot.snapshot_id, changed_by="test", store=self.store)
        self.assertEqual(self.read_json(tagged, "diagnostics.json")["input"]["snapshot_id"], original_id)
        self.assertEqual(manifest["input"]["snapshot_id"], original_id)
        self.assertEqual((first / "update_paths.csv").read_bytes(), (tagged / "update_paths.csv").read_bytes())

    def test_numeric_failure_is_saved_and_other_configurations_continue(self):
        broken = replace(self.reference, initial_variance=1e308, evolution_variance=1e308)
        output = self.run_diagnostics(scenarios={"broken": broken, "valid": self.reference})
        manifest = self.read_json(output, "diagnostics.json")
        self.assertEqual((manifest["status"], manifest["completed_configurations"], manifest["failed_configurations"]), ("failed", 2, 1))
        summaries = {item["name"]: item for item in self.read_json(output, "scenario_summaries.json")}
        self.assertIn("invalid predictive variance", summaries["broken"]["error"])
        self.assertEqual(self.rows(output, "broken"), [])
        self.assertEqual(len(self.rows(output, "valid")), 5)

    def test_failed_reference_never_fabricates_comparison_differences(self):
        broken = replace(self.reference, initial_variance=1e308, evolution_variance=1e308)
        output = self.run_diagnostics(reference=broken, scenarios={"valid": self.reference})
        self.assertEqual(self.read_json(output, "diagnostics.json")["status"], "failed")
        self.assertTrue(all(row["level_mean_difference"] == "" for row in self.rows(output, "valid")))
        self.assertIsNone(self.read_json(output, "scenario_summaries.json")[1]["influence"])
        self.assertTrue(all(path.exists() for path in plot_local_level_diagnostics(output)))

    def test_invalid_options_input_and_existing_outputs_are_rejected(self):
        for index, scenarios in enumerate(({}, {"reference": self.reference}, {"": self.reference}, {"bad": object()})):
            with self.subTest(scenarios=scenarios), self.assertRaises(ValueError):
                self.run_diagnostics(str(index), scenarios=scenarios)
            self.assertFalse((self.root / str(index)).exists())
        self.snapshot = self.register(())
        with self.assertRaises(ValueError):
            self.run_diagnostics("empty")
        self.assertFalse((self.root / "empty").exists())
        self.snapshot = self.register((2, 4))
        output = self.run_diagnostics()
        original = {path: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.run_diagnostics()
        self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))

    def test_charts_use_saved_results_preserve_sources_and_refuse_overwrites(self):
        output = self.run_diagnostics()
        original = {path: path.read_bytes() for path in output.iterdir()}
        with patch.object(LocalLevel, "filter_history", side_effect=AssertionError("No model calls")), \
             patch("model_pipelining.diagnostics.retrieve_snapshot", side_effect=AssertionError("No input retrieval")):
            charts = plot_local_level_diagnostics(output)
        self.assertTrue(all(path.read_bytes().startswith(b"\x89PNG") for path in charts))
        self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))
        with self.assertRaises(FileExistsError):
            plot_local_level_diagnostics(output)

    def test_constant_and_single_observation_residuals_have_explicit_missing_correlation(self):
        for name, values in (("single", (2,)), ("constant", (0, 0, 0))):
            self.snapshot = self.register(values)
            output = self.run_diagnostics(name, reference=LocalLevel(initial_mean=0, initial_variance=0, evolution_variance=0))
            summary = self.read_json(output, "scenario_summaries.json")[0]
            self.assertEqual(summary["status"], "completed")
            self.assertIsNone(summary["criticism"]["lag_one_correlation"])

    def test_cli_overrides_inherit_reference_and_create_saved_charts(self):
        configuration = self.root / "config.json"
        configuration.write_text(json.dumps({"reference": {"initial_mean": 0, "initial_variance": 7,
                                                            "observation_variance": 3, "evolution_variance": 0.2},
                                              "scenarios": {"shifted": {"initial_mean": 10}}}))
        output = self.root / "cli"
        result = subprocess.run([sys.executable, "scripts/diagnose_local_level.py", "--dataset", "example",
                                 "--snapshot-id", self.snapshot.snapshot_id, "--store", str(self.store),
                                 "--config", str(configuration), "--output-directory", str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summaries = self.read_json(output, "scenario_summaries.json")
        self.assertEqual(summaries[1]["configuration"], {"initial_mean": 10, "initial_variance": 7,
                                                       "observation_variance": 3, "evolution_variance": 0.2})
        self.assertTrue((output / "sensitivity.png").exists())
        self.assertTrue((output / "innovations.png").exists())


if __name__ == "__main__":
    unittest.main()
