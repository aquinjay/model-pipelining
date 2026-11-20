"""Independent prior influence, matched histories, intervals and visible failures."""

import csv
from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scipy.integrate import quad
from scipy.stats import invgamma, t

from model_pipelining import prior_influence as pi
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask, ObservationVersion, select_history_as_of
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.predictive_checks import run_predictive_checks
from model_pipelining.sampling import MCMCSettings
from model_pipelining.snapshots import register_snapshot
from model_pipelining.tags import assign_tag
from test_normal_mean import request


class PriorInfluenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        self.reference = NormalMeanVariance(prior_mean_precision=1, prior_shape=3, prior_scale=2)
        self.history = request((1, 3, 5, 7, 9)).history
        self.snapshot = self.register((1, 3, 5, 7, 9))
        self.spec = ExperimentSpec(task=ForecastTask(minimum_history=2))

    def register(self, values):
        source = self.root / f"input-{len(list(self.root.glob('input-*')))}.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows((item.timestamp.isoformat(), value) for item, value in zip(self.history, values))
        return register_snapshot(source, dataset="example", store=self.store)

    def run_comparison(self, name="comparison", **kwargs):
        options = {"dataset": "example", "reference": self.reference,
                   "scenarios": {"shifted": replace(self.reference, prior_mean=10)},
                   "issued_at": self.history[-1].timestamp, "snapshot_id": self.snapshot.snapshot_id,
                   "store": self.store, "experiment": self.spec, "output_directory": self.root / name}
        options.update(kwargs)
        return pi.run_prior_influence(**options)

    def read_json(self, output, name):
        return json.loads((output/name).read_text())

    def rows(self, output, name="reference"):
        with (output / "learning_paths.csv").open(newline="") as stream:
            return [row for row in csv.DictReader(stream) if row["configuration"] == name]

    def test_hand_posterior_from_empty_prior_and_scale_components(self):
        output = self.run_comparison()
        rows = self.rows(output)
        self.assertEqual(len(rows), 6)
        prior, first, second = rows[:3]
        self.assertEqual(prior["stage"], "prior")
        self.assertEqual(prior["history_count"], "0")
        self.assertEqual(prior["prior_mean_weight"], "1.0")
        self.assertEqual(prior["issued_at"], "")
        self.assertAlmostEqual(float(first["mean_location"]), 0.5)
        self.assertAlmostEqual(float(second["mean_location"]), 4/3)
        self.assertAlmostEqual(float(second["scale"]), 13/3)
        self.assertAlmostEqual(float(second["observation_variance_mean"]), 13/9)
        self.assertAlmostEqual(float(second["predictive_variance"]), 52/27)
        for row in rows:
            components = sum(float(row[key]) for key in ("prior_scale_component", "data_scatter_component", "prior_data_disagreement_component"))
            self.assertAlmostEqual(float(row["scale"]), components)

    def test_location_coefficient_matches_counterfactual_mean_change(self):
        output = self.run_comparison()
        for reference, shifted in zip(self.rows(output), self.rows(output, "shifted")):
            n = int(reference["history_count"])
            self.assertAlmostEqual(float(reference["prior_mean_weight"]), 1/(1+n))
            self.assertAlmostEqual(float(shifted["mean_location_difference"]), 10/(1+n))
        # The conditional coefficient does not capture the variance change
        # caused by prior/data disagreement.
        self.assertNotEqual(self.rows(output)[2]["scale"], self.rows(output, "shifted")[2]["scale"])

    def test_intervals_have_independent_mass_and_correct_parameter_semantics(self):
        output = self.run_comparison()
        for row in self.rows(output)[:3]:
            a, b, k, m = (float(row[key]) for key in ("shape", "scale", "mean_precision", "mean_location"))
            variance_law = invgamma(a=a, scale=b)
            lo, hi = float(row["variance_lower"]), float(row["variance_upper"])
            self.assertAlmostEqual(quad(variance_law.pdf, lo, hi)[0], 0.95, places=8)
            self.assertAlmostEqual(variance_law.cdf(lo), 0.025)
            self.assertNotAlmostEqual(variance_law.pdf(lo), variance_law.pdf(hi), places=5)
            predictive = t(df=2*a, loc=m, scale=math.sqrt(b/a*(1+1/k)))
            lower, upper = float(row["predictive_lower"]), float(row["predictive_upper"])
            self.assertAlmostEqual(quad(predictive.pdf, lower, upper)[0], 0.95, places=8)
            self.assertAlmostEqual(predictive.pdf(lower), predictive.pdf(upper), places=10)
            self.assertAlmostEqual(float(row["predictive_width"]), upper-lower)

    def test_variance_concentration_preserves_prior_mean_but_changes_uncertainty(self):
        concentrated = replace(self.reference, prior_shape=30, prior_scale=29)
        output = self.run_comparison(scenarios={"concentrated": concentrated,
                                                "larger-scale": replace(self.reference, prior_scale=18)})
        prior, narrow = self.rows(output)[0], self.rows(output, "concentrated")[0]
        self.assertEqual(float(prior["observation_variance_mean"]), 1)
        self.assertEqual(float(narrow["observation_variance_mean"]), 1)
        self.assertLess(float(narrow["variance_upper"])-float(narrow["variance_lower"]),
                        float(prior["variance_upper"])-float(prior["variance_lower"]))
        self.assertNotEqual(self.rows(output)[2]["predictive_width"], self.rows(output, "larger-scale")[2]["predictive_width"])
        summaries = {row["name"]: row for row in self.read_json(output, "configuration_summaries.json")}
        self.assertEqual(summaries["concentrated"]["changed_mean_prior_settings"], [])
        self.assertEqual(summaries["concentrated"]["changed_variance_prior_settings"], ["prior_shape", "prior_scale"])

    def test_strong_persistent_prior_influence_completes_without_threshold(self):
        strong = replace(self.reference, prior_mean=100, prior_mean_precision=1e8)
        output = self.run_comparison(scenarios={"strong": strong})
        final = self.rows(output, "strong")[-1]
        self.assertGreater(float(final["prior_mean_weight"]), 0.9999)
        self.assertGreater(float(final["mean_location_difference"]), 90)
        self.assertEqual(self.read_json(output, "prior_influence.json")["status"], "completed")
        self.assertEqual(self.read_json(output, "review.json")["status"], "pending")

    def test_heavy_tail_prior_has_finite_bounds_and_explicit_undefined_moments(self):
        output = self.run_comparison(scenarios={"heavy": replace(self.reference, prior_shape=0.2)})
        prior, first = self.rows(output, "heavy")[:2]
        self.assertEqual(prior["predictive_mean"], "")
        self.assertEqual(prior["predictive_moment_status"], "mean_undefined")
        self.assertEqual(prior["observation_variance_mean"], "")
        self.assertEqual(prior["observation_variance_mean_difference"], "")
        self.assertTrue(math.isfinite(float(prior["predictive_upper"])))
        self.assertEqual(first["predictive_moment_status"], "variance_undefined")
        self.assertEqual(self.read_json(output, "prior_influence.json")["status"], "completed")

    def test_early_paths_start_before_minimum_history_and_identical_configs_match(self):
        output = self.run_comparison(scenarios={"identical": self.reference})
        self.assertEqual(self.rows(output)[1]["history_count"], "1")
        for row in self.rows(output, "identical"):
            self.assertEqual(float(row["predictive_width_difference"]), 0)
            self.assertEqual(float(row["mean_location_difference"]), 0)
        self.assertEqual(self.rows(output)[-1]["history_sha256"], self.rows(output, "identical")[-1]["history_sha256"])

    def test_future_final_values_leave_all_earlier_paths_unchanged(self):
        boundary = self.history[3].timestamp
        spec = replace(self.spec, evaluation=EvaluationPlan(final_start=boundary,
                        final_end=self.history[-1].timestamp+timedelta(days=1)))
        first = self.run_comparison("first", experiment=spec, issued_at=boundary-timedelta(seconds=1))
        altered = self.register((1, 3, 5, 700, 900))
        second = self.run_comparison("second", experiment=spec, issued_at=boundary-timedelta(seconds=1), snapshot_id=altered.snapshot_id)
        for path in first.rglob("*"):
            if path.is_file() and path.name != "prior_influence.json":
                self.assertEqual(path.read_bytes(), (second/path.relative_to(first)).read_bytes())
        third = self.run_comparison("third", snapshot_id=altered.snapshot_id)
        for name in ("reference", "shifted"):
            self.assertEqual(self.rows(first, name)[:4], self.rows(third, name)[:4])

    def test_delayed_releases_and_revisions_do_not_rewrite_earlier_updates(self):
        source = self.root / "releases.csv"
        timestamps = [item.timestamp for item in self.history]
        release_rows = [(timestamps[0], 1, timestamps[0]+timedelta(hours=6)),
                        (timestamps[0], 100, timestamps[2]),
                        (timestamps[1], 3, timestamps[1]+timedelta(hours=6)),
                        (timestamps[2], 5, timestamps[2]+timedelta(hours=6)),
                        (timestamps[3], 7, timestamps[3]+timedelta(hours=6))]
        def register(rows):
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "value", "released"))
                writer.writerows((time.isoformat(), value, released.isoformat()) for time, value, released in rows)
            return register_snapshot(source, dataset="example", store=self.store)
        spec = ExperimentSpec(task=ForecastTask(minimum_history=1, availability_column="released", revision_policy="latest_available"))
        cutoff = timestamps[3]+timedelta(hours=8)
        first = self.run_comparison("first", snapshot_id=register(release_rows).snapshot_id, experiment=spec, issued_at=cutoff)
        rows = self.rows(first)
        # Daily issuance at period 1 first learns value 1; at period 2 the
        # available revision is 100, alongside released value 3.
        self.assertEqual([r["history_count"] for r in rows], ["0", "1", "2", "3", "4"])
        self.assertAlmostEqual(float(rows[1]["mean_location"]), 0.5)
        self.assertAlmostEqual(float(rows[2]["mean_location"]), 103/3)
        release_rows[1] = (timestamps[0], 1000, timestamps[2])
        second = self.run_comparison("second", snapshot_id=register(release_rows).snapshot_id, experiment=spec, issued_at=cutoff)
        self.assertEqual(self.rows(first)[:2], self.rows(second)[:2])
        self.assertNotEqual(self.rows(first)[2:], self.rows(second)[2:])
        with (first/"learning_schedule.csv").open() as stream:
            schedule = list(csv.DictReader(stream))
        self.assertEqual(schedule[0]["selection"], "no_available_history")
        self.assertEqual(schedule[-1]["selection"], "explicit_cutoff")

    def test_daily_issuance_delay_controls_historical_selection(self):
        spec = replace(self.spec, task=replace(self.spec.task, issuance_delay_seconds=3600))
        cutoff = self.history[-1].timestamp+timedelta(hours=2)
        output = self.run_comparison(experiment=spec, issued_at=cutoff)
        first = self.rows(output)[1]
        self.assertEqual(first["issued_at"], (self.history[0].timestamp+timedelta(hours=1)).isoformat())
        self.assertEqual(self.rows(output)[-1]["issued_at"], cutoff.isoformat())

    def test_shared_selector_separates_period_cutoff_from_release_time(self):
        first, second, third = (item.timestamp for item in self.history[:3])
        versions = (ObservationVersion(first, 1, first), ObservationVersion(first, 100, third),
                    ObservationVersion(second, 3, second))
        self.assertEqual([item.value for item in select_history_as_of(versions, issued_at=second, period_end=first)], [1])
        self.assertEqual([item.value for item in select_history_as_of(versions, issued_at=third, period_end=first)], [100])
        with self.assertRaises(ValueError):
            select_history_as_of(versions, issued_at=first, period_end=second)
        with self.assertRaises(ValueError):
            select_history_as_of(versions, issued_at=first.replace(tzinfo=None))

    def test_final_cutoff_matches_predictive_checks_posterior(self):
        output = self.run_comparison()
        checks = run_predictive_checks(dataset="example", model=self.reference, snapshot_id=self.snapshot.snapshot_id,
                 store=self.store, experiment=self.spec, issued_at=self.history[-1].timestamp,
                 replications=3, output_directory=self.root/"predictive")
        joint = self.read_json(checks, "posterior/summary.json")["distribution"]
        final = self.rows(output)[-1]
        for key, column in (("mean", "mean_location"), ("mean_precision", "mean_precision"), ("shape", "shape"), ("scale", "scale")):
            self.assertEqual(float(final[column]), joint[key])

    def test_replay_tag_pinning_and_scenario_order_invariance(self):
        scenarios = {"a": replace(self.reference, prior_scale=4), "z": replace(self.reference, prior_mean=20)}
        first = self.run_comparison("first", scenarios=scenarios)
        second = self.run_comparison("second", scenarios=dict(reversed(list(scenarios.items()))))
        for path in first.iterdir():
            if path.name != "prior_influence.json":
                self.assertEqual(path.read_bytes(), (second/path.name).read_bytes())
        assign_tag(dataset="example", tag="selected", snapshot_id=self.snapshot.snapshot_id, changed_by="test", store=self.store)
        tagged = self.run_comparison("tagged", snapshot_id=None, tag="selected")
        old = self.snapshot.snapshot_id
        revised = self.register((9, 8, 7, 6, 5))
        assign_tag(dataset="example", tag="selected", snapshot_id=revised.snapshot_id, changed_by="test", store=self.store)
        self.assertEqual(self.read_json(tagged, "prior_influence.json")["context"]["input"]["snapshot_id"], old)

    def test_per_update_failure_preserves_later_and_other_configurations(self):
        original = pi._summary
        def fail_one(model, case, level):
            if model.prior_mean == 10 and case is not None and len(case.history) == 2:
                raise ValueError("injected calculation failure")
            return original(model, case, level)
        with patch.object(pi, "_summary", side_effect=fail_one):
            output = self.run_comparison()
        rows = self.rows(output, "shifted")
        self.assertEqual(rows[2]["status"], "failed")
        self.assertEqual(rows[2]["predictive_lower"], "")
        self.assertEqual(rows[3]["status"], "completed")
        self.assertEqual(self.read_json(output, "prior_influence.json")["status"], "failed")

    def test_failed_reference_keeps_alternatives_without_inventing_differences(self):
        original = pi._summary
        def fail_one(model, case, level):
            if model == self.reference and case is not None and len(case.history) == 2:
                raise ValueError("reference unavailable")
            return original(model, case, level)
        with patch.object(pi, "_summary", side_effect=fail_one):
            output = self.run_comparison()
        row = self.rows(output, "shifted")[2]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["comparison_status"], "reference_unavailable")
        self.assertEqual(row["mean_location_difference"], "")
        self.assertTrue(all(path.exists() for path in pi.plot_prior_influence(output)))

    def test_invalid_priors_boundaries_and_existing_outputs_raise_before_creation(self):
        boundary = self.history[3].timestamp
        final = replace(self.spec, evaluation=EvaluationPlan(role="final", final_start=boundary,
                        final_end=self.history[-1].timestamp+timedelta(days=1)))
        invalid = ({"scenarios": {}}, {"scenarios": {"reference": self.reference}}, {"scenarios": {"": self.reference}},
                   {"scenarios": {"x": object()}}, {"reference": replace(self.reference, mcmc=MCMCSettings())},
                   {"interval_level": 1}, {"interval_level": True}, {"snapshot_id": None},
                   {"tag": "selected"}, {"experiment": final}, {"issued_at": self.history[0].timestamp},
                   {"experiment": replace(final, evaluation=replace(final.evaluation, role="development"))},
                   {"issued_at": self.history[-1].timestamp.replace(tzinfo=None)})
        for index, options in enumerate(invalid):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_comparison(f"invalid-{index}", **options)
            self.assertFalse((self.root/f"invalid-{index}").exists())
        output = self.run_comparison()
        original = (output/"learning_paths.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_comparison()
        self.assertEqual(original, (output/"learning_paths.csv").read_bytes())

    def test_saved_only_plots_preserve_evidence_and_reject_overwrites(self):
        output = self.run_comparison()
        original = {path: path.read_bytes() for path in output.iterdir()}
        with patch.object(NormalMeanVariance, "posterior", side_effect=AssertionError("No model")), \
             patch.object(pi, "_select_development_history", side_effect=AssertionError("No input")):
            charts = pi.plot_prior_influence(output)
        self.assertTrue(all(path.read_bytes().startswith(b"\x89PNG") for path in charts))
        self.assertTrue(all(path.read_bytes() == value for path, value in original.items()))
        with self.assertRaises(FileExistsError):
            pi.plot_prior_influence(output)

    def test_persistence_error_propagates_and_leaves_running_manifest(self):
        original = pi.write_csv
        def fail(path, fields, rows):
            if path.name == "learning_paths.csv":
                raise OSError("disk failure")
            return original(path, fields, rows)
        with patch.object(pi, "write_csv", side_effect=fail), self.assertRaisesRegex(OSError, "disk failure"):
            self.run_comparison()
        self.assertEqual(self.read_json(self.root/"comparison", "prior_influence.json")["status"], "running")

    def test_cli_scenario_overrides_inherit_reference_and_draw_charts(self):
        config = self.root/"priors.json"
        config.write_text(json.dumps({"reference": {"prior_mean": 2, "prior_mean_precision": 7,
                          "prior_shape": 5, "prior_scale": 8}, "scenarios": {"shifted": {"prior_mean": 10}}}))
        experiment = self.root/"experiment.json"
        experiment.write_text(json.dumps(self.spec.to_dict()))
        output = self.root/"cli"
        result = subprocess.run([sys.executable, "scripts/compare_priors.py", "--dataset", "example",
            "--snapshot-id", self.snapshot.snapshot_id, "--store", str(self.store), "--as-of", self.history[-1].timestamp.isoformat(),
            "--experiment", str(experiment), "--config", str(config), "--output-directory", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        summaries = self.read_json(output, "configuration_summaries.json")
        self.assertEqual(summaries[1]["configuration"], {"prior_mean": 10, "prior_mean_precision": 7,
                                                       "prior_shape": 5, "prior_scale": 8, "mcmc": None})
        self.assertTrue((output/"prior_learning.png").is_file())
        self.assertTrue((output/"prior_differences.png").is_file())


if __name__ == "__main__":
    unittest.main()
