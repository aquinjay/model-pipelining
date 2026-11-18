"""Joint-replication mathematics, development isolation and saved-only criticism."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import t

from model_pipelining import predictive_checks as pc
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.sampling import MCMCSettings
from model_pipelining.snapshots import register_snapshot
from model_pipelining.tags import assign_tag
from test_normal_mean import request


class PredictiveSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NormalMeanVariance(prior_mean=2, prior_mean_precision=0.5, prior_shape=5, prior_scale=8)
        cls.prior = pc.simulate_prior_predictive(cls.model, observation_count=5, replications=16000, seed=42)
        cls.posterior = pc.simulate_posterior_predictive(cls.model, request((1, 3, 5, 7)), replications=16000, seed=42)

    def test_prior_joint_parameter_and_dataset_moments(self):
        result = self.prior
        values = np.array([d.values for d in result.datasets])
        means = np.array([d.mean for d in result.datasets])
        variances = np.array([d.observation_variance for d in result.datasets])
        self.assertFalse(result.failures)
        self.assertAlmostEqual(means.mean(), 2, delta=0.07)
        self.assertAlmostEqual(variances.mean(), 2, delta=0.04)
        self.assertAlmostEqual(means.var(), 4, delta=0.2)
        self.assertAlmostEqual(values[:, 0].var(), 6, delta=0.25)
        # A shared mu/v pair creates dependence across positions. Drawing an
        # independent marginal Student-t at every position would fail this.
        self.assertAlmostEqual(np.cov(values[:, 0], values[:, 1])[0, 1], 4, delta=0.2)
        self.assertAlmostEqual(values.mean(axis=1).var(), 4.4, delta=0.2)
        self.assertAlmostEqual(values.var(axis=1, ddof=1).mean(), 2, delta=0.06)
        residuals = (values - means[:, None]) / np.sqrt(variances[:, None])
        self.assertAlmostEqual(residuals.mean(), 0, delta=0.015)
        self.assertAlmostEqual(residuals.var(), 1, delta=0.025)

    def test_posterior_replication_moments_match_hand_updated_joint_law(self):
        # k=4.5, m=17/4.5, a=7, b=8+10+(0.5*4/4.5)*(4-2)^2/2.
        k, m, a, b = 4.5, 17 / 4.5, 7, 18 + 8 / 9
        noise, mean_variance = b / (a - 1), b / ((a - 1) * k)
        values = np.array([d.values for d in self.posterior.datasets])
        self.assertAlmostEqual(np.mean([d.mean for d in self.posterior.datasets]), m, delta=0.04)
        self.assertAlmostEqual(np.mean([d.observation_variance for d in self.posterior.datasets]), noise, delta=0.06)
        self.assertAlmostEqual(np.cov(values[:, 0], values[:, 1])[0, 1], mean_variance, delta=0.1)
        self.assertAlmostEqual(values.var(axis=1, ddof=1).mean(), noise, delta=0.07)
        distribution = t(df=2*a, loc=m, scale=math.sqrt((b/a)*(1+1/k)))
        for probability in (0.025, 0.5, 0.975):
            empirical = np.mean(values[:, 0] <= distribution.ppf(probability))
            self.assertAlmostEqual(empirical, probability, delta=5*math.sqrt(probability*(1-probability)/len(values)))

    def test_prior_does_not_fit_and_seed_replay_prefix_and_rng_isolation(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with patch.object(NormalMeanVariance, "posterior", side_effect=AssertionError("No conditioning")), \
             patch.object(NormalMeanVariance, "fit", side_effect=AssertionError("No fit")):
            first = pc.simulate_prior_predictive(self.model, observation_count=5, replications=5, seed=9)
            repeated = pc.simulate_prior_predictive(self.model, observation_count=5, replications=5, seed=9)
            longer = pc.simulate_prior_predictive(self.model, observation_count=8, replications=7, seed=9)
        self.assertEqual(first, repeated)
        for short, long in zip(first.datasets, longer.datasets):
            self.assertEqual(short.mean, long.mean)
            self.assertEqual(short.observation_variance, long.observation_variance)
            self.assertEqual(short.values, long.values[:5])
        self.assertEqual(python_state, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])

    def test_prior_and_posterior_streams_are_separate_and_seed_changes_draws(self):
        prior = pc.simulate_prior_predictive(self.model, observation_count=4, replications=3)
        posterior = pc.simulate_posterior_predictive(self.model, request((1, 3, 5, 7)), replications=3)
        self.assertNotEqual(prior.datasets[0].values, posterior.datasets[0].values)
        self.assertNotEqual(prior.datasets, pc.simulate_prior_predictive(self.model, observation_count=4, replications=3, seed=1).datasets)

    def test_partial_replication_failure_retains_indices_and_later_draws(self):
        original = np.random.Generator
        calls = []
        def factory(seed):
            generator = original(seed)
            calls.append(generator)
            if len(calls) == 2:
                return Mock(wraps=generator, gamma=Mock(return_value=0))
            return generator
        expected = pc.simulate_prior_predictive(self.model, observation_count=4, replications=4)
        with patch.object(np.random, "Generator", side_effect=factory):
            result = pc.simulate_prior_predictive(self.model, observation_count=4, replications=4)
        self.assertEqual([d.replication for d in result.datasets], [0, 2, 3])
        self.assertEqual(result.failures[0]["replication"], 1)
        self.assertEqual(result.datasets[-1], expected.datasets[-1])

    def test_heavy_tailed_prior_needs_no_finite_theoretical_moments(self):
        result = pc.simulate_prior_predictive(NormalMeanVariance(prior_shape=0.3), observation_count=2, replications=20)
        self.assertEqual(len(result.datasets), 20)
        self.assertTrue(all(math.isfinite(x) for d in result.datasets for x in d.values))

    def test_invalid_options_and_requested_mcmc_are_rejected(self):
        for options in ({"replications": 0}, {"replications": True}, {"seed": -1}, {"seed": True},
                        {"seed": 2**32}, {"observation_count": 0}, {"observation_count": True}):
            arguments = {"observation_count": 3, **options}
            with self.subTest(options=options), self.assertRaises(ValueError):
                pc.simulate_prior_predictive(self.model, **arguments)
        with self.assertRaises(ValueError):
            pc.simulate_prior_predictive(replace(self.model, mcmc=MCMCSettings()), observation_count=3)


class DiscrepancyTests(unittest.TestCase):
    def test_hand_statistics_and_linear_quantiles(self):
        result = pc.dataset_statistics((1, 2, 3, 4))
        sd = math.sqrt(5 / 3)
        expected = {"mean": 2.5, "sample_sd": sd, "minimum": 1, "maximum": 4,
                    "q05": 1.15, "median": 2.5, "q95": 3.85,
                    "maximum_standardized_deviation": 1.5/sd,
                    "lag_one_correlation": 1, "half_mean_difference": 2, "max_cusum": 1/sd}
        for key, value in expected.items():
            self.assertAlmostEqual(result[key]["value"], value)
            self.assertIsNone(result[key]["unavailable_reason"])
        self.assertAlmostEqual(pc.dataset_statistics((1, 2, 3, 4, 5))["half_mean_difference"]["value"], 2.5)

    def test_single_constant_and_invalid_data_have_explicit_behavior(self):
        for values in ((2,), (2, 2, 2)):
            result = pc.dataset_statistics(values)
            self.assertIsNone(result["lag_one_correlation"]["value"])
            self.assertIsNotNone(result["lag_one_correlation"]["unavailable_reason"])
            self.assertIsNone(result["max_cusum"]["value"])
        for values in ((), (math.nan,), ((1, 2), (3, 4))):
            with self.assertRaises(ValueError):
                pc.dataset_statistics(values)
        with self.assertRaises((ValueError, FloatingPointError)):
            pc.dataset_statistics((1e308, -1e308))

    def test_tail_precision_reports_zero_count_uncertainty(self):
        result = pc._tail_summary(0, 100)
        self.assertEqual(result["fraction"], 0)
        self.assertEqual(result["mcse"], 0)
        self.assertGreater(result["wilson_95"][1], 0.03)
        self.assertIsNone(pc._tail_summary(0, 0)["fraction"])
        middle = pc._tail_summary(50, 100)
        self.assertAlmostEqual(middle["mcse"], 0.05)
        self.assertLess(middle["wilson_95"][0], 0.5)
        self.assertGreater(middle["wilson_95"][1], 0.5)
        quantiles = pc._replicated_quantiles([-1e308, 1e308])
        self.assertEqual(quantiles["0.5"], 0)
        self.assertTrue(all(math.isfinite(x) for x in quantiles.values()))
        self.assertIsNone(pc._replicated_quantiles([]))


class PredictiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        self.model = NormalMeanVariance()
        self.history = request((1, 3, 5, 7, 9)).history
        self.snapshot = self.register((1, 3, 5, 7, 9))
        self.spec = ExperimentSpec(task=ForecastTask(minimum_history=2))

    def register(self, values):
        source = self.root / f"source-{len(list(self.root.glob('source-*')))}.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows((item.timestamp.isoformat(), value) for item, value in zip(self.history, values))
        return register_snapshot(source, dataset="example", store=self.store)

    def run_checks(self, name="checks", **kwargs):
        options = {"dataset": "example", "snapshot_id": self.snapshot.snapshot_id, "store": self.store,
                   "model": self.model, "issued_at": self.history[-1].timestamp,
                   "experiment": self.spec, "replications": 40, "output_directory": self.root / name}
        options.update(kwargs)
        return pc.run_predictive_checks(**options)

    def read_json(self, directory, name):
        return json.loads((directory / name).read_text())

    def test_saved_full_history_replicates_counts_and_pending_review(self):
        output = self.run_checks()
        manifest = self.read_json(output, "predictive_checks.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["observation_count"], 5)
        self.assertEqual(self.read_json(output, "review.json")["status"], "pending")
        self.assertEqual(self.read_json(output, "posterior/summary.json")["completed_replications"], 40)
        with (output / "posterior/replicated.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 200)
        self.assertIn("source_sha256", manifest)

    def test_prior_only_persists_without_reading_any_data_or_fitting(self):
        with patch.object(pc, "retrieve_snapshot", side_effect=AssertionError("No input")), \
             patch.object(NormalMeanVariance, "posterior", side_effect=AssertionError("No posterior")):
            output = pc.run_prior_predictive_checks(model=self.model, observation_count=5,
                replications=10, output_directory=self.root / "prior")
        manifest = self.read_json(output, "predictive_checks.json")
        self.assertIsNone(manifest["context"])
        self.assertEqual(manifest["phase_status"], {"prior": "completed"})
        self.assertFalse((output / "posterior").exists())
        self.assertIsNone(self.read_json(output, "prior/summary.json")["comparisons"]["mean"]["observed"])

    def test_replay_and_moved_tag_pin_preserve_numerical_files(self):
        first = self.run_checks("first")
        second = self.run_checks("second")
        for path in first.rglob("*"):
            if path.is_file() and path.name != "predictive_checks.json":
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        assign_tag(dataset="example", tag="selected", snapshot_id=self.snapshot.snapshot_id, changed_by="test", store=self.store)
        tagged = self.run_checks("tagged", snapshot_id=None, tag="selected")
        old_id = self.snapshot.snapshot_id
        revised = self.register((9, 8, 7, 6, 5))
        assign_tag(dataset="example", tag="selected", snapshot_id=revised.snapshot_id, changed_by="test", store=self.store)
        self.assertEqual(self.read_json(tagged, "predictive_checks.json")["context"]["input"]["snapshot_id"], old_id)

    def test_future_and_final_values_cannot_change_selected_checks(self):
        boundary = self.history[3].timestamp
        spec = replace(self.spec, evaluation=EvaluationPlan(final_start=boundary,
                       final_end=self.history[-1].timestamp + timedelta(days=1)))
        first = self.run_checks("first", experiment=spec, issued_at=boundary-timedelta(seconds=1))
        changed = self.register((1, 3, 5, 7000, 9000))
        second = self.run_checks("second", snapshot_id=changed.snapshot_id, experiment=spec,
                                 issued_at=boundary-timedelta(seconds=1))
        self.assertEqual(self.read_json(first, "predictive_checks.json")["observation_count"], 3)
        for path in first.rglob("*"):
            if path.is_file() and path.name != "predictive_checks.json":
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())

    def test_delayed_releases_revisions_and_future_revisions_are_selected_as_of(self):
        source = self.root / "released.csv"
        rows = [(self.history[0].timestamp, 1, self.history[0].timestamp + timedelta(hours=6)),
                (self.history[0].timestamp, 1000, self.history[-1].timestamp),
                (self.history[1].timestamp, 3, self.history[1].timestamp + timedelta(hours=6)),
                (self.history[2].timestamp, 5, self.history[2].timestamp + timedelta(hours=6))]
        def register(rows):
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "value", "released"))
                writer.writerows((timestamp.isoformat(), value, release.isoformat()) for timestamp, value, release in rows)
            return register_snapshot(source, dataset="example", store=self.store)
        spec = ExperimentSpec(task=ForecastTask(minimum_history=2, availability_column="released", revision_policy="latest_available"))
        first = self.run_checks("first", snapshot_id=register(rows).snapshot_id, experiment=spec, issued_at=self.history[2].timestamp)
        context = self.read_json(first, "predictive_checks.json")["context"]
        self.assertEqual([x["value"] for x in context["selected_history"]], [1, 3])
        rows[1] = (rows[1][0], -999, rows[1][2])
        second = self.run_checks("second", snapshot_id=register(rows).snapshot_id, experiment=spec, issued_at=self.history[2].timestamp)
        self.assertEqual((first / "posterior/replicated.csv").read_bytes(), (second / "posterior/replicated.csv").read_bytes())

    def test_prior_draws_ignore_observed_values_of_same_length(self):
        first = self.run_checks("first")
        second = self.run_checks("second", snapshot_id=self.register((10, 30, 50, 70, 90)).snapshot_id)
        for name in ("replicated.csv", "parameter_draws.csv", "statistics.csv"):
            self.assertEqual((first / "prior" / name).read_bytes(), (second / "prior" / name).read_bytes())
        self.assertNotEqual((first / "posterior/replicated.csv").read_bytes(), (second / "posterior/replicated.csv").read_bytes())

    def test_tail_counts_and_denominators_match_saved_statistics(self):
        output = self.run_checks()
        summary = self.read_json(output, "posterior/summary.json")
        with (output / "posterior/statistics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for name, comparison in summary["comparisons"].items():
            values = [float(r["value"]) for r in rows if r["statistic"] == name and r["value"]]
            self.assertEqual(len(values), comparison["valid_replications"])
            observed = comparison["observed"]
            if observed is not None:
                count = sum(x >= observed for x in values)
                self.assertEqual(comparison["greater_equal"]["count"], count)
                self.assertAlmostEqual(comparison["greater_equal"]["fraction"], count/len(values))

    def test_undefined_observed_statistics_are_explicit_and_not_failures(self):
        self.snapshot = self.register((2, 2, 2, 2, 2))
        output = self.run_checks()
        comparison = self.read_json(output, "posterior/summary.json")["comparisons"]["lag_one_correlation"]
        self.assertIsNone(comparison["observed"])
        self.assertIn("nonconstant", comparison["comparison_unavailable_reason"])
        self.assertIsNone(comparison["greater_equal"])
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "completed")

    def test_clear_level_shift_is_retained_as_criticism_without_failing_computation(self):
        self.history = request(tuple([0.0]*20 + [10.0]*20)).history
        self.snapshot = self.register(tuple([0.0]*20 + [10.0]*20))
        output = self.run_checks(replications=200)
        comparison = self.read_json(output, "posterior/summary.json")["comparisons"]["max_cusum"]
        self.assertEqual(comparison["greater_equal"]["count"], 0)
        self.assertGreater(comparison["greater_equal"]["wilson_95"][1], 0)
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "completed")
        self.assertEqual(self.read_json(output, "review.json")["status"], "pending")

    def test_failed_statistic_calculation_keeps_draws_and_explicit_missing_counts(self):
        original = pc.dataset_statistics
        calls = []
        def fail_once(values):
            calls.append(True)
            if len(calls) == 2:
                raise FloatingPointError("injected statistic overflow")
            return original(values)
        with patch.object(pc, "dataset_statistics", side_effect=fail_once):
            output = self.run_checks()
        prior = self.read_json(output, "prior/summary.json")
        self.assertEqual(prior["completed_replications"], 40)
        self.assertEqual(prior["statistic_failures"][0]["replication"], 0)
        self.assertEqual(prior["comparisons"]["mean"]["valid_replications"], 39)
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "failed")
        self.assertEqual(self.read_json(output, "posterior/summary.json")["status"], "completed")

    def test_single_observation_workflow_preserves_undefined_statistics(self):
        self.history = request((2,)).history
        self.snapshot = self.register((2,))
        spec = ExperimentSpec(task=ForecastTask(minimum_history=1))
        output = self.run_checks(experiment=spec)
        summary = self.read_json(output, "posterior/summary.json")
        self.assertEqual(summary["comparisons"]["sample_sd"]["valid_replications"], 0)
        self.assertEqual(summary["comparisons"]["sample_sd"]["missing_replications"], 40)
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "completed")

    def test_numerical_replication_failure_is_saved_and_posterior_continues(self):
        original = pc.simulate_prior_predictive
        def partial(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, datasets=result.datasets[1:], failures=({"replication": 0, "error": "injected failure"},))
        with patch.object(pc, "simulate_prior_predictive", side_effect=partial):
            output = self.run_checks()
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "failed")
        prior = self.read_json(output, "prior/summary.json")
        self.assertEqual(prior["completed_replications"], 39)
        self.assertEqual(prior["comparisons"]["mean"]["missing_replications"], 1)
        self.assertEqual(self.read_json(output, "posterior/summary.json")["status"], "completed")

    def test_posterior_computation_failure_retains_prior_evidence(self):
        with patch.object(NormalMeanVariance, "posterior", side_effect=ValueError("unrepresentable posterior")):
            output = self.run_checks()
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "failed")
        self.assertEqual(self.read_json(output, "prior/summary.json")["status"], "completed")
        self.assertIn("unrepresentable", self.read_json(output, "posterior/summary.json")["error"])
        self.assertTrue(all(path.exists() for path in pc.plot_predictive_checks(output)))

    def test_invalid_selection_final_roles_and_existing_outputs_raise_before_creation(self):
        boundary = self.history[3].timestamp
        final = replace(self.spec, evaluation=EvaluationPlan(role="final", final_start=boundary,
                        final_end=self.history[-1].timestamp + timedelta(days=1)))
        invalid = ({"snapshot_id": None}, {"tag": "x"}, {"experiment": final},
                   {"experiment": replace(final, evaluation=replace(final.evaluation, role="development"))},
                   {"issued_at": datetime(2020, 1, 1)}, {"issued_at": self.history[0].timestamp},
                   {"replications": 0}, {"model": replace(self.model, mcmc=MCMCSettings())})
        for index, options in enumerate(invalid):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_checks(f"bad-{index}", **options)
            self.assertFalse((self.root / f"bad-{index}").exists())
        output = self.run_checks()
        saved = (output / "predictive_checks.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_checks()
        self.assertEqual(saved, (output / "predictive_checks.json").read_bytes())

    def test_saved_only_plotting_preserves_evidence_and_refuses_overwrite(self):
        output = self.run_checks()
        original = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
        with patch.object(NormalMeanVariance, "posterior", side_effect=AssertionError("No fitting")), \
             patch.object(pc, "retrieve_snapshot", side_effect=AssertionError("No input")), \
             patch.object(pc, "_simulate", side_effect=AssertionError("No draws")):
            charts = pc.plot_predictive_checks(output)
        self.assertEqual(len(charts), 2)
        self.assertTrue(all(p.read_bytes().startswith(b"\x89PNG") for p in charts))
        self.assertTrue(all(p.read_bytes() == data for p, data in original.items()))
        with self.assertRaises(FileExistsError):
            pc.plot_predictive_checks(output)

    def test_persistence_failure_propagates_and_leaves_running_manifest(self):
        original = pc._write_csv
        def fail(path, fields, rows):
            if path.name == "replicated.csv":
                raise OSError("disk failure")
            return original(path, fields, rows)
        with patch.object(pc, "_write_csv", side_effect=fail), self.assertRaisesRegex(OSError, "disk failure"):
            self.run_checks()
        self.assertEqual(self.read_json(self.root / "checks", "predictive_checks.json")["status"], "running")

    def test_cli_prior_only_and_observed_modes_create_charts(self):
        commands = (("prior", ["--prior-only", "--observation-count", "5"]),
                    ("observed", ["--dataset", "example", "--snapshot-id", self.snapshot.snapshot_id,
                                  "--store", str(self.store), "--as-of", self.history[-1].timestamp.isoformat(),
                                  "--experiment", str(self.root / "experiment.json")]))
        (self.root / "experiment.json").write_text(json.dumps(self.spec.to_dict()))
        for name, flags in commands:
            output = self.root / name
            result = subprocess.run([sys.executable, "scripts/check_normal_mean_variance.py", *flags,
                                    "--replications", "10", "--output-directory", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((output / "prior_predictive.png").is_file())
            self.assertEqual((output / "posterior_predictive.png").exists(), name == "observed")


if __name__ == "__main__":
    unittest.main()
