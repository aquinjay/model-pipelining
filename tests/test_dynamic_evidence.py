"""Fresh latent-path replications and release-aware dynamic prior evidence."""

from dataclasses import replace
from datetime import timedelta
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings, initial_posterior_components
from model_pipelining import predictive_checks as pc
from model_pipelining import prior_influence as pi
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from test_normal_mean import request
import test_predictive_checks as existing_checks


class DynamicSimulationTests(unittest.TestCase):
    def test_joint_prior_trajectory_covariance_matches_random_walk(self):
        model = LocalLevelVariance(initial_mean=2, initial_variance=2, evolution_variance=0.2)
        with patch.object(LocalLevelVariance, "posterior", side_effect=AssertionError("No fitted data")):
            result = pc.simulate_prior_predictive(model, observation_count=4, replications=10000, seed=42)
        values = np.asarray([d.values for d in result.datasets])
        self.assertFalse(result.failures)
        self.assertAlmostEqual(values[:, 0].mean(), 2, delta=0.06)
        self.assertAlmostEqual(values[:, 0].var(), 3.2, delta=0.15)
        self.assertAlmostEqual(np.cov(values[:, 0], values[:, 3])[0, 1], 2.2, delta=0.13)
        self.assertTrue(all(len(item.latent_levels) == 4 for item in result.datasets))

    def test_posterior_replications_sample_initial_state_and_resimulate_innovations(self):
        model = LocalLevelVariance(initial_mean=0, initial_variance=2, evolution_variance=0.2)
        req = request((10, 12, 9))
        posterior = model.posterior(req)
        means, covariances = initial_posterior_components(posterior)
        weights = np.asarray(posterior.weights)
        expected_mean = weights@means
        expected_variance = weights@(covariances+(means-expected_mean)**2)
        result = pc.simulate_posterior_predictive(model, req, replications=8000, seed=2)
        initial = np.asarray([d.initial_level for d in result.datasets])
        levels = np.asarray([d.latent_levels for d in result.datasets])
        self.assertAlmostEqual(initial.mean(), expected_mean, delta=0.08)
        self.assertAlmostEqual(initial.var(), expected_variance, delta=0.15)
        increments = np.diff(np.column_stack((initial, levels)), axis=1)
        self.assertAlmostEqual(increments.mean(), 0, delta=0.02)
        self.assertAlmostEqual(increments.var(), 0.2, delta=0.015)
        # A copied posterior state path would generally move toward observed
        # values; a new process path has zero-mean innovations at every index.
        self.assertTrue(np.all(np.abs(increments.mean(axis=0)) < 0.025))
        self.assertTrue(all(d.mean is None for d in result.datasets))

    def test_gapped_trajectory_increments_use_actual_elapsed_days(self):
        original = request((1, 3))
        last = replace(original.history[-1], timestamp=original.history[-1].timestamp+timedelta(days=2))
        req = replace(original, history=(original.history[0], last), issued_at=last.timestamp,
                      target_time=last.timestamp+timedelta(days=1))
        result = pc.simulate_posterior_predictive(LocalLevelVariance(evolution_variance=0.2), req, replications=6000, seed=3)
        increments = np.asarray([d.latent_levels[1]-d.latent_levels[0] for d in result.datasets])
        self.assertAlmostEqual(increments.var(), 0.6, delta=0.04)

    def test_seed_replay_prefix_and_global_rng_isolation(self):
        model = LocalLevelVariance()
        state = np.random.get_state()
        first = pc.simulate_posterior_predictive(model, request(), replications=5, seed=9)
        second = pc.simulate_posterior_predictive(model, request(), replications=7, seed=9)
        self.assertEqual(first.datasets, second.datasets[:5])
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])


# Reuse only fixture helpers; inheriting the test case would rerun old tests.
class DynamicWorkflowTests(unittest.TestCase):
    setUp = existing_checks.PredictiveWorkflowTests.setUp
    register = existing_checks.PredictiveWorkflowTests.register
    read_json = existing_checks.PredictiveWorkflowTests.read_json

    def test_dynamic_check_replay_future_isolation_and_saved_only_chart(self):
        boundary = self.history[3].timestamp
        spec = replace(self.spec, evaluation=EvaluationPlan(final_start=boundary,
                       final_end=self.history[-1].timestamp+timedelta(days=1)))
        common = {"dataset": "example", "store": self.store, "experiment": spec,
                  "issued_at": boundary-timedelta(seconds=1), "model": LocalLevelVariance(), "replications": 20}
        first = pc.run_predictive_checks(**common, snapshot_id=self.snapshot.snapshot_id, output_directory=self.root/"first")
        changed = self.register((1, 3, 5, 700, 900))
        second = pc.run_predictive_checks(**common, snapshot_id=changed.snapshot_id, output_directory=self.root/"changed")
        self.assertEqual(self.read_json(first, "predictive_checks.json")["status"], "completed")
        self.assertTrue((first/"posterior/latent_levels.csv").is_file())
        for path in first.rglob("*"):
            if path.is_file() and path.name != "predictive_checks.json":
                self.assertEqual(path.read_bytes(), (second/path.relative_to(first)).read_bytes())
        with patch.object(LocalLevelVariance, "posterior", side_effect=AssertionError("No refit")):
            self.assertTrue(all(path.is_file() for path in pc.plot_predictive_checks(first)))

    def test_failed_integration_retains_prior_and_failure_details(self):
        model = LocalLevelVariance(quadrature=QuadratureSettings(minimum_nodes=8, maximum_nodes=16, maximum_expansions=1))
        output = pc.run_predictive_checks(dataset="example", model=model, snapshot_id=self.snapshot.snapshot_id,
            store=self.store, experiment=self.spec, issued_at=self.history[-1].timestamp,
            replications=10, output_directory=self.root/"failed")
        self.assertEqual(self.read_json(output, "predictive_checks.json")["status"], "failed")
        posterior = self.read_json(output, "posterior/summary.json")
        self.assertFalse(posterior["inference_failure"]["converged"])
        self.assertEqual(self.read_json(output, "prior/summary.json")["status"], "completed")

    def test_dynamic_prior_comparison_moments_intervals_and_integration_evidence(self):
        reference = LocalLevelVariance()
        output = pi.run_prior_influence(dataset="example", reference=reference,
            scenarios={"strong-initial": replace(reference, initial_mean=10, initial_variance=0.1)},
            snapshot_id=self.snapshot.snapshot_id, store=self.store, experiment=self.spec,
            issued_at=self.history[-1].timestamp, output_directory=self.root/"influence")
        manifest = self.read_json(output, "prior_influence.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["kind"], "local_level_variance_prior_influence")
        final = self.read_json(output, "configuration_summaries.json")[0]["final"]
        self.assertIsNone(final["prior_mean_weight"])
        self.assertIsNone(final["degrees_of_freedom"])
        self.assertEqual(final["state_interval_method"], "equal_tail")
        regions = json.loads(final["predictive_intervals_json"])
        self.assertAlmostEqual(final["predictive_width"], sum(s["upper"]-s["lower"] for s in regions))
        self.assertTrue(self.read_json(output, "integration_checks.json"))
        with patch.object(LocalLevelVariance, "posterior", side_effect=AssertionError("No refit")):
            self.assertTrue(all(path.is_file() for path in pi.plot_prior_influence(output)))

    def test_dynamic_comparisons_reject_process_or_numerical_changes(self):
        reference = LocalLevelVariance()
        for index, alternative in enumerate((replace(reference, evolution_variance=1),
            replace(reference, quadrature=QuadratureSettings(relative_tolerance=1e-5)))):
            with self.assertRaises(ValueError):
                pi.run_prior_influence(dataset="example", reference=reference, scenarios={"different": alternative},
                    snapshot_id=self.snapshot.snapshot_id, store=self.store, experiment=self.spec,
                    issued_at=self.history[-1].timestamp, output_directory=self.root/f"invalid-{index}")
            self.assertFalse((self.root/f"invalid-{index}").exists())

    def test_heavy_dynamic_prior_does_not_invent_a_predictive_mean(self):
        reference = LocalLevelVariance(prior_shape=0.2, quadrature=QuadratureSettings(maximum_nodes=2048))
        output = pi.run_prior_influence(dataset="example", reference=reference, scenarios={"same": reference},
            snapshot_id=self.snapshot.snapshot_id, store=self.store, experiment=self.spec,
            issued_at=self.history[1].timestamp, output_directory=self.root/"heavy")
        import csv
        with (output/"learning_paths.csv").open() as stream:
            prior = next(csv.DictReader(stream))
        self.assertEqual(prior["predictive_mean"], "")
        self.assertEqual(prior["predictive_moment_status"], "mean_undefined")
        self.assertEqual(prior["status"], "completed")

    def test_dynamic_comparison_replay_and_future_isolation(self):
        reference = LocalLevelVariance()
        common = {"dataset": "example", "reference": reference, "scenarios": {"changed": replace(reference, prior_scale=4)},
                  "store": self.store, "experiment": self.spec, "issued_at": self.history[2].timestamp}
        first = pi.run_prior_influence(**common, snapshot_id=self.snapshot.snapshot_id, output_directory=self.root/"first")
        changed = self.register((1, 3, 5, 700, 900))
        second = pi.run_prior_influence(**common, snapshot_id=changed.snapshot_id, output_directory=self.root/"changed")
        for path in first.iterdir():
            if path.name != "prior_influence.json":
                self.assertEqual(path.read_bytes(), (second/path.name).read_bytes())

    def test_dynamic_workflow_clis(self):
        experiment = self.root/"experiment.json"
        experiment.write_text(json.dumps(self.spec.to_dict()))
        config = self.root/"priors.json"
        config.write_text(json.dumps({"reference": {}, "scenarios": {"shifted": {"initial_mean": 10}}}))
        base = ["--dataset", "example", "--snapshot-id", self.snapshot.snapshot_id, "--store", str(self.store),
                "--experiment", str(experiment), "--as-of", self.history[-1].timestamp.isoformat(),
                "--model", "local-level-variance"]
        for name, script, extra in (("checks", "scripts/check_normal_mean_variance.py", ["--replications", "10"]),
                                    ("priors", "scripts/compare_priors.py", ["--config", str(config)])):
            result = subprocess.run([sys.executable, script, *base, *extra, "--output-directory", str(self.root/name)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()
