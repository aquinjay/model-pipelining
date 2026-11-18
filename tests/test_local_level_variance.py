"""Independent Gaussian likelihood, scalar integration and marginal prediction."""

from dataclasses import replace
from datetime import timedelta
import json
import math
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import invgamma, multivariate_normal, t

from model_pipelining.local_level import LocalLevel
from model_pipelining.local_level_variance import (
    LocalLevelVariance, NumericalIntegrationError, QuadratureSettings, conditional_filter, initial_posterior_components,
)
from model_pipelining.mixtures import NormalMixture
from model_pipelining.reporting import plot_run
from test_normal_mean import FitPipelineFixtures, request


class MixtureTests(unittest.TestCase):
    def test_normal_limit_quantiles_and_total_variance(self):
        distribution = NormalMixture((1,), (2,), (4,))
        self.assertEqual(distribution.mean, 2)
        self.assertEqual(distribution.variance, 4)
        for p in (0.025, 0.5, 0.9):
            self.assertAlmostEqual(distribution.cdf(distribution.ppf(p)), p)
        mixed = NormalMixture((0.25, 0.75), (-2, 2), (1, 4))
        self.assertEqual(mixed.mean, 1)
        self.assertAlmostEqual(mixed.variance, 6.25)
        large = NormalMixture((1,), (0,), (1e308,))
        self.assertGreater(large.pdf(0), 0)
        self.assertAlmostEqual(large.pdf(0)*1e154, 1/math.sqrt(2*math.pi))

    def test_disconnected_hpd_preserves_gap_and_mass(self):
        mixture = NormalMixture((0.5, 0.5), (-4, 4), (0.25, 0.25))
        region, evidence = mixture.highest_density(0.95)
        self.assertEqual(len(region.segments), 2)
        self.assertFalse(any(item.lower <= 0 <= item.upper for item in region.segments))
        mass = sum(quad(mixture.pdf, item.lower, item.upper)[0] for item in region.segments)
        self.assertAlmostEqual(mass, 0.95, places=7)
        self.assertTrue(evidence["resolution_verified"])
        for segment in region.segments:
            self.assertAlmostEqual(mixture.pdf(segment.lower), evidence["density_threshold"], places=9)

    def test_asymmetric_mixture_does_not_average_conditional_endpoints(self):
        mixture = NormalMixture((0.7, 0.3), (-2, 3), (0.16, 1))
        region, _ = mixture.highest_density(0.8)
        self.assertEqual(len(region.segments), 2)
        self.assertAlmostEqual(sum(mixture.cdf(s.upper)-mixture.cdf(s.lower) for s in region.segments), 0.8)
        self.assertNotAlmostEqual(mixture.ppf(0.5), mixture.mean, places=1)

    def test_invalid_components_and_probabilities(self):
        for weights, means, variances in (((), (), ()), ((0,), (0,), (1,)), ((1,), (0,), (0,)),
                                           ((1,), (math.nan,), (1,)), ((1, 2), (0,), (1,))):
            with self.assertRaises(ValueError):
                NormalMixture(weights, means, variances)
        mixture = NormalMixture((1,), (0,), (1,))
        for p in (0, 1, True, math.nan):
            with self.assertRaises(ValueError):
                mixture.ppf(p)
        with self.assertRaisesRegex(ValueError, "not representable"):
            NormalMixture((1,), (1e308,), (1,)).ppf(0.1)

    def test_tiny_broad_component_does_not_loosen_bulk_quantile_precision(self):
        mixture = NormalMixture((1-1e-15, 1e-15), (0, 0), (1, 1e40))
        for probability in (0.025, 0.1, 0.9, 0.975):
            value = mixture.ppf(probability)
            self.assertAlmostEqual(mixture.cdf(value), probability, places=8)


class LearnedVarianceTests(unittest.TestCase):
    def setUp(self):
        self.model = LocalLevelVariance(initial_mean=1, initial_variance=2, evolution_variance=0.2)
        self.req = request((1, 3, 2))

    def batch(self, variance, values=(1, 3, 2), gap=1):
        n = len(values)
        times = np.arange(1, n+1, dtype=float)
        covariance = self.model.initial_variance+self.model.evolution_variance*np.minimum.outer(times, times)
        observation = covariance+variance*np.eye(n)
        residual = np.asarray(values)-self.model.initial_mean
        cross = self.model.initial_variance+self.model.evolution_variance*times
        mean = self.model.initial_mean+cross@np.linalg.solve(observation, residual)
        predicted = self.model.initial_variance+self.model.evolution_variance*(n+gap)+variance-cross@np.linalg.solve(observation, cross)
        likelihood = multivariate_normal.logpdf(values, mean=np.full(n, self.model.initial_mean), cov=observation)
        return float(likelihood), float(mean), float(predicted), observation

    def test_conditional_filter_matches_batch_gaussian_and_existing_filter(self):
        for variance in (0.1, 1, 10):
            mean, covariance, loglike = conditional_filter(self.model, self.req.history, [variance])
            expected_likelihood, expected_mean, expected_predictive, _ = self.batch(variance)
            self.assertAlmostEqual(loglike[0], expected_likelihood)
            self.assertAlmostEqual(mean[0], expected_mean)
            self.assertAlmostEqual(covariance[0]+variance+self.model.evolution_variance, expected_predictive)
            old = LocalLevel(initial_mean=1, initial_variance=2, evolution_variance=0.2, observation_variance=variance)
            self.assertAlmostEqual(old.filter_history(self.req.history)[-1].level_mean, mean[0])

    def test_independent_adaptive_integration_matches_variance_and_predictive_cdf(self):
        posterior = self.model.posterior(self.req)
        def integrate(function):
            def integrand(log_v):
                v = math.exp(log_v)
                likelihood, mean, variance, _ = self.batch(v)
                return math.exp(likelihood+invgamma.logpdf(v, a=3, scale=2)+log_v+10)*function(v, mean, variance)
            return quad(integrand, -15, 20, epsabs=1e-9, epsrel=1e-9)[0]
        normalizer = integrate(lambda v, m, q: 1)
        expected_v = integrate(lambda v, m, q: v)/normalizer
        self.assertAlmostEqual(posterior.observation_variance_mean, expected_v, places=6)
        self.assertAlmostEqual(posterior.level_mean, integrate(lambda v, m, q: m)/normalizer, places=7)
        mixture = self.model.predictive_distribution(self.req)
        for p in (0.025, 0.5, 0.9):
            boundary = mixture.ppf(p)
            probability = integrate(lambda v, m, q: ndtr((boundary-m)/math.sqrt(q)))/normalizer
            self.assertAlmostEqual(probability, p, places=6)
        self.assertLessEqual(posterior.evidence["tail_probability_bound"], self.model.quadrature.tail_tolerance)

    def test_known_level_conjugate_limit_and_student_t_forecast(self):
        model = LocalLevelVariance(initial_mean=2, initial_variance=0, evolution_variance=0)
        posterior = model.posterior(self.req)
        a, b = 4.5, 3
        self.assertAlmostEqual(posterior.observation_variance_mean, b/(a-1), places=6)
        for p in (0.025, 0.5, 0.975):
            self.assertAlmostEqual(posterior.ppf(p), invgamma.ppf(p, a, scale=b), places=5)
        forecast = model.forecast(self.req)
        reference = t(df=2*a, loc=2, scale=math.sqrt(b/a))
        for q in forecast.quantiles:
            self.assertAlmostEqual(q.value, reference.ppf(q.probability), places=5)
        for region in forecast.highest_density_regions:
            mass = sum(reference.cdf(s.upper)-reference.cdf(s.lower) for s in region.segments)
            self.assertAlmostEqual(mass, region.level, places=6)

    def test_initial_state_smoothing_matches_independent_batch_conditioning(self):
        posterior = self.model.posterior(self.req)
        means, variances = initial_posterior_components(posterior)
        for index in (0, len(means)//2, len(means)-1):
            _, _, _, observation = self.batch(posterior.variances[index])
            cross = np.full(3, self.model.initial_variance)
            residual = np.array((1, 3, 2))-self.model.initial_mean
            expected_mean = self.model.initial_mean+cross@np.linalg.solve(observation, residual)
            expected_variance = self.model.initial_variance-cross@np.linalg.solve(observation, cross)
            self.assertAlmostEqual(means[index], expected_mean, places=7)
            self.assertAlmostEqual(variances[index], expected_variance, places=7)

    def test_gaps_fractional_issuance_and_long_horizon_use_elapsed_time(self):
        later = replace(self.req.history[-1], timestamp=self.req.history[-1].timestamp+timedelta(days=3))
        req = replace(self.req, history=(self.req.history[0], later), issued_at=later.timestamp+timedelta(hours=6),
                      target_time=later.timestamp+timedelta(days=3))
        posterior = self.model.posterior(req)
        distribution = self.model.predictive_distribution(req)
        self.assertAlmostEqual(distribution.variance, posterior.predictive(3).variance)
        self.assertGreater(distribution.variance, posterior.predictive(1).variance)

    def test_no_retained_state_or_global_random_state_mutation(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        first = self.model.posterior(self.req)
        self.model.posterior(request((100, 200, 300)))
        self.assertEqual(first, self.model.posterior(self.req))
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])

    def test_required_precision_failure_retains_candidate_and_rejects_forecast(self):
        model = replace(self.model, quadrature=QuadratureSettings(minimum_nodes=8, maximum_nodes=16, maximum_expansions=1))
        fitted = model.fit(self.req)
        self.assertEqual(fitted.evidence.checks[0]["status"], "failed")
        self.assertFalse(fitted.evidence.metadata["quadrature"]["converged"])
        with self.assertRaisesRegex(ValueError, "numerical inference"):
            model.forecast(self.req)
        with self.assertRaises(NumericalIntegrationError) as caught:
            model.posterior(self.req)
        self.assertIn("refinement_history", caught.exception.evidence)

    def test_hpd_probability_and_saved_numerical_evidence(self):
        fitted = self.model.fit(self.req)
        self.assertTrue(all(c["status"] == "passed" for c in fitted.evidence.checks))
        mixture = self.model.predictive_distribution(self.req)
        for region in fitted.forecast.highest_density_regions:
            self.assertAlmostEqual(sum(mixture.cdf(s.upper)-mixture.cdf(s.lower) for s in region.segments), region.level)
        json.dumps(fitted.evidence.metadata, allow_nan=False)
        self.assertIn("quadrature_nodes", fitted.evidence.metadata)

    def test_undefined_variance_and_extreme_requested_probability_are_explicit(self):
        model = LocalLevelVariance(initial_mean=2, initial_variance=0, evolution_variance=0, prior_shape=0.1)
        fitted = model.fit(request((2,)))
        self.assertTrue(all(c["status"] == "passed" for c in fitted.evidence.checks), fitted.evidence.metadata)
        self.assertIsNone(fitted.evidence.metadata["predictive_variance"])
        extreme = replace(self.req, quantile_levels=(1e-9,), highest_density_levels=())
        result = self.model.fit(extreme)
        self.assertEqual(result.evidence.checks[1]["status"], "failed")

    def test_invalid_settings_requests_and_configuration_round_trip(self):
        self.assertEqual(LocalLevelVariance.from_configuration(self.model.configuration), self.model)
        for options in ({"prior_shape": 0}, {"evolution_variance": -1}, {"initial_mean": math.nan},
                        {"initial_variance": True}, {"quadrature": {}}):
            with self.assertRaises(ValueError):
                LocalLevelVariance(**options)
        for options in ({"minimum_nodes": True}, {"minimum_nodes": 4}, {"tail_tolerance": 0}, {"relative_tolerance": math.inf}):
            with self.assertRaises(ValueError):
                QuadratureSettings(**options)
        for req in (replace(self.req, history=()), replace(self.req, history=self.req.history[::-1]),
                    replace(self.req, issued_at=self.req.issued_at.replace(tzinfo=None)),
                    replace(self.req, issued_at=self.req.history[0].timestamp)):
            with self.assertRaises(ValueError):
                self.model.posterior(req)


class LearnedVariancePipelineTests(FitPipelineFixtures, unittest.TestCase):
    def test_pipeline_fit_evidence_replay_future_isolation_and_saved_only_plot(self):
        snapshot = self.register((1, 3, 5))
        model = LocalLevelVariance()
        first, replay = self.run_model("first", snapshot, model), self.run_model("replay", snapshot, model)
        self.assertEqual(self.read_json(first, "manifest.json")["status"], "completed")
        for path in first.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                self.assertEqual(path.read_bytes(), (replay/path.relative_to(first)).read_bytes())
        changed = self.run_model("changed", self.register((1, 3, 500)), model)
        self.assertEqual(self.read_json(first, "fits/000001/fit.json")["metadata"],
                         self.read_json(changed, "fits/000001/fit.json")["metadata"])
        with patch.object(LocalLevelVariance, "fit", side_effect=AssertionError("No refit")):
            self.assertTrue(plot_run(first).is_file())

    def test_failed_quadrature_fails_run_without_dropping_evidence(self):
        model = LocalLevelVariance(quadrature=QuadratureSettings(minimum_nodes=8, maximum_nodes=16, maximum_expansions=1))
        run = self.run_model("failed", model=model)
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertFalse(self.read_json(run, "fits/000001/fit.json")["metadata"]["quadrature"]["converged"])

    def test_cli_learned_and_legacy_modes(self):
        snapshot = self.register((1, 3, 5))
        for name, flags in (("legacy", []), ("learned", ["--learn-observation-variance"])):
            output = self.root/name
            result = subprocess.run([sys.executable, "scripts/run_local_level.py", "--dataset", "example",
                "--snapshot-id", snapshot.snapshot_id, "--store", str(self.store), "--minimum-history", "2",
                "--output-directory", str(output), *flags], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((output/"forecast.png").is_file())
            self.assertTrue((output/"prior_predictive.csv").is_file())


if __name__ == "__main__":
    unittest.main()
