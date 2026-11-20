"""Independent joint-density integration, Student-t forecasts, and sampler evidence."""

import csv
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
from scipy.stats import invgamma, norm, normal_inverse_gamma

from model_pipelining import backtest
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.reporting import compare_runs, plot_fit, plot_run
from model_pipelining.sampling import MCMCSettings
from model_pipelining.snapshots import register_snapshot
from optional_dependencies import SAMPLER_PACKAGES, optional_module, requires
from test_normal_mean import FitPipelineFixtures, request

emcee = optional_module("emcee")


class NormalMeanVarianceTests(unittest.TestCase):
    def test_hand_posterior_and_predictive_moments(self):
        model = NormalMeanVariance(prior_mean_precision=1, prior_shape=3, prior_scale=2)
        posterior = model.posterior(request((1, 3)))
        self.assertAlmostEqual(posterior.mean, 4 / 3)
        self.assertEqual(posterior.mean_precision, 3)
        self.assertEqual(posterior.shape, 4)
        self.assertAlmostEqual(posterior.scale, 13 / 3)
        self.assertAlmostEqual(posterior.prior_mean_weight, 1 / 3)
        self.assertAlmostEqual(posterior.observation_variance_mean, 13 / 9)
        self.assertAlmostEqual(posterior.mean_variance, 13 / 27)
        self.assertAlmostEqual(posterior.predictive_variance, 52 / 27)
        distribution = model.predictive_distribution(request((1, 3)))
        self.assertAlmostEqual(distribution.mean(), posterior.mean)
        self.assertAlmostEqual(distribution.var(), posterior.predictive_variance)
        self.assertNotAlmostEqual(posterior.predictive_scale_squared, posterior.predictive_variance)

    def test_raw_prior_times_likelihood_quadrature(self):
        model = NormalMeanVariance(prior_mean=0, prior_mean_precision=1, prior_shape=3, prior_scale=2)
        req = request((1, 3), quantiles=(0.1, 0.9), levels=(0.8,))
        # Integrate raw prior * likelihood in (mu, log(v)); do not use the
        # adapter's updated parameters or its predictive law in this integrand.
        def integral(function):
            def outer(lv):
                variance = math.exp(lv)
                def inner(mu):
                    logp = (3 * math.log(2) - math.lgamma(3) - 4 * lv - 2 / variance
                            - 0.5 * math.log(2 * math.pi * variance) - mu**2 / (2 * variance)
                            - math.log(2 * math.pi * variance)
                            - ((1 - mu)**2 + (3 - mu)**2) / (2 * variance) + lv)
                    return math.exp(logp) * function(mu, variance)
                return quad(inner, -math.inf, math.inf, epsabs=1e-10)[0]
            return quad(outer, -9, 10, epsabs=1e-9)[0]
        evidence = integral(lambda mu, variance: 1)
        posterior = model.posterior(req)
        self.assertAlmostEqual(integral(lambda mu, variance: mu) / evidence, posterior.mean, places=6)
        self.assertAlmostEqual(integral(lambda mu, variance: variance) / evidence,
                               posterior.observation_variance_mean, places=6)
        for quantile in model.forecast(req).quantiles:
            probability = integral(lambda mu, variance: ndtr((quantile.value - mu) / math.sqrt(variance))) / evidence
            self.assertAlmostEqual(probability, quantile.probability, places=6)

    def test_joint_posterior_density_ratio_matches_raw_likelihood(self):
        model = NormalMeanVariance(prior_mean=1, prior_mean_precision=2, prior_shape=2.5, prior_scale=4)
        posterior = model.posterior(request())
        def ratio(mu, variance):
            raw = normal_inverse_gamma.logpdf(mu, variance, mu=1, lmbda=2, a=2.5, b=4)
            raw += sum(norm.logpdf(y, loc=mu, scale=math.sqrt(variance)) for y in (1, 3, 5))
            updated = normal_inverse_gamma.logpdf(mu, variance, mu=posterior.mean,
                lmbda=posterior.mean_precision, a=posterior.shape, b=posterior.scale)
            return raw - updated
        offset = ratio(0, 1)
        for mu, variance in ((2, 0.5), (4, 3), (-2, 10)):
            self.assertAlmostEqual(ratio(mu, variance), offset, places=10)

    def test_predictive_density_and_hpd_are_integrated_student_t(self):
        model, req = NormalMeanVariance(), request()
        posterior = model.posterior(req)
        distribution = model.predictive_distribution(req)
        # Integrate v after analytically integrating conditional mu; independent
        # inverse-gamma density and Gaussian mixture check the Student-t law.
        for x in (-2, posterior.mean, 8):
            density = quad(lambda v: invgamma.pdf(v, posterior.shape, scale=posterior.scale)
                           * norm.pdf(x, posterior.mean, math.sqrt(v * (1 + 1 / posterior.mean_precision))),
                           0, math.inf)[0]
            self.assertAlmostEqual(density, distribution.pdf(x), places=9)
        for region in model.forecast(req).highest_density_regions:
            segment = region.segments[0]
            mass = quad(distribution.pdf, segment.lower, segment.upper)[0]
            self.assertAlmostEqual(mass, region.level, places=9)
            self.assertAlmostEqual(distribution.pdf(segment.lower), distribution.pdf(segment.upper), places=12)
            self.assertGreater(distribution.pdf(posterior.mean), distribution.pdf(segment.lower))
        plug_in = norm(posterior.mean, math.sqrt(posterior.predictive_variance))
        self.assertNotAlmostEqual(distribution.ppf(0.99), plug_in.ppf(0.99), places=3)

    def test_undefined_variance_preserves_finite_forecasts_and_null_metadata(self):
        model = NormalMeanVariance(prior_shape=0.1)
        req = request((1,))
        posterior = model.posterior(req)
        self.assertIsNone(posterior.observation_variance_mean)
        self.assertIsNone(posterior.mean_variance)
        self.assertIsNone(posterior.predictive_variance)
        result = model.fit(req)
        self.assertEqual(result.evidence.metadata["moment_status"], "variance_undefined_shape_le_one")
        self.assertTrue(math.isfinite(result.forecast.mean))
        self.assertTrue(all(math.isfinite(q.value) for q in result.forecast.quantiles))
        json.dumps(result.evidence.metadata, allow_nan=False)

    def test_prior_influence_is_descriptive_and_updates_variance(self):
        req = request()
        weak = NormalMeanVariance(prior_mean=100, prior_mean_precision=0.01).fit(req)
        strong = NormalMeanVariance(prior_mean=100, prior_mean_precision=1e6).fit(req)
        self.assertGreater(strong.evidence.metadata["posterior"]["prior_mean_weight"], 0.999)
        self.assertGreater(strong.forecast.mean, weak.forecast.mean)
        self.assertGreater(strong.evidence.metadata["predictive_variance"], weak.evidence.metadata["predictive_variance"])
        self.assertTrue(all(c["status"] == "passed" for c in strong.evidence.checks if c["required"]))

    def test_gapped_history_long_horizon_and_no_retained_state(self):
        req = request()
        gapped = replace(req, history=(req.history[0], req.history[2]), target_time=req.target_time + timedelta(days=10))
        model = NormalMeanVariance()
        original = model.fit(gapped)
        model.fit(request((100, 200, 300, 400)))
        self.assertEqual(original, model.fit(gapped))
        self.assertEqual(original.forecast, model.forecast(replace(gapped, target_time=req.target_time)))

    def test_invalid_settings_history_probabilities_and_numerical_failures(self):
        for options in ({"prior_shape": 0}, {"prior_scale": -1}, {"prior_mean_precision": 0},
                        {"prior_shape": True}, {"prior_mean": math.nan}, {"mcmc": {}},
                        {"prior_mean_precision": math.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NormalMeanVariance(**options)
        req = request()
        for invalid in (replace(req, history=()), replace(req, history=req.history[::-1]),
                        replace(req, issued_at=req.issued_at.replace(tzinfo=None)),
                        replace(req, target_time=req.issued_at),
                        replace(req, history=(replace(req.history[0], value=True),)),
                        replace(req, history=(replace(req.history[0], value=math.inf),))):
            with self.assertRaises(ValueError):
                NormalMeanVariance().posterior(invalid)
        for invalid in (replace(req, quantile_levels=(0, 1)), replace(req, highest_density_levels=(1,))):
            with self.assertRaises(ValueError):
                NormalMeanVariance().forecast(invalid)
        with self.assertRaises(ValueError):
            NormalMeanVariance().posterior(request((1e308, -1e308)))
        with self.assertRaisesRegex(ValueError, "Unrepresentable"):
            NormalMeanVariance(prior_mean_precision=5e-324).posterior(request())


@requires(*SAMPLER_PACKAGES)
class JointSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = MCMCSettings(warmup=1500, draws=6000, proposal_scale=1.7)
        cls.model = NormalMeanVariance(mcmc=cls.settings)
        cls.result = cls.model.fit(request())

    def test_joint_audit_agrees_with_parameter_and_predictive_references(self):
        evidence = self.result.evidence
        self.assertTrue(all(c["status"] == "passed" for c in evidence.checks if c["required"]), evidence.diagnostics)
        names = {d["quantity"] for d in evidence.diagnostics}
        self.assertTrue({"precision", "log_variance", "observation_variance", "mu_and_predictive_mean",
                         "centered_predictive_second_moment", "predictive_hpd_mass_0.95"} <= names)
        self.assertEqual(len(evidence.draws), 30000)
        self.assertEqual(self.result.forecast, NormalMeanVariance().forecast(request()))
        self.assertEqual(evidence.metadata["sampler"]["dimensions"], 2)
        for chain in evidence.metadata["sampler"]["chains"]:
            retained = [r for r in evidence.draws if r["chain"] == chain["chain"] and not r["warmup"]]
            self.assertEqual(len(retained), 6000)
            self.assertAlmostEqual(sum(r["accepted"] for r in retained) / 6000, chain["posterior_acceptance_fraction"])

    def test_target_includes_nonconstant_transform_jacobian(self):
        rows = [self.result.evidence.draws[i] for i in (0, 60, 1500, 9000)]
        def raw(row):
            mu, variance = row["mu"], row["observation_variance"]
            return (normal_inverse_gamma.logpdf(mu, variance, mu=0, lmbda=0.01, a=3, b=2)
                    + sum(norm.logpdf(y, mu, math.sqrt(variance)) for y in (1, 3, 5)))
        offset = rows[0]["log_probability"] - rows[0]["log_jacobian"] - raw(rows[0])
        for row in rows:
            self.assertAlmostEqual(row["log_probability"] - row["log_jacobian"] - raw(row), offset, places=9)
        self.assertNotEqual(rows[0]["log_jacobian"], rows[-1]["log_jacobian"])

    def test_replay_order_seed_and_global_random_state_isolation(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        self.model.fit(request((1, 3, 5, 100)))
        self.assertEqual(self.result, self.model.fit(request()))
        self.assertEqual(python_state, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        short = replace(self.model, mcmc=MCMCSettings(chains=2, warmup=0, draws=8))
        self.assertNotEqual(short.fit(request()).evidence.draws,
                            replace(short, mcmc=replace(short.mcmc, seed=1)).fit(request()).evidence.draws)

    def test_stuck_sampler_fails_and_does_not_bypass_audit(self):
        model = NormalMeanVariance(mcmc=MCMCSettings(chains=2, warmup=0, draws=20, proposal_scale=1e-12))
        result = model.fit(request())
        self.assertEqual(result.forecast, NormalMeanVariance().forecast(request()))
        self.assertEqual(len(result.evidence.draws), 40)
        self.assertEqual(result.evidence.checks[2]["status"], "failed")
        with self.assertRaisesRegex(ValueError, "audit failed"):
            model.forecast(request())

    def test_partial_chain_and_initialization_failures_retain_evidence(self):
        original = emcee.EnsembleSampler.sample
        calls = []
        def fail_one(sampler, *args, **kwargs):
            calls.append(sampler)
            for index, state in enumerate(original(sampler, *args, **kwargs)):
                yield state
                if len(calls) == 2 and index == 4:
                    raise RuntimeError("injected joint chain failure")
        model = NormalMeanVariance(mcmc=MCMCSettings(chains=3, warmup=10, draws=20))
        with patch.object(emcee.EnsembleSampler, "sample", fail_one):
            result = model.fit(request())
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result.evidence.draws), 65)
        self.assertEqual(result.evidence.metadata["sampler"]["failures"][0]["chain"], 1)
        self.assertEqual(result.evidence.checks[1]["status"], "failed")
        with patch("emcee.EnsembleSampler", side_effect=RuntimeError("backend unavailable")):
            result = model.fit(request())
        self.assertEqual(len(result.evidence.metadata["sampler"]["failures"]), 3)
        self.assertFalse(result.evidence.draws)

    def test_heavy_tail_audit_uses_existing_moments_and_records_omissions(self):
        # The conditional standardized mean and log-variance/precision retain
        # finite moments even when the predictive distribution has no variance.
        for shape, history in ((0.1, (1,)), (0.6, (1, 2))):
            model = NormalMeanVariance(prior_shape=shape, mcmc=MCMCSettings(chains=2, warmup=20, draws=100))
            result = model.fit(request(history))
            metadata = result.evidence.metadata["sampler"]
            self.assertTrue(metadata["omitted_moment_diagnostics"])
            quantities = {d["quantity"] for d in result.evidence.diagnostics}
            self.assertIn("log_variance", quantities)
            self.assertNotIn("observation_variance", quantities)
            if shape < 0.5:
                self.assertNotIn("mu_and_predictive_mean", quantities)
            json.dumps(result.evidence.metadata, allow_nan=False)


class JointPipelineTests(FitPipelineFixtures, unittest.TestCase):
    def test_exact_fit_freezes_before_scoring_and_replays_with_saved_only_reports(self):
        snapshot = self.register()
        model = NormalMeanVariance()
        original_select = backtest.select_actuals
        frozen = {}
        def inspect(*args):
            directory = self.root / "first"
            for path in directory.rglob("*"):
                if path.is_file() and path.name != "manifest.json":
                    frozen[path.relative_to(directory)] = path.read_bytes()
            self.assertIn("fit_index.csv", {str(p) for p in frozen})
            return original_select(*args)
        with patch.object(backtest, "select_actuals", side_effect=inspect):
            first = self.run_model("first", snapshot, model)
        second = self.run_model("second", snapshot, model)
        for path, content in frozen.items():
            self.assertEqual((first / path).read_bytes(), content)
            self.assertEqual((second / path).read_bytes(), content)
        configuration = self.read_json(first, "manifest.json")["model"]["configuration"]
        self.assertEqual(NormalMeanVariance(**configuration), model)
        with patch.object(NormalMeanVariance, "fit", side_effect=AssertionError("No refit")):
            self.assertTrue(plot_run(first).is_file())
            self.assertTrue(compare_runs(first, second, output_directory=self.root / "comparison").is_dir())

    @requires(*SAMPLER_PACKAGES)
    def test_future_values_cannot_change_earlier_joint_fits_or_chains(self):
        model = NormalMeanVariance(mcmc=MCMCSettings(chains=2, warmup=10, draws=20))
        first = self.run_model("first", self.register(), model)
        second = self.run_model("second", self.register((1, 3, 500, 700)), model)
        for filename in ("fit.json", "draws.csv"):
            if filename == "fit.json":
                a = self.read_json(first, f"fits/000001/{filename}")
                b = self.read_json(second, f"fits/000001/{filename}")
                self.assertEqual(a["metadata"], b["metadata"])
            else:
                self.assertEqual((first / f"fits/000001/{filename}").read_bytes(),
                                 (second / f"fits/000001/{filename}").read_bytes())

    @requires(*SAMPLER_PACKAGES)
    def test_joint_audit_replay_and_saved_trace_without_fitting(self):
        snapshot = self.register((1, 3, 5))
        model = NormalMeanVariance(mcmc=MCMCSettings(warmup=1500, draws=6000, proposal_scale=1.7))
        first, second = self.run_model("first", snapshot, model), self.run_model("second", snapshot, model)
        self.assertEqual(self.read_json(first, "manifest.json")["status"], "completed")
        for path in first.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        manifest = self.read_json(first, "manifest.json")
        self.assertIn("sampling_variance.py", str(manifest))
        config = dict(manifest["model"]["configuration"])
        settings = config.pop("mcmc")
        self.assertEqual(NormalMeanVariance(**config, mcmc=MCMCSettings(**settings)), model)
        with patch.object(NormalMeanVariance, "fit", side_effect=AssertionError("No refit")), \
             patch("emcee.EnsembleSampler", side_effect=AssertionError("No sampling")):
            self.assertTrue(plot_fit(first, origin_index=1).read_bytes().startswith(b"\x89PNG"))
        with self.assertRaises(FileExistsError):
            plot_fit(first, origin_index=1)

    @requires(*SAMPLER_PACKAGES)
    def test_failed_joint_audit_is_saved_and_fails_run(self):
        model = NormalMeanVariance(mcmc=MCMCSettings(chains=2, warmup=0, draws=20, proposal_scale=1e-12))
        run = self.run_model("failed", model=model)
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertEqual(self.read_json(run, "review.json")["status"], "pending")
        self.assertEqual(len(self.read_csv(run, "fit_index.csv")), 2)
        self.assertEqual(len(self.read_csv(run, "fits/000001/draws.csv")), 40)
        self.assertTrue(plot_fit(run, origin_index=1).is_file())

    def test_release_selection_and_final_roles(self):
        source = self.root / "released.csv"
        history = request((1, 3, 5, 7, 9)).history
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value", "released"))
            writer.writerows((x.timestamp.isoformat(), x.value, (x.timestamp + timedelta(hours=6)).isoformat()) for x in history)
        snapshot = register_snapshot(source, dataset="example", store=self.store)
        spec = ExperimentSpec(task=ForecastTask(minimum_history=1, availability_column="released"),
                              evaluation=EvaluationPlan(role="final", final_start=history[3].timestamp,
                                                        final_end=history[-1].timestamp + timedelta(days=1),
                                                        benchmarks=("training-mean", "last-value")))
        run = self.run_model("final", snapshot, NormalMeanVariance(), spec)
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "completed")
        self.assertEqual(self.read_json(run, "fits/000002/fit.json")["identity"]["training_count"], 2)

    @requires(*SAMPLER_PACKAGES)
    def test_cli_exact_audit_and_incompatible_flags(self):
        snapshot = self.register((1, 3, 5))
        for name, flags in (("exact", ["--prior-mean", "1", "--prior-mean-precision", "2", "--prior-shape", "3", "--prior-scale", "4"]),
                            ("audit", ["--mcmc-config", "configs/normal_mean_variance_mcmc.json", "--plot-fit-origin", "1"])):
            command = [sys.executable, "scripts/run_normal_mean.py", "--learn-variance", "--dataset", "example",
                       "--snapshot-id", snapshot.snapshot_id, "--store", str(self.store),
                       "--minimum-history", "2", "--output-directory", str(self.root / name)]
            result = subprocess.run(command + flags, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.read_json(self.root / name, "manifest.json")["model"]["name"], "normal-mean-variance")
            if name == "exact":
                config = self.read_json(self.root / name, "manifest.json")["model"]["configuration"]
                self.assertEqual(config, {"prior_mean": 1, "prior_mean_precision": 2,
                                          "prior_shape": 3, "prior_scale": 4, "mcmc": None})
        for index, flags in enumerate((["--learn-variance", "--observation-variance", "1"], ["--prior-shape", "3"])):
            output = self.root / f"invalid-{index}"
            result = subprocess.run([sys.executable, "scripts/run_normal_mean.py", "--dataset", "example",
                                    "--snapshot-id", snapshot.snapshot_id, "--store", str(self.store),
                                    "--output-directory", str(output), *flags], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
