"""Sampler computation, precision, reference agreement, isolation, and replay."""

from dataclasses import replace
from datetime import timedelta
import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from model_pipelining.normal_mean import NormalMean
from model_pipelining.sampling import MCMCSettings, diagnostic_summary, _chain_seeds
from optional_dependencies import SAMPLER_PACKAGES, optional_module, requires
from test_normal_mean import request, FitPipelineFixtures

emcee = optional_module("emcee")


@requires(*SAMPLER_PACKAGES)
class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NormalMean(mcmc=MCMCSettings())
        cls.result = cls.model.fit(request())

    def test_independent_chains_match_reference_for_parameters_and_prediction(self):
        evidence = self.result.evidence
        self.assertTrue(all(item["status"] == "passed" for item in evidence.checks if item["required"]))
        self.assertEqual(len(evidence.diagnostics), 10)
        for item in evidence.diagnostics:
            self.assertLessEqual(item["absolute_error"], item["reference_tolerance"])
            self.assertLessEqual(item["rhat"], 1.01)
            self.assertGreaterEqual(item["ess_bulk"], 400)
            self.assertGreaterEqual(item["ess_tail"], 400)
        metadata = evidence.metadata["sampler"]
        self.assertTrue(metadata["independent_chains"])
        self.assertEqual(metadata["walkers_per_chain"], 1)
        self.assertEqual(len({item["seed"] for item in metadata["chains"]}), 4)
        self.assertEqual(len(evidence.draws), 12000)
        self.assertEqual(sum(not row["warmup"] for row in evidence.draws), 8000)
        for chain in range(4):
            selected = [row for row in evidence.draws if row["chain"] == chain]
            self.assertEqual([row["draw"] for row in selected if row["warmup"]], list(range(1000)))
            self.assertEqual([row["draw"] for row in selected if not row["warmup"]], list(range(2000)))
            self.assertAlmostEqual(sum(row["accepted"] for row in selected if not row["warmup"]) / 2000,
                                   metadata["chains"][chain]["posterior_acceptance_fraction"])
        self.assertEqual(self.result.forecast, NormalMean().forecast(request()))
        self.assertEqual(evidence.checks[-1]["status"], "not_applicable")

    def test_sampled_target_log_density_is_prior_times_likelihood_up_to_a_constant(self):
        rows = [self.result.evidence.draws[index] for index in (0, 25, 100, 800)]
        def independent_logp(mu):
            return -mu**2 / 200 - sum((value - mu)**2 / 2 for value in (1, 3, 5))
        offset = rows[0]["log_probability"] - independent_logp(rows[0]["mu"])
        for row in rows:
            self.assertAlmostEqual(row["log_probability"] - independent_logp(row["mu"]), offset)

    def test_replay_and_no_global_python_or_numpy_random_state_changes(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        repeated = self.model.fit(request())
        self.assertEqual(self.result, repeated)
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])

    def test_per_fit_seeds_and_draws_are_independent_of_execution_order_and_future_data(self):
        short = request()
        longer = request((1, 3, 5, 1000))
        self.assertNotEqual(_chain_seeds(short, MCMCSettings()), _chain_seeds(longer, MCMCSettings()))
        self.model.fit(longer)
        self.assertEqual(self.result.evidence.draws, self.model.fit(short).evidence.draws)
        later = replace(short, target_time=short.target_time + timedelta(days=1))
        self.assertNotEqual(_chain_seeds(short, MCMCSettings()), _chain_seeds(later, MCMCSettings()))
        changed_seed = replace(self.model, mcmc=replace(self.model.mcmc, seed=1)).fit(short)
        self.assertNotEqual(changed_seed.evidence.draws, self.result.evidence.draws)

    def test_bad_mixing_missing_diagnostics_and_biased_reference_fail(self):
        generator = np.random.default_rng(42)
        good = generator.normal(size=(4, 2000))
        settings = MCMCSettings()
        result = diagnostic_summary(good, name="independent_normal", reference=0, reference_sd=1, settings=settings)
        self.assertEqual(result["status"], "passed")
        bad_values = (np.ones((4, 100)), np.full((4, 100), math.nan), np.ones((1, 100)),
                      good + np.arange(4)[:, None] * 10, good + 5)
        for values in bad_values:
            with self.subTest(shape=values.shape):
                result = diagnostic_summary(values, name="bad", reference=0, reference_sd=1, settings=settings)
                self.assertEqual(result["status"], "failed")
                self.assertIsNotNone(result["error"])
        with patch("arviz.mcse", return_value=math.nan):
            result = diagnostic_summary(good, name="missing", reference=0, reference_sd=1, settings=settings)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["mcse_mean"])
        nonfinite_reference = diagnostic_summary(good, name="overflowed_reference_sd", reference=0,
                                                 reference_sd=math.inf, settings=settings)
        self.assertEqual(nonfinite_reference["status"], "failed")
        self.assertIsNone(nonfinite_reference["reference_sd"])

    def test_poor_proposal_fails_without_changing_exact_forecast_or_dropping_draws(self):
        poor = NormalMean(mcmc=MCMCSettings(chains=2, warmup=0, draws=20, proposal_scale=1e-12))
        result = poor.fit(request())
        self.assertEqual(result.forecast, NormalMean().forecast(request()))
        self.assertEqual(len(result.evidence.draws), 40)
        self.assertIn("failed", [item["status"] for item in result.evidence.checks])
        with self.assertRaisesRegex(ValueError, "audit failed"):
            poor.forecast(request())

    def test_partial_backend_failure_retains_draws_and_continues_independent_chains(self):
        original_sample = emcee.EnsembleSampler.sample
        calls = []
        def fail_one(sampler, *args, **kwargs):
            calls.append(sampler)
            for index, state in enumerate(original_sample(sampler, *args, **kwargs)):
                yield state
                if len(calls) == 2 and index == 4:
                    raise RuntimeError("injected chain failure")
        model = NormalMean(mcmc=MCMCSettings(chains=3, warmup=10, draws=20))
        with patch.object(emcee.EnsembleSampler, "sample", fail_one):
            result = model.fit(request())
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result.evidence.draws), 65)
        self.assertEqual(result.evidence.metadata["sampler"]["failures"][0]["chain"], 1)
        self.assertEqual(result.evidence.checks[1]["status"], "failed")
        self.assertEqual(result.evidence.checks[2]["status"], "failed")
        failed_chain = result.evidence.metadata["sampler"]["chains"][1]
        self.assertEqual(failed_chain["warmup_draws"], 5)
        self.assertAlmostEqual(failed_chain["warmup_acceptance_fraction"],
                               sum(row["accepted"] for row in result.evidence.draws if row["chain"] == 1) / 5)
        with patch("emcee.EnsembleSampler", side_effect=RuntimeError("backend initialization failed")):
            result = model.fit(request())
        self.assertEqual(len(result.evidence.metadata["sampler"]["failures"]), 3)
        self.assertFalse(result.evidence.draws)

    def test_invalid_sampler_settings_are_rejected(self):
        for options in ({"chains": 1}, {"chains": True}, {"draws": 7}, {"warmup": -1}, {"seed": -1},
                        {"seed": 2**32}, {"max_rhat": 1}, {"min_tail_ess": 0}, {"proposal_scale": 0},
                        {"proposal_scale": 1e300}, {"proposal_scale": 1e-300}, {"initial_spread": math.inf},
                        {"reference_mcse_multiplier": math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MCMCSettings(**options)


@requires(*SAMPLER_PACKAGES)
class SamplerPipelineTests(FitPipelineFixtures, unittest.TestCase):
    def test_sampler_replay_persists_all_evidence_and_trace_is_saved_only(self):
        snapshot = self.register((1, 3, 5))
        model = NormalMean(mcmc=MCMCSettings())
        first = self.run_model("first", snapshot, model)
        second = self.run_model("second", snapshot, model)
        self.assertEqual(self.read_json(first, "manifest.json")["status"], "completed")
        for path in first.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        original = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
        from model_pipelining.reporting import plot_fit
        with patch.object(NormalMean, "fit", side_effect=AssertionError("No fitting in reports")), \
             patch("emcee.EnsembleSampler", side_effect=AssertionError("No sampler in reports")):
            chart = plot_fit(first, origin_index=1)
        self.assertTrue(chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))
        with self.assertRaises(FileExistsError):
            plot_fit(first, origin_index=1)

    def test_sampler_failure_saves_draws_fails_cases_and_preserves_pending_review(self):
        model = NormalMean(mcmc=MCMCSettings(chains=2, warmup=0, draws=20, proposal_scale=1e-12))
        run = self.run_model("failed", model=model)
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertTrue(all(row["status"] == "failed" for row in self.read_csv(run, "fit_index.csv")))
        for index in (1, 2):
            self.assertEqual(len(self.read_csv(run, f"fits/{index:06d}/draws.csv")), 40)
        self.assertEqual(self.read_json(run, "review.json")["status"], "pending")


if __name__ == "__main__":
    unittest.main()
