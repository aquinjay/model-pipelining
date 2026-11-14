"""Independent numerical checks and shared-runner integration for LocalLevel."""

import csv
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import random
from statistics import mean, variance
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy

from model_pipelining.backtest import run_backtest
from model_pipelining.local_level import LocalLevel
from model_pipelining.models import ForecastRequest, Observation, TrainingMean
from model_pipelining.reporting import compare_runs
from model_pipelining.snapshots import register_snapshot


def history(values):
    return tuple(Observation(datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index), float(value))
                 for index, value in enumerate(values))


def request(observations):
    issued_at = observations[-1].timestamp
    return ForecastRequest(observations, issued_at, issued_at + timedelta(days=1))


class LocalLevelTests(unittest.TestCase):
    def test_hand_calculated_two_observation_example(self):
        model = LocalLevel(initial_mean=0, initial_variance=1, observation_variance=1, evolution_variance=1)
        steps = model.filter_history(history((2, 4)))
        # Independent hand calculation: first prior R=2, Q=3, gain=2/3;
        # second R=5/3, Q=8/3, gain=5/8, posterior mean=3, variance=5/8.
        self.assertEqual(steps[0].observation_mean, 0)
        self.assertEqual(steps[0].observation_variance, 3)
        self.assertAlmostEqual(steps[0].level_mean, 4 / 3)
        self.assertAlmostEqual(steps[0].level_variance, 2 / 3)
        self.assertAlmostEqual(steps[1].observation_variance, 8 / 3)
        self.assertAlmostEqual(steps[1].level_mean, 3)
        self.assertAlmostEqual(steps[1].level_variance, 5 / 8)
        self.assertAlmostEqual(model.forecast(request(history((2, 4)))).mean, 3)

    def test_zero_evolution_matches_batch_normal_conjugacy(self):
        model = LocalLevel(initial_mean=2, initial_variance=3, observation_variance=4, evolution_variance=0)
        values = (1, 5, 2, 8, -1)
        final = model.filter_history(history(values))[-1]
        precision = 1 / 3 + len(values) / 4
        self.assertAlmostEqual(final.level_mean, (2 / 3 + sum(values) / 4) / precision)
        self.assertAlmostEqual(final.level_variance, 1 / precision)

    def test_filter_matches_joint_gaussian_conditioning(self):
        # This reference conditions a full multivariate Gaussian in one batch;
        # it does not repeat the scalar forward recursion under test.
        for evolution in (0, 0.2, 2):
            with self.subTest(evolution=evolution):
                model = LocalLevel(initial_mean=-2, initial_variance=3, observation_variance=1.5,
                                   evolution_variance=evolution)
                values = numpy.array((1, -1, 3, 2, 5), dtype=float)
                steps = model.filter_history(history(values))
                for count, step in enumerate(steps, start=1):
                    times = numpy.arange(1, count + 1)
                    covariance = 3.0 + evolution * numpy.minimum.outer(times, times)
                    covariance += 1.5 * numpy.eye(count)
                    cross_covariance = 3 + evolution * times
                    expected_mean = -2 + cross_covariance @ numpy.linalg.solve(covariance, values[:count] + 2)
                    expected_variance = 3 + count * evolution - cross_covariance @ numpy.linalg.solve(
                        covariance, cross_covariance)
                    self.assertAlmostEqual(step.level_mean, expected_mean, places=11)
                    self.assertAlmostEqual(step.level_variance, expected_variance, places=11)

    def test_known_level_with_no_evolution_stays_known(self):
        model = LocalLevel(initial_mean=7, initial_variance=0, evolution_variance=0)
        for step in model.filter_history(history((100, -100, 1))):
            self.assertEqual(step.level_mean, 7)
            self.assertEqual(step.level_variance, 0)

    def test_more_observation_noise_means_less_reaction(self):
        low_noise = LocalLevel(initial_mean=0, initial_variance=1, observation_variance=0.1, evolution_variance=0)
        high_noise = LocalLevel(initial_mean=0, initial_variance=1, observation_variance=100, evolution_variance=0)
        observation = request(history((10,)))
        self.assertGreater(low_noise.forecast(observation).mean, high_noise.forecast(observation).mean)

    def test_forward_trace_and_forecast_have_no_future_dependence_or_retained_state(self):
        model = LocalLevel()
        early = history((1, 2, 3))
        before = model.forecast(request(early))
        full = model.filter_history(history((1, 2, 3, 900, -900)))
        self.assertEqual(model.filter_history(early), full[:3])
        model.forecast(request(history((100, 200, 300))))
        self.assertEqual(model.forecast(request(early)), before)

    def test_configuration_is_complete_and_model_is_immutable(self):
        model = LocalLevel(initial_mean=3, initial_variance=4, observation_variance=2, evolution_variance=0.5)
        copied = LocalLevel(**json.loads(json.dumps(model.configuration)))
        self.assertEqual(model, copied)
        self.assertEqual(set(model.configuration), {"initial_mean", "initial_variance",
                                                  "observation_variance", "evolution_variance"})
        with self.assertRaises(FrozenInstanceError):
            model.initial_mean = 20

    def test_invalid_parameters_are_rejected(self):
        for field in ("initial_mean", "initial_variance", "observation_variance", "evolution_variance"):
            for value in (True, "1", None, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    LocalLevel(**{field: value})
        for settings in ({"initial_variance": -1}, {"observation_variance": 0},
                         {"observation_variance": -1}, {"evolution_variance": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                LocalLevel(**settings)

    def test_invalid_history_and_request_alignment_are_rejected(self):
        model = LocalLevel()
        valid = history((1, 2, 3))
        invalid_histories = ((), valid[::-1], (valid[0], valid[0]), (valid[0], valid[2]),
                             (Observation(datetime(2020, 1, 1), 1),),
                             (Observation(valid[0].timestamp, math.nan),))
        for observations in invalid_histories:
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                model.filter_history(observations)
        invalid_requests = (
            ForecastRequest((), valid[-1].timestamp, valid[-1].timestamp + timedelta(days=1)),
            ForecastRequest(valid, valid[0].timestamp, valid[-1].timestamp + timedelta(days=1)),
            ForecastRequest(valid, valid[-1].timestamp, valid[-1].timestamp + timedelta(days=2)),
            ForecastRequest(valid, valid[-1].timestamp, datetime(2020, 1, 4)),
        )
        for invalid in invalid_requests:
            with self.subTest(request=invalid), self.assertRaises(ValueError):
                model.forecast(invalid)

    def test_nonfinite_filter_arithmetic_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "variance"):
            LocalLevel(initial_variance=1e308, evolution_variance=1e308).filter_history(history((1,)))

    def test_prior_simulation_is_repeatable_and_does_not_use_global_random_state(self):
        model = LocalLevel()
        state = random.getstate()
        first = model.simulate_prior(count=20, seed=123)
        self.assertEqual(model.simulate_prior(count=20, seed=123), first)
        self.assertNotEqual(model.simulate_prior(count=20, seed=124), first)
        self.assertEqual(random.getstate(), state)

    def test_prior_simulation_matches_first_observation_moments(self):
        model = LocalLevel(initial_mean=3, initial_variance=2, evolution_variance=1, observation_variance=4)
        draws = [model.simulate_prior(count=1, seed=seed)[0] for seed in range(4000)]
        self.assertAlmostEqual(mean(draw.level for draw in draws), 3, delta=0.1)
        self.assertAlmostEqual(variance(draw.level for draw in draws), 3, delta=0.2)
        self.assertAlmostEqual(mean(draw.value for draw in draws), 3, delta=0.15)
        self.assertAlmostEqual(variance(draw.value for draw in draws), 7, delta=0.4)

    def test_simulated_state_errors_agree_with_posterior_variance(self):
        model = LocalLevel(initial_mean=3, initial_variance=4, evolution_variance=0.2, observation_variance=1)
        normalized_squared_errors = []
        for seed in range(400):
            draws = model.simulate_prior(count=25, seed=seed)
            final = model.filter_history(history(draw.value for draw in draws))[-1]
            normalized_squared_errors.append((final.level_mean - draws[-1].level) ** 2 / final.level_variance)
        # A broad deterministic simulation check, not a proof of calibration or
        # a test that all generated states can be recovered exactly.
        self.assertAlmostEqual(mean(normalized_squared_errors), 1, delta=0.25)

    def test_invalid_simulation_requests_are_rejected(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                LocalLevel().simulate_prior(count=count)
        with self.assertRaises(ValueError):
            LocalLevel().simulate_prior(count=1, seed=True)


class LocalLevelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        source = self.root / "observations.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows((item.timestamp.isoformat(), item.value) for item in history((2, 4, 3, 6, 5)))
        self.snapshot = register_snapshot(source, dataset="example", store=self.store)

    def run_model(self, model, name, **options):
        return run_backtest(dataset="example", snapshot_id=self.snapshot.snapshot_id, store=self.store,
                            minimum_history=options.pop("minimum_history", 2), model=model,
                            output_directory=self.root / name, **options)

    def test_shared_runner_saves_local_level_configuration_and_repeats_forecasts(self):
        model = LocalLevel(initial_mean=0, initial_variance=1, observation_variance=1, evolution_variance=1)
        run = self.run_model(model, "local")
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["model"]["configuration"], model.configuration)
        self.assertEqual(manifest["status"], "completed")
        self.assertIn("local_level.py", manifest["source_sha256"])
        with (run / "predictions.csv").open(newline="") as stream:
            first = next(csv.DictReader(stream))
        self.assertAlmostEqual(float(first["prediction"]), 3)
        replay = self.run_model(LocalLevel(**manifest["model"]["configuration"]), "replay")
        for name in ("predictions.csv", "evaluation.csv", "scores.json"):
            self.assertEqual((run / name).read_bytes(), (replay / name).read_bytes())

    def test_local_level_forecast_ignores_changed_future_actuals(self):
        first = self.run_model(LocalLevel(), "first")
        source = self.root / "changed.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows((item.timestamp.isoformat(), item.value) for item in history((2, 4, 300, 600, 500)))
        self.snapshot = register_snapshot(source, dataset="example", store=self.store)
        changed = self.run_model(LocalLevel(), "changed")
        def first_prediction(path):
            with (path / "predictions.csv").open(newline="") as stream:
                return next(csv.DictReader(stream))
        self.assertEqual(first_prediction(first), first_prediction(changed))

    def test_saved_comparison_uses_common_cases_without_model_calls(self):
        baseline = self.run_model(TrainingMean(), "baseline")
        local = self.run_model(LocalLevel(), "local")
        originals = {path: path.read_bytes() for directory in (baseline, local) for path in directory.iterdir()}
        with patch.object(LocalLevel, "forecast", side_effect=AssertionError("No fitting in reports")):
            result = compare_runs(baseline, local, output_directory=self.root / "comparison")
        summary = json.loads((result / "comparison.json").read_text())
        self.assertEqual(summary["case_count"], 3)
        self.assertEqual([item["model"]["name"] for item in summary["runs"]], ["training-mean", "local-level"])
        self.assertTrue((result / "comparison.png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue(all(path.read_bytes() == original for path, original in originals.items()))
        with self.assertRaises(FileExistsError):
            compare_runs(baseline, local, output_directory=result)

    def test_comparison_rejects_different_tasks_and_changed_actuals(self):
        baseline = self.run_model(TrainingMean(), "baseline")
        different = self.run_model(LocalLevel(), "different", minimum_history=3)
        with self.assertRaisesRegex(ValueError, "must match"):
            compare_runs(baseline, different, output_directory=self.root / "bad")
        local = self.run_model(LocalLevel(), "local")
        contents = (local / "evaluation.csv").read_text()
        (local / "evaluation.csv").write_text(contents.replace("3.0", "300.0", 1))
        with self.assertRaisesRegex(ValueError, "must match"):
            compare_runs(baseline, local, output_directory=self.root / "changed")

    def test_comparison_rejects_failed_or_partial_runs(self):
        class BrokenMean(TrainingMean):
            def forecast(self, request):
                raise RuntimeError("cannot forecast")

        baseline = self.run_model(TrainingMean(), "baseline")
        failed = self.run_model(BrokenMean(), "failed")
        with self.assertRaisesRegex(ValueError, "complete runs"):
            compare_runs(baseline, failed, output_directory=self.root / "bad")


if __name__ == "__main__":
    unittest.main()
