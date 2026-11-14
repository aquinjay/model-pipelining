"""Shared output/scoring invariants, with independently known uncertainty cases."""

import csv
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model_pipelining.backtest import run_backtest
from model_pipelining.local_level import LocalLevel
from model_pipelining.models import (
    Forecast, ForecastRequest, HighestDensityRegion, Interval, Observation, Quantile, TrainingMean,
)
from model_pipelining.reporting import compare_runs, plot_run
from model_pipelining.snapshots import register_snapshot
from model_pipelining.uncertainty import (
    evaluate_highest_density, evaluate_intervals, interval_probabilities, pinball_loss, required_quantiles,
    validate_forecast, validate_interval_levels,
)


class UncertaintyMathTests(unittest.TestCase):
    def test_local_level_forecasts_observation_variance_not_just_latent_variance(self):
        model = LocalLevel(initial_mean=0, initial_variance=1, observation_variance=1, evolution_variance=1)
        history = tuple(Observation(datetime(2020, 1, index, tzinfo=UTC), value)
                        for index, value in enumerate((2, 4), start=1))
        request = ForecastRequest(history, history[-1].timestamp,
                                  history[-1].timestamp + timedelta(days=1), (0.025, 0.975), (0.8, 0.95))
        distribution = model.predictive_distribution(request)
        # The independently calculated posterior at day 2 is N(3, 5/8).
        # Tomorrow adds W=1 and observation V=1: variance 21/8, not 5/8.
        self.assertAlmostEqual(distribution.mean, 3)
        self.assertAlmostEqual(distribution.variance, 21 / 8)
        result = model.forecast(request)
        self.assertAlmostEqual(result.mean, 3)
        self.assertAlmostEqual(result.quantiles[0].value, 3 - math.sqrt(21 / 8) * 1.959963984540054)
        self.assertAlmostEqual(result.quantiles[1].value, 3 + math.sqrt(21 / 8) * 1.959963984540054)
        self.assertEqual(result.highest_density_regions[-1].segments,
                         (Interval(result.quantiles[0].value, result.quantiles[1].value),))
        for region in result.highest_density_regions:
            segment = region.segments[0]
            self.assertAlmostEqual(distribution.cdf(segment.upper) - distribution.cdf(segment.lower), region.level)
            self.assertAlmostEqual(distribution.pdf(segment.lower), distribution.pdf(segment.upper))
            self.assertGreater(distribution.pdf(result.mean), distribution.pdf(segment.lower))
        with self.assertRaises(FrozenInstanceError):
            result.quantiles[0].value = 999

    def test_central_interval_requests_and_invalid_levels(self):
        self.assertEqual(validate_interval_levels([0.95, 0.8]), (0.8, 0.95))
        probabilities = required_quantiles((0.8, 0.95))
        for actual, expected in zip(probabilities, (0.025, 0.1, 0.9, 0.975)):
            self.assertAlmostEqual(actual, expected)
        for invalid in ((), (0,), (1,), (-0.1,), (True,), (math.nan,), (math.inf,),
                        ("0.8",), (0.8, 0.8), None, (math.nextafter(1, 0),)):
            with self.subTest(levels=invalid), self.assertRaises(ValueError):
                validate_interval_levels(invalid)

    def test_shared_quantiles_do_not_assume_a_gaussian_shape_or_centered_mean(self):
        # A rare, heavy upper tail can put the mean outside a central interval.
        result = Forecast(100, (Quantile(0.9, 3), Quantile(0.1, 1)))
        normalized = validate_forecast(result, (0.1, 0.9))
        self.assertEqual(normalized.mean, 100)
        self.assertEqual(normalized.quantiles, (Quantile(0.1, 1), Quantile(0.9, 3)))
        self.assertEqual(validate_forecast(2, ()), Forecast(2))

    def test_invalid_quantile_outputs_are_rejected(self):
        probabilities = (0.1, 0.9)
        invalid = (
            2, Forecast(2), Forecast(math.nan),
            Forecast(2, (Quantile(0.1, 1),)),
            Forecast(2, (Quantile(0.1, 1), Quantile(0.1, 3))),
            Forecast(2, (Quantile(0.1, 1), Quantile(0.9, 3), Quantile(0.5, 2))),
            Forecast(2, (Quantile(0.1, 3), Quantile(0.9, 1))),
            Forecast(2, (Quantile(0.1, math.inf), Quantile(0.9, 3))),
            Forecast(2, (Quantile(0.1, True), Quantile(0.9, 3))),
            Forecast(2, (Quantile(False, 1), Quantile(0.9, 3))),
            Forecast(2, [(0.1, 1), (0.9, 3)]),
        )
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_forecast(output, probabilities)
        with self.assertRaises(ValueError):
            validate_forecast(Forecast(2, (Quantile(0.1, 1), Quantile(0.9, 3))), ())

    def test_pinball_loss_has_the_documented_unscaled_values(self):
        self.assertAlmostEqual(pinball_loss(2, 1, 0.9), 0.9)
        self.assertAlmostEqual(pinball_loss(0, 1, 0.9), 0.1)
        self.assertEqual(pinball_loss(1, 1, 0.9), 0)
        self.assertEqual(pinball_loss(4, 2, 0.5), 1)
        with self.assertRaises(ValueError):
            pinball_loss(1e308, -1e308, 0.5)

    def test_interval_score_width_coverage_and_tail_penalties(self):
        lower, upper = interval_probabilities(0.8)
        quantiles = {lower: 1, upper: 3}
        for actual, expected_score, covered in ((2, 2, True), (1, 2, True), (3, 2, True),
                                                 (0, 12, False), (4, 12, False)):
            with self.subTest(actual=actual):
                result = evaluate_intervals(quantiles, actual, (0.8,))[0]
                self.assertEqual(result["width"], 2)
                self.assertEqual(result["covered"], covered)
                self.assertAlmostEqual(result["interval_score"], expected_score)
                self.assertAlmostEqual(2 * (result["lower_pinball_loss"] + result["upper_pinball_loss"]) / 0.2,
                                       expected_score)
        tied = evaluate_intervals({lower: 2, upper: 2}, 2, (0.8,))[0]
        self.assertTrue(tied["covered"])
        self.assertEqual(tied["width"], 0)
        with self.assertRaises(ValueError):
            evaluate_intervals({lower: -1e308, upper: 1e308}, 0, (0.8,))

    def test_disconnected_highest_density_regions_do_not_cover_the_gap(self):
        regions = (HighestDensityRegion(0.8, (Interval(-4, -3), Interval(3, 4))),)
        result = evaluate_highest_density(regions, actual=0)[0]
        self.assertFalse(result["covered"])
        self.assertEqual(result["total_width"], 2)
        self.assertEqual(result["segment_count"], 2)
        self.assertTrue(evaluate_highest_density(regions, actual=3.5)[0]["covered"])
        self.assertNotIn("interval_score", result)

    def test_highest_density_validation_requires_complete_nested_disjoint_regions(self):
        valid = Forecast(0, highest_density_regions=(
            HighestDensityRegion(0.8, (Interval(-4, -3), Interval(3, 4))),
            HighestDensityRegion(0.95, (Interval(-5, -2), Interval(2, 5))),
        ))
        self.assertEqual(validate_forecast(valid, (), (0.8, 0.95)), valid)
        invalid = (
            (), (HighestDensityRegion(0.8, ()),),
            (HighestDensityRegion(0.8, (Interval(3, 1),)),),
            (HighestDensityRegion(0.8, (Interval(1, math.inf),)),),
            (HighestDensityRegion(0.8, (Interval(-4, 0), Interval(-1, 4))),),
            (HighestDensityRegion(0.8, (Interval(3, 4), Interval(-4, -3))),),
            (valid.highest_density_regions[0], valid.highest_density_regions[0]),
            (HighestDensityRegion(0.8, (Interval(-4, 4),)),
             HighestDensityRegion(0.95, (Interval(-5, -2), Interval(2, 5)))),
        )
        for regions in invalid:
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                validate_forecast(Forecast(0, highest_density_regions=regions), (), (0.8, 0.95))


class UncertaintyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"
        self.snapshot = self.register((2, 4, 3, 6, 5))

    def register(self, values):
        source = self.root / "input.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows(((datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat(), value)
                             for index, value in enumerate(values))
        return register_snapshot(source, dataset="example", store=self.store)

    def run_model(self, name="local", model=None, **options):
        return run_backtest(dataset="example", snapshot_id=self.snapshot.snapshot_id, store=self.store,
                            output_directory=self.root / name, minimum_history=2,
                            model=LocalLevel() if model is None else model, **options)

    def read_json(self, run, name):
        return json.loads((run / name).read_text())

    def read_csv(self, run, name):
        with (run / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_requested_quantiles_are_frozen_saved_and_scored_reproducibly(self):
        run = self.run_model()
        manifest = self.read_json(run, "manifest.json")
        self.assertEqual(manifest["format_version"], 2)
        self.assertTrue(manifest["model"]["supports_quantiles"])
        self.assertTrue(manifest["model"]["supports_highest_density"])
        self.assertEqual(manifest["evaluation"]["primary_interval_method"], "highest_density")
        self.assertEqual(manifest["evaluation"]["uncertainty_target"], "future_observation")
        self.assertEqual(manifest["evaluation"]["interval_levels"], [0.8, 0.95])
        self.assertEqual(len(self.read_csv(run, "quantiles.csv")), 12)
        self.assertEqual(len(self.read_csv(run, "interval_evaluation.csv")), 6)
        self.assertEqual(len(self.read_csv(run, "highest_density.csv")), 6)
        self.assertEqual(len(self.read_csv(run, "highest_density_evaluation.csv")), 6)
        scores = self.read_json(run, "scores.json")["uncertainty"]
        self.assertEqual((scores["status"], scores["expected_count"], scores["scored_count"]), ("completed", 3, 3))
        replay = self.run_model("replay")
        for name in ("predictions.csv", "quantiles.csv", "highest_density.csv", "evaluation.csv",
                     "interval_evaluation.csv", "highest_density_evaluation.csv", "scores.json"):
            self.assertEqual((run / name).read_bytes(), (replay / name).read_bytes())

    def test_point_only_and_legacy_numeric_adapters_have_no_invented_uncertainty(self):
        class LegacyMean:
            name = "legacy"
            configuration = {}

            def forecast(self, request):
                return 2

        for name, model in (("baseline", TrainingMean()), ("legacy", LegacyMean())):
            with self.subTest(name=name):
                run = self.run_model(name, model=model)
                scores = self.read_json(run, "scores.json")["uncertainty"]
                self.assertEqual(scores["status"], "not_applicable")
                self.assertEqual(scores["intervals"], [])
                self.assertFalse((run / "quantiles.csv").exists())
                self.assertEqual(self.read_json(run, "manifest.json")["status"], "completed")
                optional = [check for check in self.read_json(run, "checks.json") if not check["required"]]
                self.assertTrue(all(check["status"] == "not_applicable" for check in optional))

    def test_missing_declared_quantiles_fail_an_origin_without_skipping_later_cases(self):
        class MissingQuantile:
            name = "missing-quantile"
            configuration = {}
            supports_quantiles = True

            def forecast(self, request):
                levels = request.quantile_levels[:-1] if len(request.history) == 3 else request.quantile_levels
                return Forecast(2, tuple(Quantile(p, p * 10) for p in levels))

        run = self.run_model(model=MissingQuantile())
        self.assertEqual([row["status"] for row in self.read_csv(run, "predictions.csv")],
                         ["passed", "failed", "passed"])
        quantiles = self.read_csv(run, "quantiles.csv")
        self.assertTrue(all(row["status"] == "failed" and row["value"] == "" for row in quantiles[4:8]))
        scores = self.read_json(run, "scores.json")["uncertainty"]
        self.assertEqual((scores["expected_count"], scores["scored_count"], scores["failed_count"]), (3, 2, 1))
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertTrue(all(row["scored_count"] == 2 for row in scores["intervals"]))

    def test_undeclared_quantiles_and_invalid_capability_are_rejected(self):
        class Undeclared(TrainingMean):
            def forecast(self, request):
                return Forecast(2, (Quantile(0.1, 1), Quantile(0.9, 3)))

        run = self.run_model(model=Undeclared())
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        class BadFlag(TrainingMean):
            supports_quantiles = "yes"

        with self.assertRaisesRegex(ValueError, "boolean"):
            self.run_model("bad-flag", model=BadFlag())
        self.assertFalse((self.root / "bad-flag").exists())

    def test_failed_uncertainty_scoring_is_visible_even_when_point_scoring_succeeds(self):
        class WideInterval(TrainingMean):
            supports_quantiles = True

            def forecast(self, request):
                return Forecast(2, tuple(Quantile(p, -1e308 if p < 0.5 else 1e308)
                                         for p in request.quantile_levels))

        class WideRegion(LocalLevel):
            def forecast(self, request):
                forecast = super().forecast(request)
                return Forecast(forecast.mean, forecast.quantiles, tuple(
                    HighestDensityRegion(level, (Interval(-1e308, 1e308),)) for level in request.highest_density_levels
                ))

        for name, model in (("wide-quantiles", WideInterval()), ("wide-regions", WideRegion())):
            with self.subTest(name=name):
                run = self.run_model(name, model=model)
                scores = self.read_json(run, "scores.json")
                self.assertTrue(scores["complete"])
                self.assertEqual(scores["scored_count"], 3)
                self.assertEqual(scores["uncertainty"]["scored_count"], 0)
                self.assertIsNone(scores["uncertainty"]["intervals"][0]["mean_interval_score"])
                self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")

    def test_future_actuals_do_not_change_earlier_quantiles(self):
        first = self.run_model("first")
        self.snapshot = self.register((2, 4, 300, 600, 500))
        changed = self.run_model("changed")
        self.assertEqual(self.read_csv(first, "quantiles.csv")[:4], self.read_csv(changed, "quantiles.csv")[:4])
        self.assertEqual(self.read_csv(first, "highest_density.csv")[:2], self.read_csv(changed, "highest_density.csv")[:2])

    def test_quantiles_are_closed_before_any_uncertainty_scoring(self):
        from model_pipelining import backtest

        original = backtest.evaluate_intervals
        frozen = []
        def inspect_score(*arguments):
            self.assertTrue((self.root / "local" / "highest_density.csv").exists())
            contents = (self.root / "local" / "quantiles.csv").read_bytes()
            if frozen:
                self.assertEqual(contents, frozen[0])
            frozen.append(contents)
            return original(*arguments)

        with patch.object(backtest, "evaluate_intervals", side_effect=inspect_score):
            run = self.run_model()
        self.assertEqual((run / "quantiles.csv").read_bytes(), frozen[0])

    def test_invalid_interval_configuration_fails_before_run_creation(self):
        with self.assertRaises(ValueError):
            self.run_model(interval_levels=(0.8, 0.8))
        self.assertFalse((self.root / "local").exists())

    def test_reports_read_saved_quantiles_without_data_or_model_calls(self):
        run = self.run_model(interval_levels=(0.9,))
        original = {path.name: path.read_bytes() for path in run.iterdir()}
        shutil.rmtree(self.store)
        with patch.object(LocalLevel, "forecast", side_effect=AssertionError("No model calls")):
            self.assertTrue(plot_run(run).exists())
        self.assertTrue(all((run / name).read_bytes() == contents for name, contents in original.items()))
        other = self.run_without_store_for_bad_interval(run)
        self.assertFalse((other / "forecast.png").exists())

    def run_without_store_for_bad_interval(self, run):
        other = self.root / "report-copy"
        shutil.copytree(run, other, ignore=shutil.ignore_patterns("forecast.png"))
        with self.assertRaisesRegex(ValueError, "not saved"):
            plot_run(other, interval_level=0.8)
        return other

    def test_comparisons_only_compare_uncertainty_at_common_supported_levels(self):
        baseline = self.run_model("baseline", model=TrainingMean())
        local = self.run_model("local")
        same = self.run_model("same", interval_levels=(0.95,))
        mixed = compare_runs(baseline, local, output_directory=self.root / "mixed")
        self.assertEqual(self.read_json(mixed, "comparison.json")["uncertainty_comparison"]["status"], "not_applicable")
        common = compare_runs(local, same, output_directory=self.root / "common")
        comparison = self.read_json(common, "comparison.json")["uncertainty_comparison"]
        self.assertEqual(comparison["status"], "completed")
        self.assertEqual([row["level"] for row in comparison["intervals"]], [0.95])
        self.assertEqual([row["level"] for row in comparison["highest_density_regions"]], [0.95])

    def test_highest_density_only_adapter_and_plot_preserve_disconnected_sets(self):
        class Disconnected(TrainingMean):
            supports_highest_density = True

            def forecast(self, request):
                return Forecast(0, highest_density_regions=tuple(
                    HighestDensityRegion(level, (Interval(-4 - level, -3 + level), Interval(3 - level, 4 + level)))
                    for level in request.highest_density_levels
                ))

        run = self.run_model(model=Disconnected())
        scores = self.read_json(run, "scores.json")["uncertainty"]
        self.assertEqual(scores["status"], "completed")
        self.assertEqual(scores["quantiles"], [])
        self.assertEqual(scores["intervals"], [])
        self.assertEqual(scores["highest_density_regions"][0]["mean_segment_count"], 2)
        self.assertFalse((run / "quantiles.csv").exists())
        from model_pipelining.reporting import _draw_uncertainty
        with patch("model_pipelining.reporting.Figure") as figure:
            axes = figure.return_value.subplots.return_value
            _draw_uncertainty(axes, [1, 2, 3], run, self.read_json(run, "manifest.json"),
                              self.read_csv(run, "evaluation.csv"))
            calls = axes.fill_between.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(first < second for first, second in zip(calls[0].args[2], calls[1].args[1])))
            self.assertIn("highest-density", calls[0].kwargs["label"])
        self.assertTrue(plot_run(run).exists())

    def test_missing_declared_highest_density_output_fails_the_whole_forecast(self):
        class MissingRegion(LocalLevel):
            def forecast(self, request):
                forecast = super().forecast(request)
                if len(request.history) == 3:
                    return Forecast(forecast.mean, forecast.quantiles, forecast.highest_density_regions[:-1])
                return forecast

        run = self.run_model(model=MissingRegion())
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        rows = self.read_csv(run, "highest_density_evaluation.csv")
        self.assertEqual([row["status"] for row in rows], ["passed", "passed", "failed", "failed", "passed", "passed"])

    def test_version_one_point_artifacts_still_render_and_compare(self):
        baseline = self.run_model("baseline", model=TrainingMean())
        manifest = self.read_json(baseline, "manifest.json")
        manifest["format_version"] = 1
        manifest.pop("evaluation")
        manifest["model"].pop("supports_quantiles")
        (baseline / "manifest.json").write_text(json.dumps(manifest))
        scores = self.read_json(baseline, "scores.json")
        scores.pop("uncertainty")
        (baseline / "scores.json").write_text(json.dumps(scores))
        self.assertTrue(plot_run(baseline).exists())
        new = self.run_model("new")
        comparison = compare_runs(baseline, new, output_directory=self.root / "comparison")
        self.assertEqual(self.read_json(comparison, "comparison.json")["uncertainty_comparison"]["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
