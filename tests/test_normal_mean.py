"""Independent conjugacy/integration checks and per-fit evidence behavior."""

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

from model_pipelining import backtest
from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.inference import FitEvidence, FittedForecast, save_fit_evidence
from model_pipelining.models import Forecast, ForecastRequest, Observation
from model_pipelining.normal_mean import NormalMean
from model_pipelining.reporting import compare_runs, plot_fit, plot_run
from model_pipelining.sampling import MCMCSettings
from model_pipelining.snapshots import register_snapshot


def request(values=(1.0, 3.0, 5.0), *, quantiles=(0.025, 0.1, 0.9, 0.975), levels=(0.8, 0.95)):
    start = datetime(2020, 1, 1, tzinfo=UTC)
    history = tuple(Observation(start + timedelta(days=i), value) for i, value in enumerate(values))
    return ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=1), quantiles, levels)


class NormalMeanTests(unittest.TestCase):
    def test_hand_calculated_posterior_and_predictive_variance(self):
        model = NormalMean(prior_mean=-1, prior_variance=4, observation_variance=2)
        posterior = model.posterior(request())
        self.assertAlmostEqual(posterior.mean, 17 / 7)
        self.assertAlmostEqual(posterior.variance, 4 / 7)
        self.assertAlmostEqual(posterior.prior_mean_weight, 1 / 7)
        self.assertAlmostEqual(posterior.predictive_variance, 18 / 7)
        self.assertAlmostEqual(model.predictive_distribution(request()).variance, 18 / 7)

    def test_independent_quadrature_of_prior_times_likelihood_and_predictive_cdf(self):
        from scipy.integrate import quad
        from scipy.special import ndtr
        model = NormalMean(prior_mean=-1, prior_variance=4, observation_variance=2)
        fitted = model.fit(request())
        def density(mu):
            return math.exp(-((mu + 1)**2) / 8 - sum((x - mu)**2 / 4 for x in (1, 3, 5)))
        normalization = quad(density, -math.inf, math.inf)[0]
        expected = quad(lambda mu: mu * density(mu), -math.inf, math.inf)[0] / normalization
        second = quad(lambda mu: (mu - expected)**2 * density(mu), -math.inf, math.inf)[0] / normalization
        self.assertAlmostEqual(fitted.forecast.mean, expected, places=9)
        self.assertAlmostEqual(fitted.evidence.metadata["posterior"]["variance"], second, places=9)
        for quantile in fitted.forecast.quantiles:
            cdf = quad(lambda mu: ndtr((quantile.value - mu) / math.sqrt(2)) * density(mu), -math.inf, math.inf)[0] / normalization
            self.assertAlmostEqual(cdf, quantile.probability, places=8)

    def test_regions_have_exact_mass_equal_boundary_density_and_parameter_uncertainty(self):
        model = NormalMean(prior_variance=4, observation_variance=2)
        forecast = model.forecast(request())
        predictive = model.predictive_distribution(request())
        for region in forecast.highest_density_regions:
            segment, = region.segments
            self.assertAlmostEqual(predictive.cdf(segment.upper) - predictive.cdf(segment.lower), region.level)
            self.assertAlmostEqual(predictive.pdf(segment.lower), predictive.pdf(segment.upper))
        from statistics import NormalDist
        plugin = NormalDist(forecast.mean, math.sqrt(2))
        self.assertGreater(forecast.quantiles[-1].value, plugin.inv_cdf(0.975))

    def test_prior_influence_is_independent_of_an_approval_threshold(self):
        weak = NormalMean(prior_mean=100, prior_variance=1e10)
        strong = NormalMean(prior_mean=100, prior_variance=1e-8)
        first, second = weak.fit(request()), strong.fit(request())
        self.assertAlmostEqual(first.forecast.mean, 3, places=6)
        self.assertAlmostEqual(second.forecast.mean, 100, places=4)
        self.assertTrue(all(item["status"] == "passed" for item in second.evidence.checks if item["required"]))
        weight = strong.posterior(request()).prior_mean_weight
        shifted = replace(strong, prior_mean=101).forecast(request()).mean
        self.assertAlmostEqual(shifted - second.forecast.mean, weight)
        wider = replace(strong, prior_variance=1).posterior(request())
        self.assertNotEqual(wider.variance, strong.posterior(request()).variance)

    def test_no_retained_state_and_future_request_does_not_change_an_earlier_fit(self):
        model = NormalMean()
        first = model.fit(request())
        model.fit(request((1, 3, 5, 1000)))
        self.assertEqual(first, model.fit(request()))

    def test_accepts_gapped_eligible_history_delayed_issuance_and_longer_horizon(self):
        original = request()
        history = (original.history[0], original.history[2])
        delayed = replace(original, history=history, issued_at=original.issued_at + timedelta(hours=6),
                          target_time=original.target_time + timedelta(days=4))
        self.assertTrue(math.isfinite(NormalMean().forecast(delayed).mean))

    def test_invalid_settings_and_history_raise(self):
        for options in ({"prior_mean": True}, {"prior_variance": 0}, {"observation_variance": -1},
                        {"prior_variance": float("nan")}, {"prior_mean": float("inf")}, {"mcmc": {}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NormalMean(**options)
        original = request()
        invalid = (replace(original, history=()), replace(original, history=original.history[::-1]),
                   replace(original, issued_at=original.issued_at.replace(tzinfo=None)),
                   replace(original, target_time=original.issued_at),
                   replace(original, history=(Observation(original.target_time, 1),)),
                   replace(original, history=(original.history[0], original.history[0])),
                   replace(original, history=(replace(original.history[0], value=True),)),
                   replace(original, history=(replace(original.history[0], value=math.nan),)))
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                NormalMean().fit(case)

    def test_extreme_representable_variances_do_not_require_overflowing_precisions(self):
        model = NormalMean(prior_variance=1e300, observation_variance=1e-100)
        self.assertAlmostEqual(model.posterior(request()).variance / (1e-100 / 3), 1)
        self.assertAlmostEqual(model.forecast(request()).mean, 3)
        with self.assertRaises(ValueError):
            NormalMean(prior_variance=1.7e308, observation_variance=1.7e308).fit(request((1,)))


class FitPipelineFixtures:
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "store"

    def register(self, values=(1, 3, 5, 7)):
        source = self.root / f"source-{len(list(self.root.glob('source-*')))}.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value"))
            writer.writerows((item.timestamp.isoformat(), item.value) for item in request(values).history)
        return register_snapshot(source, dataset="example", store=self.store)

    def run_model(self, name, snapshot=None, model=None, experiment=None):
        snapshot = self.register() if snapshot is None else snapshot
        return run_backtest(dataset="example", snapshot_id=snapshot.snapshot_id, store=self.store,
                            model=NormalMean() if model is None else model,
                            experiment=experiment or ExperimentSpec(task=ForecastTask(minimum_history=2)),
                            output_directory=self.root / name)

    def read_json(self, run, filename):
        return json.loads((run / filename).read_text())

    def read_csv(self, run, filename):
        with (run / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))


class FitPipelineTests(FitPipelineFixtures, unittest.TestCase):
    def test_exact_fit_identity_posterior_and_diagnostic_applicability_are_saved(self):
        snapshot = self.register()
        run = self.run_model("run", snapshot)
        rows = self.read_csv(run, "fit_index.csv")
        self.assertEqual(len(rows), 2)
        first = self.read_json(run, rows[0]["fit_directory"] + "/fit.json")
        self.assertEqual(first["identity"]["snapshot_id"], snapshot.snapshot_id)
        self.assertEqual(first["identity"]["history_sha256"], rows[0]["history_sha256"])
        self.assertEqual(first["metadata"]["forecast_method"], "exact_parameter_marginalized_normal")
        self.assertEqual(first["checks"][1]["status"], "not_applicable")
        self.assertFalse((run / rows[0]["fit_directory"] / "draws.csv").exists())
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "completed")
        with self.assertRaisesRegex(ValueError, "no MCMC"):
            plot_fit(run, origin_index=1)

    def test_fit_evidence_and_predictions_are_frozen_before_actual_selection(self):
        original = backtest.select_actuals
        frozen = {}
        def inspect_actuals(*args):
            run = self.root / "run"
            for path in run.rglob("*"):
                if path.is_file() and path.name != "manifest.json":
                    frozen[path] = path.read_bytes()
            self.assertTrue((run / "fit_index.csv").is_file())
            self.assertEqual(len(list((run / "fits").glob("*/fit.json"))), 2)
            self.assertFalse((run / "evaluation.csv").exists())
            return original(*args)
        with patch.object(backtest, "select_actuals", side_effect=inspect_actuals):
            self.run_model("run")
        self.assertTrue(all(path.read_bytes() == contents for path, contents in frozen.items()))

    def test_failed_required_fit_check_is_saved_and_later_cases_continue(self):
        fit = NormalMean.fit
        def fail_one(model, req):
            result = fit(model, req)
            if len(req.history) == 2:
                return replace(result, evidence=replace(result.evidence, checks=(
                    {"name": "missing_diagnostic", "required": True, "status": "failed"},)))
            return result
        with patch.object(NormalMean, "fit", fail_one):
            run = self.run_model("run")
        self.assertEqual([row["status"] for row in self.read_csv(run, "predictions.csv")], ["failed", "passed"])
        self.assertEqual(self.read_json(run, "fits/000001/fit.json")["status"], "failed")
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertEqual(self.read_json(run, "review.json")["status"], "pending")

    def test_missing_or_invalid_fit_evidence_cannot_pass(self):
        for index, checks in enumerate(((), ({"name": "x", "required": False, "status": "passed"},),
                                        ({"name": "x", "required": True, "status": "not_applicable"},))):
            result = FittedForecast(Forecast(1), FitEvidence({}, checks))
            with patch.object(NormalMean, "fit", return_value=result):
                run = self.run_model(f"bad-{index}")
            self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
            self.assertTrue(all(not row["fit_directory"] for row in self.read_csv(run, "fit_index.csv")))
        with patch.object(NormalMean, "fit", return_value=Forecast(1)):
            run = self.run_model("wrong-return")
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")

    def test_nonfinite_evidence_is_rejected_without_partial_publication(self):
        evidence = FitEvidence({"bad": math.nan}, ({"name": "x", "required": True, "status": "passed"},))
        with self.assertRaises(ValueError):
            save_fit_evidence(FittedForecast(Forecast(1), evidence), self.root / "bad", identity={})
        self.assertFalse((self.root / "bad").exists())

    def test_fit_write_errors_propagate_and_leave_running_manifest(self):
        with patch.object(backtest, "save_fit_evidence", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.run_model("run")
        self.assertEqual(self.read_json(self.root / "run", "manifest.json")["status"], "running")
        # Model-side file errors remain per-case failures; persistence is the
        # runner's responsibility and must not accidentally change that boundary.
        with patch.object(NormalMean, "fit", side_effect=OSError("adapter file unavailable")):
            run = self.run_model("model-error")
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "failed")
        self.assertEqual(len(self.read_csv(run, "fit_index.csv")), 2)

    def test_future_changes_leave_earlier_posterior_and_forecasts_unchanged(self):
        first = self.run_model("first", self.register())
        second = self.run_model("second", self.register((1, 3, 500, 700)))
        self.assertEqual(self.read_csv(first, "predictions.csv")[0], self.read_csv(second, "predictions.csv")[0])
        a = self.read_json(first, "fits/000001/fit.json")
        b = self.read_json(second, "fits/000001/fit.json")
        self.assertEqual(a["metadata"], b["metadata"])
        self.assertEqual(a["identity"]["history_sha256"], b["identity"]["history_sha256"])

    def test_replay_preserves_fit_and_forecast_files_and_saved_only_reports(self):
        snapshot = self.register()
        first, second = self.run_model("first", snapshot), self.run_model("second", snapshot)
        for path in first.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        with patch.object(NormalMean, "fit", side_effect=AssertionError("No fitting in reports")), \
             patch.object(backtest, "retrieve_snapshot", side_effect=AssertionError("No input access")):
            self.assertTrue(plot_run(first).is_file())
            self.assertTrue(compare_runs(first, second, output_directory=self.root / "comparison").is_dir())

    def test_release_aware_requests_and_final_roles_work_with_normal_mean(self):
        source = self.root / "released.csv"
        history = request((1, 3, 5, 7, 9)).history
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "value", "released"))
            writer.writerows((x.timestamp.isoformat(), x.value, (x.timestamp + timedelta(hours=6)).isoformat()) for x in history)
        snapshot = register_snapshot(source, dataset="example", store=self.store)
        specification = ExperimentSpec(task=ForecastTask(minimum_history=1, availability_column="released"),
                                       evaluation=EvaluationPlan(role="final", final_start=history[3].timestamp,
                                                                 final_end=history[-1].timestamp + timedelta(days=1)))
        run = self.run_model("final", snapshot, experiment=specification)
        self.assertEqual(self.read_json(run, "manifest.json")["status"], "completed")
        self.assertEqual(self.read_json(run, "fits/000002/fit.json")["identity"]["training_count"], 2)

    def test_cli_exact_and_configured_mcmc_with_explicit_final_and_trace(self):
        snapshot = self.register()
        spec = ExperimentSpec(task=ForecastTask(minimum_history=2),
                              evaluation=EvaluationPlan(role="development", final_start=request().target_time,
                                                        final_end=request().target_time + timedelta(days=1)))
        config = self.root / "experiment.json"
        config.write_text(json.dumps(spec.to_dict()))
        for name, flags in (("exact", []), ("sampled", ["--mcmc-config", "configs/normal_mean_mcmc.json", "--plot-fit-origin", "2"])):
            output = self.root / name
            result = subprocess.run([sys.executable, "scripts/run_normal_mean.py", "--dataset", "example", "--snapshot-id", snapshot.snapshot_id,
                                     "--store", str(self.store), "--experiment", str(config), "--evaluation-role", "final",
                                     "--output-directory", str(output), *flags], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.read_json(output, "manifest.json")["task"]["evaluation_role"], "final")
            self.assertTrue((output / "forecast.png").is_file())
            if flags:
                self.assertTrue((output / "fits/000002/trace.png").is_file())


if __name__ == "__main__":
    unittest.main()
