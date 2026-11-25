"""Tests for robust scoring, probabilistic baselines, upgraded comparisons, and decision reviews."""

import csv
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from scipy.stats import t

from model_pipelining.artifacts import write_csv
from model_pipelining.backtest import run_backtest
from model_pipelining.decision_value import (
    binary_murphy_curve,
    brier_decomposition,
    mean_murphy_curve,
    murphy_dominance,
    quantile_murphy_curve,
)
from model_pipelining.decisions import (
    DecisionSpec,
    compare_decision_scores,
    distribution_event_probability,
    distribution_quantile,
    issue_decisions_from_run,
    plot_decision_scores,
    score_decisions,
)
from model_pipelining.distribution_scoring import score_distribution
from model_pipelining.experiment_review import build_experiment_review
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.local_level import LocalLevel
from model_pipelining.models import ForecastRequest, Observation
from model_pipelining.probabilistic_baselines import (
    EmpiricalClimatology,
    EmpiricalResidualPersistence,
    GaussianClimatology,
    ResidualPersistence,
)
from model_pipelining.reporting import compare_runs
from model_pipelining.uncertainty import pinball_loss
from model_pipelining.robust_scoring import (
    newey_west_se,
    summarize_loss_series,
    summarize_paired_differences,
    summarize_point_errors,
)
from model_pipelining.snapshots import register_snapshot
from scripts.run_probabilistic_baseline import main as baseline_cli_main


def _make_snapshot(root: Path, values: list[float], *, dataset: str = "robust-example") -> tuple[Path, str]:
    store = root / "store"
    raw = root / f"{dataset}.csv"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [{"timestamp": (start + timedelta(days=i)).isoformat(), "value": v} for i, v in enumerate(values)]
    write_csv(raw, ("timestamp", "value"), rows)
    snap = register_snapshot(raw, dataset=dataset, store=store)
    return store, snap.snapshot_id


class RobustScoringTests(unittest.TestCase):
    def test_point_error_summary_exposes_bias_and_rmse_outlier_concentration(self):
        # 19 tiny errors (+1.0) and 1 shock error (+20.0):
        # Total abs error = 39 -> worst 5% (1 case) share = 20 / 39 ~ 51.3%.
        # Total sq error = 19 + 400 = 419 -> worst 5% (1 case) share = 400 / 419 ~ 95.5%!
        rows = [{"status": "passed", "error": 1.0} for _ in range(19)]
        rows.append({"status": "passed", "error": 20.0})
        summary = summarize_point_errors(rows)
        self.assertEqual(summary["count"], 20)
        self.assertAlmostEqual(summary["mean_error"], 39.0 / 20.0)
        self.assertEqual(summary["median_error"], 1.0)
        self.assertAlmostEqual(summary["mae"], 39.0 / 20.0)
        self.assertEqual(summary["mdae"], 1.0)
        self.assertAlmostEqual(summary["rmse"], math.sqrt(419.0 / 20.0))
        self.assertEqual(summary["top_5pct_case_count"], 1)
        self.assertAlmostEqual(summary["top_5pct_abs_error_share"], 20.0 / 39.0)
        self.assertAlmostEqual(summary["top_5pct_sq_error_share"], 400.0 / 419.0)
        self.assertGreater(summary["top_5pct_sq_error_share"], summary["top_5pct_abs_error_share"])

    def test_signed_loss_summary_handles_negative_log_density(self):
        # Densities > 1 yield negative log scores; signed mode omits ratio share and reports tail mean.
        signed = summarize_loss_series([-2.0, -1.5, -1.0, -0.5, 10.0], nonnegative=False)
        self.assertNotIn("top_5pct_share", signed)
        self.assertAlmostEqual(signed["mean"], 1.0)
        self.assertAlmostEqual(signed["median"], -1.0)
        self.assertAlmostEqual(signed["mean_minus_median"], 2.0)
        self.assertEqual(signed["top_5pct_mean"], 10.0)

        with self.assertRaisesRegex(ValueError, "negative"):
            summarize_loss_series([-0.1, 1.0], nonnegative=True)

    def test_hac_se_reflects_serial_correlation_and_paired_summary(self):
        # Positively autocorrelated paired differences: HAC SE must exceed naive SE.
        positive_ac = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] * 3
        naive_pos, hac_pos, lag_pos = newey_west_se(positive_ac)
        self.assertGreater(lag_pos, 0)
        self.assertGreater(hac_pos, naive_pos)

        # Alternating (negatively autocorrelated) differences: HAC SE must be below naive SE.
        negative_ac = [1.0, -1.0] * 12
        naive_neg, hac_neg, _ = newey_west_se(negative_ac)
        self.assertLess(hac_neg, naive_neg)

        # Hand-checked paired summary: [-2, -1, 0, +1, +10] -> mean = +1.6, median = 0.0,
        # 2 wins for first, 2 for second, 1 tie; top 5% (1 case) abs share = 10 / 14.
        paired = summarize_paired_differences([-2.0, -1.0, 0.0, 1.0, 10.0])
        self.assertAlmostEqual(paired["mean_difference"], 1.6)
        self.assertEqual(paired["median_difference"], 0.0)
        self.assertEqual((paired["first_better_count"], paired["second_better_count"], paired["tie_count"]), (2, 2, 1))
        self.assertAlmostEqual(paired["top_5pct_abs_difference_share"], 10.0 / 14.0)


class ProbabilisticBaselineTests(unittest.TestCase):
    def test_gaussian_climatology_matches_exact_student_t_and_window(self):
        start = datetime(2025, 1, 1, tzinfo=UTC)
        values = [2.0, 4.0, 6.0, 8.0, 20.0]
        history = tuple(Observation(start + timedelta(days=i), v) for i, v in enumerate(values))
        req = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=3),
                              quantile_levels=(0.1, 0.9), highest_density_levels=(0.8,))
        fc = GaussianClimatology().forecast(req)
        mean_val = sum(values) / 5
        var_val = sum((v - mean_val) ** 2 for v in values) / 4
        expected_scale = math.sqrt(var_val * (1.0 + 1.0 / 5))
        dist = t(df=4, loc=mean_val, scale=expected_scale)
        self.assertAlmostEqual(fc.mean, mean_val)
        self.assertEqual(fc.distribution, {"family": "student_t", "location": mean_val, "scale": expected_scale, "df": 4.0})
        self.assertAlmostEqual(fc.quantiles[0].value, float(dist.ppf(0.1)), places=12)
        self.assertAlmostEqual(fc.highest_density_regions[0].segments[0].upper, float(dist.ppf(0.9)), places=12)

        # Rolling window=3 uses only [6.0, 8.0, 20.0].
        fc_win = GaussianClimatology(window=3).forecast(req)
        self.assertEqual(fc_win.distribution["df"], 2.0)
        self.assertAlmostEqual(fc_win.mean, (6.0 + 8.0 + 20.0) / 3)

        # Constant or too-short history fails visibly.
        const_hist = tuple(Observation(start + timedelta(days=i), 5.0) for i in range(4))
        with self.assertRaisesRegex(ValueError, "positive"):
            GaussianClimatology().forecast(ForecastRequest(const_hist, const_hist[-1].timestamp,
                                                           const_hist[-1].timestamp + timedelta(days=1)))
        with self.assertRaisesRegex(ValueError, "at least 3"):
            GaussianClimatology().forecast(ForecastRequest(history[:2], history[1].timestamp,
                                                           history[1].timestamp + timedelta(days=1)))

    def test_residual_persistence_matches_exact_horizon_differences(self):
        start = datetime(2025, 1, 1, tzinfo=UTC)
        values = [10.0, 12.0, 11.0, 15.0, 14.0]
        history = tuple(Observation(start + timedelta(days=i), v) for i, v in enumerate(values))

        # 1-day lead: diffs = [2, -1, 4, -1] -> m=4, mean_sq=(4+1+16+1)/4 = 5.5
        req1 = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=1),
                               quantile_levels=(0.1, 0.9), highest_density_levels=(0.8,))
        fc1 = ResidualPersistence().forecast(req1)
        self.assertEqual(fc1.mean, 14.0)
        self.assertEqual(fc1.distribution["df"], 4.0)
        self.assertAlmostEqual(fc1.distribution["scale"], math.sqrt(5.5), places=12)

        # 2-day lead: diffs = [11-10=1, 15-12=3, 14-11=3] -> m=3, mean_sq=(1+9+9)/3 = 19/3
        req2 = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=2))
        fc2 = ResidualPersistence().forecast(req2)
        self.assertEqual(fc2.mean, 14.0)
        self.assertEqual(fc2.distribution["df"], 3.0)
        self.assertAlmostEqual(fc2.distribution["scale"], math.sqrt(19.0 / 3.0), places=12)

        # Rolling window=2 on 1-day lead uses only last two diffs [4, -1] -> mean_sq=8.5, df=2.
        fc_win = ResidualPersistence(window=2).forecast(req1)
        self.assertEqual(fc_win.distribution["df"], 2.0)
        self.assertAlmostEqual(fc_win.distribution["scale"], math.sqrt(8.5), places=12)

        const_hist = tuple(Observation(start + timedelta(days=i), 5.0) for i in range(4))
        with self.assertRaisesRegex(ValueError, "positive"):
            ResidualPersistence().forecast(ForecastRequest(const_hist, const_hist[-1].timestamp,
                                                           const_hist[-1].timestamp + timedelta(days=1)))


class UpgradedComparisonAndReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # 50-day series with a shift so LocalLevel, ResidualPersistence, and GaussianClimatology differ clearly.
        values = [10.0 + math.sin(i / 3.0) + (5.0 if i >= 38 else 0.0) for i in range(50)]
        self.store, self.snapshot_id = _make_snapshot(self.root, values)
        self.spec = ExperimentSpec(
            task=ForecastTask(minimum_history=30),
            evaluation=EvaluationPlan(benchmarks=("last-value",)),
        )
        self.local_run = run_backtest(
            dataset="robust-example", snapshot_id=self.snapshot_id, store=self.store,
            experiment=self.spec, model=LocalLevel(initial_mean=10.0, initial_variance=25.0,
                                                   observation_variance=1.0, evolution_variance=0.5),
            output_directory=self.root / "local-run",
        )
        self.persist_run = run_backtest(
            dataset="robust-example", snapshot_id=self.snapshot_id, store=self.store,
            experiment=self.spec, model=ResidualPersistence(),
            output_directory=self.root / "persist-run",
        )

    def test_compare_runs_saves_robust_point_and_paired_distribution_diagnostics(self):
        comp_dir = compare_runs(self.local_run, self.persist_run, output_directory=self.root / "comp")
        comp = json.loads((comp_dir / "comparison.json").read_text(encoding="utf-8"))

        # Point comparison includes bias, MdAE, concentration, and paired HAC SEs.
        pc = comp["point_comparison"]
        self.assertEqual(pc["status"], "completed")
        self.assertEqual(len(pc["runs"]), 2)
        for run_summary in pc["runs"]:
            self.assertIn("mean_error", run_summary)
            self.assertIn("mdae", run_summary)
            self.assertIn("top_5pct_sq_error_share", run_summary)
        self.assertIn("mean_difference_hac_se", pc["paired_absolute_error_differences"])
        self.assertIn("first_win_rate", pc["paired_squared_error_differences"])
        self.assertIn(pc["murphy_comparison"]["dominance_status"],
                      ("first_dominates", "second_dominates", "tied", "crossing"))
        comp_md = (comp_dir / "comparison.md").read_text(encoding="utf-8")
        self.assertIn("Mean Murphy dominance", comp_md)
        self.assertIn("Paired CRPS diff", comp_md)

        # Distribution comparison retains legacy fields and adds robust CRPS/log-score diagnostics.
        dc = comp["distribution_comparison"]
        self.assertEqual(dc["status"], "completed")
        self.assertEqual(len(dc["crps_distributions"]), 2)
        self.assertIn("top_5pct_share", dc["crps_distributions"][0])
        self.assertNotIn("top_5pct_share", dc["log_score_distributions"][0])
        self.assertIn("top_5pct_mean", dc["log_score_distributions"][0])
        self.assertAlmostEqual(
            dc["paired_crps_differences"]["mean_difference"],
            dc["mean_crps_first_minus_second"],
        )
        self.assertGreater(dc["paired_crps_differences"]["mean_difference_hac_se"], 0.0)
        self.assertTrue((comp_dir / "distribution_scores.png").read_bytes().startswith(b"\x89PNG"))

    def test_experiment_review_surfaces_robust_metrics_and_integrates_decision_evaluations(self):
        dec_spec = DecisionSpec(
            name="threshold-13",
            policy="binary_cost_loss",
            event_description="Next-day value >= 13.0",
            action_description="Act when P >= 0.25",
            target_description="Next-day synthetic value",
            horizon_days=1,
            cost_loss_ratio=0.25,
            threshold=13.0,
            calibration_reference={"description": "Predeclared prior rate", "base_rate": 0.20},
            fallback="climatology",
        )
        issued_local = issue_decisions_from_run(
            run_directory=self.local_run, spec=dec_spec, output_directory=self.root / "dec-local-issued",
        )
        scored_local = score_decisions(
            issued_directory=issued_local, outcomes_source=self.local_run,
            output_directory=self.root / "dec-local-scored",
        )

        # Complete review with decision evaluation attached.
        review_dir = build_experiment_review(
            forecast_directory=self.local_run,
            decision_directories=(scored_local,),
            require_decisions=True,
            output_directory=self.root / "review-with-decision",
        )
        report = json.loads((review_dir / "report.json").read_text(encoding="utf-8"))
        markdown = (review_dir / "report.md").read_text(encoding="utf-8")
        self.assertEqual(report["status"], "ready_for_review")
        self.assertEqual(report["decision_evaluation_count"], 1)
        forecast_details = report["sources"][0]["details"]
        self.assertIn("robust_point_summary", forecast_details)
        self.assertIn("robust_distribution_summary", forecast_details)
        self.assertIn("MdAE", markdown)
        self.assertIn("Full-law CRPS", markdown)
        self.assertIn("Highest-density regions", markdown)
        self.assertIn("Mean pinball loss", markdown)
        self.assertIn("Decision `threshold-13`", markdown)

        # Requiring decisions when none are provided marks the review incomplete.
        missing_dir = build_experiment_review(
            forecast_directory=self.local_run,
            require_decisions=True,
            output_directory=self.root / "review-missing-decision",
        )
        missing_report = json.loads((missing_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(missing_report["status"], "incomplete")
        self.assertTrue(any(c["name"] == "required_decisions" and c["status"] == "missing"
                            for c in missing_report["checks"]))

        # Attaching a decision evaluation from a DIFFERENT model must fail validation visibly.
        issued_persist = issue_decisions_from_run(
            run_directory=self.persist_run, spec=dec_spec, output_directory=self.root / "dec-persist-issued",
        )
        scored_persist = score_decisions(
            issued_directory=issued_persist, outcomes_source=self.persist_run,
            output_directory=self.root / "dec-persist-scored",
        )
        mismatch_dir = build_experiment_review(
            forecast_directory=self.local_run,
            decision_directories=(scored_persist,),
            output_directory=self.root / "review-mismatch",
        )
        mismatch_report = json.loads((mismatch_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(mismatch_report["status"], "failed")
        self.assertIn("differs from forecast", mismatch_report["sources"][1]["error"])

        # CLI exercises both scripts/run_probabilistic_baseline.py and scripts/review_experiment.py --decision-directories.
        cli_run_dir = self.root / "cli-clim-run"
        self.assertEqual(baseline_cli_main([
            "--model", "gaussian-climatology", "--dataset", "robust-example",
            "--snapshot-id", self.snapshot_id, "--store", str(self.store),
            "--minimum-history", "30", "--output-directory", str(cli_run_dir),
        ]), 0)
        self.assertTrue((cli_run_dir / "forecast.png").is_file())
        self.assertTrue((cli_run_dir / "predictive_distributions.json").is_file())

        cli_review = self.root / "cli-review"
        proc = subprocess.run([
            sys.executable, "scripts/review_experiment.py",
            "--forecast-directory", str(self.local_run),
            "--decision-directories", str(scored_local),
            "--require-decisions",
            "--output-directory", str(cli_review),
        ], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((cli_review / "report.md").is_file())


class EmpiricalDistributionsAndCalibrationMurphyTests(unittest.TestCase):
    def test_empirical_samples_crps_matches_double_sum_and_exact_discrete_ties(self):
        # Weighted empirical distribution with duplicate atoms and unnormalized weights.
        record = {"family": "empirical_samples", "values": [2.0, 5.0, 2.0, 10.0], "weights": [1.0, 2.0, 1.0, 4.0]}
        # Normalized: P(Y=2)=0.25, P(Y=5)=0.25, P(Y=10)=0.50.
        self.assertAlmostEqual(distribution_event_probability(record, threshold=5.0, comparison="<"), 0.25)
        self.assertAlmostEqual(distribution_event_probability(record, threshold=5.0, comparison="<="), 0.50)
        self.assertAlmostEqual(distribution_event_probability(record, threshold=5.0, comparison=">="), 0.75)
        self.assertAlmostEqual(distribution_event_probability(record, threshold=5.0, comparison=">"), 0.50)
        self.assertEqual(distribution_quantile(record, 0.25), 2.0)
        self.assertEqual(distribution_quantile(record, 0.26), 5.0)
        self.assertEqual(distribution_quantile(record, 0.51), 10.0)

        atoms = [(2.0, 0.25), (5.0, 0.25), (10.0, 0.50)]
        pair_term = 0.5 * sum(wi * wj * abs(xi - xj) for xi, wi in atoms for xj, wj in atoms)
        for actual in (-3.0, 2.0, 3.5, 5.0, 7.2, 10.0, 15.0):
            with self.subTest(actual=actual):
                expected_crps = sum(w * abs(x - actual) for x, w in atoms) - pair_term
                result = score_distribution(record, actual)
                self.assertAlmostEqual(result["crps"], expected_crps, places=12)
                self.assertIsNone(result["log_score"])

    def test_brier_decomposition_identity_and_murphy_integral_identities(self):
        cases = ((0.85, 1), (0.75, 1), (0.65, 0), (0.45, 1), (0.35, 0), (0.15, 0), (0.05, 0), (0.55, 1))
        decomp = brier_decomposition(cases, base_rate=0.40, bins=5)
        # Exact Murphy (1973) + within-bin algebraic identity.
        reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"] + decomp["within_bin_term"]
        self.assertAlmostEqual(decomp["brier_score"], reconstructed, places=13)
        self.assertGreaterEqual(decomp["reliability"], 0.0)
        self.assertGreaterEqual(decomp["resolution"], 0.0)
        self.assertGreaterEqual(decomp["ece"], 0.0)

        # Integrating binary elementary score S_alpha(p, y) over alpha in (0, 1) recovers 0.5 * Brier score.
        # Piecewise-linear exact trapezoid over dense midpoint grid.
        n_grid = 2000
        alphas = [(i + 0.5) / n_grid for i in range(n_grid)]
        curve = binary_murphy_curve(cases, alphas=alphas)
        integral_est = sum(pt["mean_elementary_score"] for pt in curve) / n_grid
        self.assertAlmostEqual(integral_est, 0.5 * decomp["brier_score"], places=4)

        # Integrating quantile elementary score S_{theta, tau}(a, y) over theta recovers mean pinball loss.
        q_cases = ((12.0, 15.0), (10.0, 7.0), (8.0, 8.0), (14.0, 11.0))
        tau = 0.75
        low, high = 5.0, 17.0
        n_q = 2400
        step = (high - low) / n_q
        thetas = [low + (i + 0.5) * step for i in range(n_q)]
        q_curve = quantile_murphy_curve(q_cases, quantile_level=tau, thresholds=thetas)
        q_integral = sum(pt["mean_elementary_score"] for pt in q_curve) * step
        expected_pinball = sum(pinball_loss(y, a, tau) for a, y in q_cases) / len(q_cases)
        self.assertAlmostEqual(q_integral, expected_pinball, places=3)

        # Integrating mean elementary score S_theta(x, y) over theta recovers 0.5 * MSE.
        m_curve = mean_murphy_curve(q_cases, thresholds=thetas)
        m_integral = sum(pt["mean_elementary_score"] for pt in m_curve) * step
        expected_half_mse = 0.5 * sum((a - y) ** 2 for a, y in q_cases) / len(q_cases)
        self.assertAlmostEqual(m_integral, expected_half_mse, places=2)

        # Dominance detection: perfect dominates constant, constant crosses another, identical ties.
        perfect_curve = binary_murphy_curve([(y, y) for _, y in cases], alphas=(0.2, 0.5, 0.8))
        model_curve = binary_murphy_curve(cases, alphas=(0.2, 0.5, 0.8))
        self.assertEqual(murphy_dominance(perfect_curve, model_curve)["dominance_status"], "first_dominates")
        self.assertEqual(murphy_dominance(model_curve, perfect_curve)["dominance_status"], "second_dominates")
        self.assertEqual(murphy_dominance(model_curve, model_curve)["dominance_status"], "tied")
        # Construct crossing curves: A pays a false alarm at alpha=0.2 while B misses at alpha=0.8.
        crossing_a = binary_murphy_curve(((0.9, 1), (0.4, 0)), alphas=(0.2, 0.5, 0.8))
        crossing_b = binary_murphy_curve(((0.6, 1), (0.1, 0)), alphas=(0.2, 0.5, 0.8))
        self.assertEqual(murphy_dominance(crossing_a, crossing_b)["dominance_status"], "crossing")

    def test_empirical_baselines_end_to_end_backtest_comparison_decisions_and_review(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Bimodal/skewed synthetic series (seasonal-like jump + asymmetric drops).
            values = [5.0 + (20.0 if (i % 14) >= 7 else 0.0) - (4.0 if i % 5 == 0 else 0.0) for i in range(45)]
            store, snap_id = _make_snapshot(root, values, dataset="empirical-demo")
            spec = ExperimentSpec(task=ForecastTask(minimum_history=30))

            emp_persist = run_backtest(
                dataset="empirical-demo", snapshot_id=snap_id, store=store,
                experiment=spec, model=EmpiricalResidualPersistence(),
                output_directory=root / "emp-persist",
            )
            emp_clim = run_backtest(
                dataset="empirical-demo", snapshot_id=snap_id, store=store,
                experiment=spec, model=EmpiricalClimatology(window=14),
                output_directory=root / "emp-clim",
            )
            local = run_backtest(
                dataset="empirical-demo", snapshot_id=snap_id, store=store,
                experiment=spec, model=LocalLevel(initial_mean=15.0, initial_variance=50.0,
                                                  observation_variance=4.0, evolution_variance=9.0),
                output_directory=root / "local",
            )

            # Empirical runs have exact CRPS, no invented log_score, and no invented HPD regions.
            ep_scores = json.loads((emp_persist / "scores.json").read_text(encoding="utf-8"))
            self.assertEqual(ep_scores["distribution"]["status"], "completed")
            self.assertIsNotNone(ep_scores["distribution"]["mean_crps"])
            self.assertIsNone(ep_scores["distribution"]["mean_log_score"])

            # compare_runs handles both empirical-vs-continuous and empirical-vs-empirical cleanly!
            comp_mixed = compare_runs(local, emp_persist, output_directory=root / "comp-mixed")
            mixed_json = json.loads((comp_mixed / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(mixed_json["distribution_comparison"]["status"], "completed")
            self.assertIsNotNone(mixed_json["distribution_comparison"]["paired_crps_differences"])
            self.assertIsNone(mixed_json["distribution_comparison"]["paired_log_score_differences"])
            self.assertTrue((comp_mixed / "distribution_scores.png").read_bytes().startswith(b"\x89PNG"))

            comp_emp = compare_runs(emp_persist, emp_clim, output_directory=root / "comp-emp")
            self.assertTrue((comp_emp / "distribution_scores.png").read_bytes().startswith(b"\x89PNG"))

            # Decision issuance, scoring (with calibration + Murphy curve), comparison, and review.
            dec_spec = DecisionSpec(
                name="high-regime", policy="binary_cost_loss",
                event_description="Next-day value >= 18.0", action_description="Act",
                target_description="Next-day value", horizon_days=1,
                cost_loss_ratio=0.30, threshold=18.0,
                calibration_reference={"description": "Prior rate", "base_rate": 0.40},
                fallback="climatology",
            )
            ep_iss = issue_decisions_from_run(run_directory=emp_persist, spec=dec_spec, output_directory=root / "ep-iss")
            ep_sc = score_decisions(issued_directory=ep_iss, outcomes_source=emp_persist, output_directory=root / "ep-sc")
            self.assertTrue(plot_decision_scores(ep_sc).read_bytes().startswith(b"\x89PNG"))

            ll_iss = issue_decisions_from_run(run_directory=local, spec=dec_spec, output_directory=root / "ll-iss")
            ll_sc = score_decisions(issued_directory=ll_iss, outcomes_source=local, output_directory=root / "ll-sc")

            ep_sum = json.loads((ep_sc / "decision_summary.json").read_text(encoding="utf-8"))
            self.assertIn("brier_score", ep_sum["calibration"])
            self.assertEqual(ep_sum["murphy_curve"]["kind"], "binary_cost_loss_regret")

            dec_comp = compare_decision_scores(ll_sc, ep_sc, output_directory=root / "dec-comp")
            dc_json = json.loads((dec_comp / "decision_comparison.json").read_text(encoding="utf-8"))
            self.assertIn(dc_json["murphy_comparison"]["dominance_status"],
                          ("first_dominates", "second_dominates", "tied", "crossing"))
            self.assertTrue((dec_comp / "decision_comparison.png").read_bytes().startswith(b"\x89PNG"))
            dec_comp_md = (dec_comp / "decision_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Murphy dominance", dec_comp_md)
            self.assertIn("Brier decomposition", dec_comp_md)

            rev = build_experiment_review(
                forecast_directory=emp_persist, decision_directories=(ep_sc,),
                require_decisions=True, output_directory=root / "emp-review",
            )
            rev_json = json.loads((rev / "report.json").read_text(encoding="utf-8"))
            rev_md = (rev / "report.md").read_text(encoding="utf-8")
            self.assertEqual(rev_json["status"], "ready_for_review")
            self.assertIn("not applicable (discrete empirical law)", rev_md)
            self.assertIn("Event calibration: Brier", rev_md)


if __name__ == "__main__":
    unittest.main()
