"""Behavioral and numerical checks for the model-independent saved decision layer."""

import csv
from datetime import UTC, date, datetime, timedelta
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from scipy.stats import norm, t

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.backtest import run_backtest
from model_pipelining.decision_value import evaluate_decisions
from model_pipelining.decisions import (
    DecisionSpec,
    compare_decision_scores,
    distribution_event_probability,
    distribution_quantile,
    event_indicator,
    issue_decisions_from_probabilities,
    issue_decisions_from_run,
    plot_decision_scores,
    score_decisions,
)
from model_pipelining.local_level import LocalLevel
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.snapshots import register_snapshot
from scripts.run_decisions import main as cli_main


def _make_daily_snapshot(root: Path, values: list[float], *, dataset: str = "synthetic-decision") -> tuple[Path, str]:
    store = root / "store"
    raw = root / "series.csv"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [{"timestamp": (start + timedelta(days=index)).isoformat(), "value": value}
            for index, value in enumerate(values)]
    write_csv(raw, ("timestamp", "value"), rows)
    snapshot = register_snapshot(raw, dataset=dataset, store=store)
    return store, snapshot.snapshot_id


class DistributionExtractionAndSpecTests(unittest.TestCase):
    def test_event_probability_and_quantile_match_scipy_for_mixture_and_student_t(self):
        # Two-component Gaussian mixture.
        mixture = {"family": "gaussian_mixture", "weights": [0.3, 0.7], "means": [10.0, 20.0], "variances": [4.0, 9.0]}
        threshold = 18.0
        expected_ge = 0.3 * norm.sf(threshold, loc=10.0, scale=2.0) + 0.7 * norm.sf(threshold, loc=20.0, scale=3.0)
        expected_le = 0.3 * norm.cdf(threshold, loc=10.0, scale=2.0) + 0.7 * norm.cdf(threshold, loc=20.0, scale=3.0)
        self.assertAlmostEqual(distribution_event_probability(mixture, threshold=threshold, comparison=">="),
                               expected_ge, places=13)
        self.assertAlmostEqual(distribution_event_probability(mixture, threshold=threshold, comparison="<="),
                               expected_le, places=13)
        q75 = distribution_quantile(mixture, 0.75)
        self.assertAlmostEqual(distribution_event_probability(mixture, threshold=q75, comparison="<="), 0.75, places=8)

        # Extreme upper tail stays accurate without 1 - cdf cancellation.
        single = {"family": "gaussian_mixture", "weights": [1.0], "means": [0.0], "variances": [1.0]}
        tail_prob = distribution_event_probability(single, threshold=7.0, comparison=">=")
        self.assertAlmostEqual(tail_prob, float(norm.sf(7.0)), delta=1e-16)
        self.assertGreater(tail_prob, 0.0)

        # Student-t distribution.
        student = {"family": "student_t", "location": 5.0, "scale": 2.5, "df": 6.0}
        self.assertAlmostEqual(distribution_event_probability(student, threshold=8.0, comparison=">="),
                               float(t.sf(8.0, df=6.0, loc=5.0, scale=2.5)), places=13)
        self.assertAlmostEqual(distribution_quantile(student, 0.8),
                               float(t.ppf(0.8, df=6.0, loc=5.0, scale=2.5)), places=12)

    def test_event_indicator_and_spec_round_trip(self):
        self.assertEqual(event_indicator(10.0, threshold=10.0, comparison=">="), 1)
        self.assertEqual(event_indicator(10.0, threshold=10.0, comparison=">"), 0)
        self.assertEqual(event_indicator(10.0, threshold=10.0, comparison="<="), 1)
        self.assertEqual(event_indicator(10.0, threshold=10.0, comparison="<"), 0)

        binary = DecisionSpec(
            name="heat-threshold",
            policy="binary_cost_loss",
            event_description="Next-day value >= 12.0",
            action_description="Trigger alert",
            target_description="Next-day scalar value",
            horizon_days=1,
            cost_loss_ratio=0.25,
            threshold=12.0,
            comparison=">=",
            calibration_reference={"description": "Predeclared training rate", "base_rate": 0.3},
            fallback="climatology",
        )
        restored = DecisionSpec.from_dict(binary.to_dict())
        self.assertEqual(restored, binary)
        self.assertEqual(restored.sha256, binary.sha256)
        self.assertEqual(binary.resolve_fallback_action(), "act")

        continuous = DecisionSpec(
            name="capacity-plan",
            policy="continuous_asymmetric",
            event_description="Next-day demand level",
            action_description="Provision capacity",
            target_description="Next-day scalar value",
            horizon_days=1,
            under_cost=3.0,
            over_cost=1.0,
            calibration_reference={"description": "Historical 75th percentile", "baseline_action": 12.5},
            fallback="baseline",
        )
        self.assertEqual(DecisionSpec.from_dict(continuous.to_dict()), continuous)
        self.assertEqual(continuous.resolve_fallback_action(), 12.5)

        with self.assertRaisesRegex(ValueError, "cost_loss_ratio"):
            DecisionSpec(
                name="bad", policy="binary_cost_loss", event_description="e", action_description="a",
                target_description="t", horizon_days=1, cost_loss_ratio=1.0,
                calibration_reference={"description": "d", "base_rate": 0.2}, fallback="wait",
            )
        with self.assertRaisesRegex(ValueError, "Unknown DecisionSpec"):
            DecisionSpec.from_dict({**binary.to_dict(), "extra": 1})


class EndToEndBayesianAndNoaaDecisionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # 42-day synthetic series: first 30 history, 12 scored targets.
        values = [10.0 + (i % 5) * 0.8 + (4.0 if i >= 35 else 0.0) for i in range(42)]
        self.store, self.snapshot_id = _make_daily_snapshot(self.root, values)
        self.local_run = run_backtest(
            dataset="synthetic-decision",
            snapshot_id=self.snapshot_id,
            output_directory=self.root / "local-level-run",
            model=LocalLevel(initial_mean=10.0, initial_variance=25.0, observation_variance=1.0, evolution_variance=0.5),
            store=self.store,
        )
        self.student_run = run_backtest(
            dataset="synthetic-decision",
            snapshot_id=self.snapshot_id,
            output_directory=self.root / "student-t-run",
            model=NormalMeanVariance(prior_mean=10.0, prior_mean_precision=1.0, prior_shape=3.0, prior_scale=2.0),
            store=self.store,
        )

    def test_bayesian_binary_issuance_ignores_outcomes_replays_and_separates_mutation(self):
        spec = DecisionSpec(
            name="local-level-threshold-13",
            policy="binary_cost_loss",
            event_description="Next-day value at least 13.0",
            action_description="Act when probability exceeds 0.30",
            target_description="Next-day synthetic value",
            horizon_days=1,
            cost_loss_ratio=0.30,
            threshold=13.0,
            comparison=">=",
            calibration_reference={"description": "Predeclared prior rate", "base_rate": 0.20},
            fallback="wait",
        )
        # Copy forecast-time files only (no evaluation.csv or scores.json) to prove
        # issuance never reads outcomes.
        forecast_only = self.root / "forecast-only-run"
        forecast_only.mkdir()
        for name in ("manifest.json", "predictions.csv", "predictive_distributions.json"):
            shutil.copy2(self.local_run / name, forecast_only / name)

        issued_a = issue_decisions_from_run(
            run_directory=forecast_only, spec=spec, output_directory=self.root / "issued-a",
        )
        issued_b = issue_decisions_from_run(
            run_directory=self.local_run, spec=spec, output_directory=self.root / "issued-b",
        )
        # Exact byte-for-byte replay of frozen recommendations and spec.
        self.assertEqual((issued_a / "recommendations.csv").read_bytes(),
                         (issued_b / "recommendations.csv").read_bytes())
        self.assertEqual((issued_a / "decision_spec.json").read_bytes(),
                         (issued_b / "decision_spec.json").read_bytes())

        scored_orig = score_decisions(
            issued_directory=issued_a,
            outcomes_source=self.local_run,
            output_directory=self.root / "scored-orig",
        )
        orig_rec_bytes = (issued_a / "recommendations.csv").read_bytes()
        orig_summary = json.loads((scored_orig / "decision_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(orig_summary["summary"]["cases"], 12)
        self.assertEqual(orig_summary["summary"]["fallback_count"], 0)
        self.assertEqual(orig_summary["sensitivity"]["status"], "completed")
        self.assertIn(0.30, [row["alpha"] for row in orig_summary["sensitivity"]["curve"]])

        # Verify scored numbers match direct decision_value.evaluate_decisions on the same pairs.
        with (issued_a / "recommendations.csv").open(encoding="utf-8", newline="") as stream:
            rec_rows = list(csv.DictReader(stream))
        with (self.local_run / "evaluation.csv").open(encoding="utf-8", newline="") as stream:
            eval_rows = list(csv.DictReader(stream))
        pairs = [(float(rec["event_probability"]), 1 if float(ev["actual"]) >= 13.0 else 0)
                 for rec, ev in zip(rec_rows, eval_rows)]
        direct = evaluate_decisions(pairs, alpha=0.30, base_rate=0.20)
        self.assertAlmostEqual(orig_summary["summary"]["mean_forecast_expense"], direct["mean_forecast_expense"])
        self.assertAlmostEqual(orig_summary["summary"]["mean_climatology_expense"], direct["mean_climatology_expense"])
        self.assertAlmostEqual(orig_summary["summary"]["relative_value"], direct["relative_value"])

        # Flip the last outcome in a separate outcomes CSV: expense must change while
        # the frozen recommendations remain byte-for-byte untouched.
        mutated_csv = self.root / "mutated-outcomes.csv"
        mutated_rows = [{"target_time": row["target_time"], "actual": row["actual"]} for row in eval_rows]
        mutated_rows[-1]["actual"] = 0.0  # forces event=0 instead of event=1
        write_csv(mutated_csv, ("target_time", "actual"), mutated_rows)

        scored_mutated = score_decisions(
            issued_directory=issued_a,
            outcomes_source=mutated_csv,
            output_directory=self.root / "scored-mutated",
        )
        mutated_summary = json.loads((scored_mutated / "decision_summary.json").read_text(encoding="utf-8"))
        self.assertEqual((issued_a / "recommendations.csv").read_bytes(), orig_rec_bytes)
        self.assertNotEqual(mutated_summary["summary"]["observed_events"], orig_summary["summary"]["observed_events"])
        self.assertNotEqual(mutated_summary["summary"]["mean_perfect_expense"], orig_summary["summary"]["mean_perfect_expense"])

        # Chart generation works from saved files only and refuses overwrite.
        chart = plot_decision_scores(scored_orig)
        self.assertTrue(chart.read_bytes().startswith(b"\x89PNG"))
        with self.assertRaises(FileExistsError):
            plot_decision_scores(scored_orig)

    def test_student_t_continuous_asymmetric_newsvendor_end_to_end(self):
        spec = DecisionSpec(
            name="student-t-asymmetric-3-to-1",
            policy="continuous_asymmetric",
            event_description="Next-day continuous target value",
            action_description="Choose 75th predictive quantile for 3:1 under/over cost",
            target_description="Next-day synthetic value",
            horizon_days=1,
            under_cost=3.0,
            over_cost=1.0,
            calibration_reference={"description": "Predeclared baseline action", "baseline_action": 12.0},
            fallback="baseline",
        )
        issued = issue_decisions_from_run(
            run_directory=self.student_run, spec=spec, output_directory=self.root / "issued-cont",
        )
        scored = score_decisions(
            issued_directory=issued, outcomes_source=self.student_run, output_directory=self.root / "scored-cont",
        )
        summary = json.loads((scored / "decision_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary"]["cases"], 12)
        self.assertAlmostEqual(summary["summary"]["quantile_level"], 0.75)
        self.assertIsNone(summary["sensitivity"])
        self.assertGreaterEqual(summary["expense_distribution"]["top_5pct_expense_share"], 0.0)
        self.assertLessEqual(summary["expense_distribution"]["top_5pct_expense_share"], 1.0)
        # Verify the first action equals the exact 75th percentile of the frozen Student-t law.
        frozen_dists = json.loads((self.student_run / "predictive_distributions.json").read_text(encoding="utf-8"))
        with (issued / "recommendations.csv").open(encoding="utf-8", newline="") as stream:
            first_rec = next(csv.DictReader(stream))
        expected_action = distribution_quantile(frozen_dists[0]["distribution"], 0.75)
        self.assertAlmostEqual(float(first_rec["action"]), expected_action, places=12)

    def test_noaa_saved_probabilities_comparable_records_and_comparison(self):
        noaa_dir = self.root / "noaa-range"
        noaa_dir.mkdir()
        cases = ((0.80, 1, 12.0), (0.45, 1, 10.5), (0.60, 0, 8.0), (0.20, 0, 6.5), (0.10, 0, 5.0))
        same_day_probs = (0.90, 0.60, 0.15, 0.10, 0.05)
        rows = []
        for idx, ((ind_p, ev, actual_range), sd_p) in enumerate(zip(cases, same_day_probs)):
            target_day = date(2026, 1, 1) + timedelta(days=idx)
            issued_dt = datetime(target_day.year, target_day.month, target_day.day, tzinfo=UTC) - timedelta(days=1)
            rows.append({
                "date": target_day.isoformat(),
                "issued_at": issued_dt.isoformat(),
                "training_cutoff": issued_dt.isoformat(),
                "actual_range_celsius": actual_range,
                "event": ev,
                "independent_probability": ind_p,
                "same_day_probability": sd_p,
            })
        write_csv(noaa_dir / "matched-range.csv", tuple(rows[0].keys()), rows)

        # Also create a forecast-only CSV without event/actual columns to prove issuance ignores outcomes.
        forecast_only_dir = self.root / "noaa-forecast-only"
        forecast_only_dir.mkdir()
        write_csv(
            forecast_only_dir / "matched-range.csv",
            ("date", "issued_at", "independent_probability", "same_day_probability"),
            [{k: r[k] for k in ("date", "issued_at", "independent_probability", "same_day_probability")} for r in rows],
        )

        spec = DecisionSpec(
            name="noaa-range-10c",
            policy="binary_cost_loss",
            event_description="Next-day temperature range at least 10 C",
            action_description="Act on high daily range",
            target_description="Central Park next-day TMAX - TMIN",
            horizon_days=1,
            cost_loss_ratio=0.20,
            threshold=10.0,
            comparison=">=",
            calibration_reference={"description": "2025 calibration rate", "base_rate": 0.40},
            fallback="climatology",
        )
        issued_ind = issue_decisions_from_probabilities(
            source_directory=forecast_only_dir, method="independent", spec=spec,
            output_directory=self.root / "noaa-issued-ind",
        )
        issued_sd = issue_decisions_from_probabilities(
            source_directory=noaa_dir, method="same_day", spec=spec,
            output_directory=self.root / "noaa-issued-sd",
        )
        scored_ind = score_decisions(
            issued_directory=issued_ind, outcomes_source=noaa_dir, output_directory=self.root / "noaa-scored-ind",
        )
        scored_sd = score_decisions(
            issued_directory=issued_sd, outcomes_source=noaa_dir, output_directory=self.root / "noaa-scored-sd",
        )

        ind_summary = json.loads((scored_ind / "decision_summary.json").read_text(encoding="utf-8"))
        # Matches hand calculation in test_decision_value.py: mean_forecast=0.12, clim=0.20, perf=0.08, V=2/3.
        self.assertAlmostEqual(ind_summary["summary"]["mean_forecast_expense"], 0.12)
        self.assertAlmostEqual(ind_summary["summary"]["relative_value"], 2.0 / 3.0)

        comparison_dir = compare_decision_scores(
            scored_ind, scored_sd, output_directory=self.root / "noaa-comparison",
        )
        comp = json.loads((comparison_dir / "decision_comparison.json").read_text(encoding="utf-8"))
        # Independent paid 0.20 on case 3 (false alarm) while same_day waited (0.0); other 4 cases tied.
        # Differences (ind - sd) = [0, 0, +0.20, 0, 0] -> mean = +0.04, median = 0.0, top 5% (1 case) share = 1.0.
        pd = comp["paired_differences"]
        self.assertAlmostEqual(pd["mean_expense_difference"], 0.04)
        self.assertAlmostEqual(pd["median_expense_difference"], 0.0)
        self.assertEqual((pd["first_better_count"], pd["second_better_count"], pd["tie_count"]), (0, 1, 4))
        self.assertAlmostEqual(pd["action_agreement_rate"], 0.8)
        self.assertAlmostEqual(pd["top_5pct_abs_difference_share"], 1.0)
        self.assertGreater(pd["mean_difference_naive_se"], 0.0)
        self.assertGreater(pd["mean_difference_hac_se"], 0.0)
        self.assertTrue((comparison_dir / "decision_comparison.png").read_bytes().startswith(b"\x89PNG"))

    def test_fallback_tamper_mismatch_and_cli(self):
        # Inject one failed forecast case into a copy of local_run to verify explicit fallback handling.
        failed_run = self.root / "failed-case-run"
        shutil.copytree(self.local_run, failed_run)
        with (failed_run / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            pred_rows = list(csv.DictReader(stream))
            pred_fields = list(pred_rows[0].keys())
        pred_rows[0]["status"] = "failed"
        pred_rows[0]["prediction"] = ""
        pred_rows[0]["error"] = "ValueError: synthetic failure"
        (failed_run / "predictions.csv").unlink()
        write_csv(failed_run / "predictions.csv", pred_fields, pred_rows)
        dists = json.loads((failed_run / "predictive_distributions.json").read_text(encoding="utf-8"))
        dists[0]["status"] = "failed"
        dists[0]["distribution"] = None
        dists[0]["error"] = "ValueError: synthetic failure"
        write_json(failed_run / "predictive_distributions.json", dists, overwrite=True)

        spec = DecisionSpec(
            name="fallback-check",
            policy="binary_cost_loss",
            event_description="Next-day value >= 13",
            action_description="Act",
            target_description="Next-day value",
            horizon_days=1,
            cost_loss_ratio=0.30,
            threshold=13.0,
            calibration_reference={"description": "Prior rate", "base_rate": 0.50},
            fallback="climatology",
        )
        spec_path = self.root / "spec.json"
        write_json(spec_path, spec.to_dict())

        # Exercise CLI end-to-end: issue -> score -> compare.
        issued_fail = self.root / "cli-issued-fail"
        scored_fail = self.root / "cli-scored-fail"
        self.assertEqual(cli_main([
            "issue", "--run-directory", str(failed_run),
            "--spec", str(spec_path), "--output-directory", str(issued_fail),
        ]), 0)
        self.assertEqual(cli_main([
            "score", "--issued-directory", str(issued_fail),
            "--outcomes", str(failed_run), "--output-directory", str(scored_fail),
        ]), 0)
        fail_summary = json.loads((scored_fail / "decision_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(fail_summary["summary"]["fallback_count"], 1)
        self.assertEqual(fail_summary["sensitivity"]["status"], "omitted_due_to_fallbacks")
        checks = json.loads((scored_fail / "checks.json").read_text(encoding="utf-8"))
        fallback_check = next(item for item in checks if item["name"] == "all_decisions_from_forecasts_without_fallback")
        self.assertEqual(fallback_check["status"], "failed")
        self.assertFalse(fallback_check["required"])
        self.assertTrue((scored_fail / "decision_scores.png").is_file())

        # Compare against a clean student-t run on the identical cohort via CLI.
        issued_clean = self.root / "cli-issued-clean"
        scored_clean = self.root / "cli-scored-clean"
        comp_out = self.root / "cli-comp"
        self.assertEqual(cli_main([
            "issue", "--run-directory", str(self.student_run),
            "--spec", str(spec_path), "--output-directory", str(issued_clean),
        ]), 0)
        self.assertEqual(cli_main([
            "score", "--issued-directory", str(issued_clean),
            "--outcomes", str(self.student_run), "--output-directory", str(scored_clean), "--no-plot",
        ]), 0)
        self.assertFalse((scored_clean / "decision_scores.png").exists())
        self.assertEqual(cli_main([
            "compare", "--first", str(scored_fail), "--second", str(scored_clean),
            "--output-directory", str(comp_out),
        ]), 0)
        self.assertTrue((comp_out / "decision_comparison.json").is_file())

        # Tampering with issued recommendations must be caught by score_decisions.
        tampered = self.root / "tampered-issued"
        shutil.copytree(issued_clean, tampered)
        rec_path = tampered / "recommendations.csv"
        rec_path.write_text(rec_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            score_decisions(issued_directory=tampered, outcomes_source=self.student_run,
                            output_directory=self.root / "tampered-scored")

        # Mismatched DecisionSpec must be rejected by compare_decision_scores.
        other_spec = DecisionSpec(
            name="different-ratio", policy="binary_cost_loss", event_description="e", action_description="a",
            target_description="t", horizon_days=1, cost_loss_ratio=0.50, threshold=13.0,
            calibration_reference={"description": "Prior rate", "base_rate": 0.50}, fallback="wait",
        )
        issued_other = issue_decisions_from_run(
            run_directory=self.student_run, spec=other_spec, output_directory=self.root / "issued-other",
        )
        scored_other = score_decisions(
            issued_directory=issued_other, outcomes_source=self.student_run, output_directory=self.root / "scored-other",
        )
        with self.assertRaisesRegex(ValueError, "identical DecisionSpec"):
            compare_decision_scores(scored_clean, scored_other, output_directory=self.root / "bad-comp")


if __name__ == "__main__":
    unittest.main()
