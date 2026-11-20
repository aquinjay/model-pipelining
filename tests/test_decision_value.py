"""Hand checks for cost-loss rules and the saved-range decision workflow."""

from datetime import date, timedelta
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.decision_value import decide, evaluate_decisions, expense, value_curve
from scripts.study_decision_value import ALPHAS, main, run_study


CASES = ((.8, 1), (.45, 1), (.6, 0), (.2, 0), (.1, 0))


class CostLossTests(unittest.TestCase):
    def test_five_case_hand_calculation(self):
        # At alpha = .2 we act on the first three cases. Two hits and one false alarm
        # cost .2 + .2 + .2. The two correct negatives cost zero, giving .6 / 5 = .12.
        # A calibration rate of .4 triggers always-act, costing .2 per case.
        # Perfect foresight acts on the two events for .4 / 5 = .08 per case.
        # Relative value is (.2 - .12) / (.2 - .08) = 2/3.
        result = evaluate_decisions(CASES, alpha=.2, base_rate=.4)
        self.assertEqual(result["cases"], 5)
        self.assertEqual([result[name] for name in ("hits", "false_alarms", "misses", "correct_negatives")],
                         [2, 1, 0, 2])
        self.assertAlmostEqual(result["mean_forecast_expense"], .12)
        self.assertAlmostEqual(result["mean_climatology_expense"], .2)
        self.assertAlmostEqual(result["mean_perfect_expense"], .08)
        self.assertAlmostEqual(result["relative_value"], 2 / 3)

    def test_calibration_rate_drives_climatology_even_when_evaluation_rate_differs(self):
        # Evaluation prevalence is 2/5 = .4, but calibration prevalence is 1/10 = .1.
        # At alpha = .2, climatology never acts, so it pays for both losses: 2/5 = .4.
        # A forecast paying .12 has value (.4 - .12) / (.4 - .08) = .875.
        result = evaluate_decisions(CASES, alpha=.2, base_rate=.1)
        self.assertAlmostEqual(result["mean_climatology_expense"], .4)
        self.assertAlmostEqual(result["relative_value"], .875)

    def test_expense_and_tie_rule(self):
        self.assertTrue(decide(.6, .2))
        self.assertFalse(decide(.2, .2))
        self.assertEqual([expense(acted, event, .2) for acted, event in
                          ((True, 1), (True, 0), (False, 1), (False, 0))], [.2, .2, 1., 0.])
        result = evaluate_decisions([(.2, 1), (.2, 0)], alpha=.2, base_rate=.2)
        self.assertEqual((result["hits"], result["false_alarms"], result["misses"], result["correct_negatives"]),
                         (0, 0, 1, 1))
        self.assertEqual(result["mean_climatology_expense"], .5)

    def test_perfect_and_climatology_values_across_alphas(self):
        # No-cost correct negatives and C-cost hits make the forecast's expense
        # identical to the perfect reference. A constant .4 forecast takes the same
        # action as the reference climatology for every case, including the tie.
        alphas = (.6, .2, .4)
        perfect = ((event, event) for _, event in CASES)
        climatology = ((.4, event) for _, event in CASES)
        self.assertEqual([row["alpha"] for row in value_curve(perfect, alphas=iter(alphas), base_rate=.4)], list(alphas))
        self.assertEqual([row["relative_value"] for row in value_curve(
            [(event, event) for _, event in CASES], alphas=alphas, base_rate=.4)], [1., 1., 1.])
        self.assertEqual([row["relative_value"] for row in value_curve(
            climatology, alphas=alphas, base_rate=.4)], [0., 0., 0.])

    def test_negative_and_undefined_values(self):
        # Always acting on a 1-event, 1-nonevent table costs .75 per case.
        # Never-act climatology costs .5, and the perfect forecast costs .375.
        # Value is (.5 - .75) / (.5 - .375) = -2.
        negative = evaluate_decisions([(1, 1), (1, 0)], alpha=.75, base_rate=.5)
        self.assertEqual(negative["relative_value"], -2.)
        for cases, rate in (([(.7, 0), (.3, 0)], .1), ([(.7, 1), (.3, 1)], .9)):
            with self.subTest(cases=cases):
                result = evaluate_decisions(cases, alpha=.2, base_rate=rate)
                self.assertEqual(result["mean_climatology_expense"], result["mean_perfect_expense"])
                self.assertIsNone(result["relative_value"])
                self.assertIn("equal mean expense", result["relative_value_reason"])

    def test_invalid_inputs_raise_value_error(self):
        for alpha in (0, 1, -1, 1.2, float("nan"), float("inf"), ".2", True):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "alpha"):
                decide(.2, alpha)
        for value in (-.1, 1.1, float("nan"), float("inf"), None, "0.2", False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "probability"):
                    decide(value, .2)
                with self.assertRaisesRegex(ValueError, "base_rate"):
                    evaluate_decisions(CASES, alpha=.2, base_rate=value)
        for event in (-1, 2, .5, "1", None):
            with self.subTest(event=event):
                with self.assertRaisesRegex(ValueError, "event"):
                    expense(True, event, .2)
                with self.assertRaisesRegex(ValueError, "event"):
                    evaluate_decisions([(.7, event)], alpha=.2, base_rate=.4)
        for cases in ([], None, [(.2,)], [(.2, 1, 7)]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                evaluate_decisions(cases, alpha=.2, base_rate=.4)
        with self.assertRaises(ValueError):
            expense(1, 0, .2)
        with self.assertRaises(ValueError):
            value_curve(CASES, alphas=[], base_rate=.4)


class SavedDecisionStudyTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "range"
        self.source.mkdir()
        self.output = self.root / "decisions"
        self.summary = {
            "evaluation_start": "2026-01-01",
            "evaluation_end": "2026-01-05",
            "cases": 5,
            "observed_events": 2,
            "threshold_celsius": "10",
            "calibration": {"cases": 10, "observed_events": 1},
        }
        write_json(self.source / "summary.json", self.summary)
        rows = [dict(date=date(2026, 1, 1) + timedelta(days=index), event=event,
                     independent_probability=probability, same_day_probability=event)
                for index, (probability, event) in enumerate(CASES)]
        write_csv(self.source / "matched-range.csv", rows[0], rows)

    def test_cli_saves_grid_expenses_and_hashes_without_overwriting(self):
        self.assertEqual(main(["--input-directory", str(self.source), "--output-directory", str(self.output)]), 0)
        path = self.output / "decision-value-summary.json"
        report = json.loads(path.read_bytes())
        self.assertEqual(report["alpha_grid"], list(ALPHAS))
        self.assertEqual(report["base_rate"], .1)
        self.assertEqual(report["base_rate_source"]["observed_events"], 1)
        self.assertEqual(report["evaluation_events"], 2)
        self.assertEqual(report["input_sha256"], {
            name: sha256((self.source / name).read_bytes()).hexdigest() for name in ("matched-range.csv", "summary.json")
        })
        with (self.output / "decision-value.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 38)
        independent = next(row for row in rows if row["method"] == "independent" and row["alpha"] == "0.2")
        self.assertAlmostEqual(float(independent["relative_value"]), .875)
        self.assertEqual(float(independent["mean_climatology_expense"]), .4)
        self.assertTrue(all(row["relative_value"] == 1. for row in report["methods"]["same_day"]))
        self.assertTrue((self.output / "decision-value.png").read_bytes().startswith(b"\x89PNG"))
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            run_study(self.source, self.output)
        self.assertEqual(path.read_bytes(), original)

    def test_cli_rejects_missing_calibration_or_incomplete_cases(self):
        for summary, message in (
            ({**self.summary, "calibration": None}, "frozen 2025 calibration"),
            ({**self.summary, "calibration": {"cases": 0, "observed_events": 0}}, "2025 calibration cohort"),
            ({**self.summary, "cases": 6}, "complete evaluation cohort"),
        ):
            with self.subTest(summary=summary):
                write_json(self.source / "summary.json", summary, overwrite=True)
                with self.assertRaisesRegex(ValueError, message):
                    run_study(self.source, self.output)
                self.assertFalse(self.output.exists())
