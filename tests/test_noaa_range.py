"""Check finite residual probabilities and the frozen range study workflow."""

import csv
from datetime import date
from decimal import Decimal
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.snapshots import register_snapshot
from model_pipelining.temperature_regression import TemperatureRegression
from scripts.study_noaa_range import (
    ARCHIVE_START, CALIBRATION_START, HISTORY_START, STATION, _case_probabilities,
    _csv_rows, _dates, _midnight, _residual_differences, evaluate, freeze,
)


class NoaaRangeTests(unittest.TestCase):
    def test_residual_pairing_preserves_threshold_ties_and_negative_ranges(self):
        fixtures = {
            "TMIN": [(Decimal(0), Decimal(0)), (Decimal(12), Decimal(0))],
            "TMAX": [(Decimal(0), Decimal(0)), (Decimal(12), Decimal(0))],
        }
        draws = _residual_differences(fixtures)
        self.assertEqual(draws["independent"], [-12, 0, 0, 12])
        self.assertEqual(draws["same_day"], [0, 0])
        self.assertEqual(_case_probabilities(10, 10, draws["independent"]), (.75, .25))
        self.assertEqual(_case_probabilities(10, 10, draws["same_day"]), (1., 0.))
        self.assertEqual(_case_probabilities(12, 10, draws["independent"]), (.75, 0.))

    def calibration_fixture(self, root):
        pair, maximum = root / "pair", root / "maximum"
        pair.mkdir()
        maximum.mkdir()
        (maximum / "evaluation").mkdir()
        history_dates = _dates(HISTORY_START, date(2025, 12, 31))
        calibration_dates = _dates(CALIBRATION_START, date(2025, 12, 31))
        minimum = [round(12 + 10 * math.sin(index / 58) + (index % 5) / 3, 1)
                   for index in range(len(history_dates))]
        maximum_values = [round(value + 8 + (index % 7) / 2, 1) for index, value in enumerate(minimum)]
        model = TemperatureRegression("annual-plus-lag")
        experiment = ExperimentSpec(task=ForecastTask(minimum_history=365, units="degrees Celsius"), evaluation=EvaluationPlan(
            role="final", final_start=_midnight(CALIBRATION_START), final_end=_midnight(ARCHIVE_START), benchmarks=("last-value",),
        ))
        calibration = [dict(date=day) for day in calibration_dates]
        design = {"model": model.configuration}
        for suffix, element, values, store, output, key in (
            ("min", "tmin", minimum, pair / "snapshots", pair / "minimum", "minimum_snapshot"),
            ("max", "tmax", maximum_values, maximum / "evaluation/snapshots",
             maximum / "evaluation/annual-plus-lag", "maximum_snapshot"),
        ):
            normalized = root / f"{element}.csv"
            write_csv(normalized, ("timestamp", "value"), (dict(timestamp=_midnight(day).isoformat(), value=value)
                      for day, value in zip(history_dates, values, strict=True)))
            dataset = f"noaa-central-park-{element}"
            receipt = register_snapshot(normalized, dataset=dataset, store=store)
            run_backtest(dataset=dataset, snapshot_id=receipt.snapshot_id, store=store, model=model,
                         experiment=experiment, output_directory=output)
            design[key] = receipt.snapshot_id
            for row, scored in zip(calibration, _csv_rows(output / "evaluation.csv"), strict=True):
                row[f"actual_{suffix}"] = scored["actual"]
                row[f"annual-plus-lag_{suffix}"] = scored["prediction"]
        write_json(pair / "design.json", design)
        write_csv(pair / "matched-extrema.csv", calibration[0], calibration)
        return pair, maximum

    def later_fixture(self, path, last_maximum):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("STATION", "DATE", "TMIN", "TMIN_ATTRIBUTES", "TMAX", "TMAX_ATTRIBUTES"))
            for day, minimum, maximum in ((1, "5.2", "13.2"), (2, "7.4", "17.4"), (3, "4.2", last_maximum)):
                writer.writerow((STATION, date(2026, 1, day), minimum, ",,W", maximum, ",,W"))

    def test_freeze_evaluate_and_replay_keep_future_actuals_out_of_forecasts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair, maximum = self.calibration_fixture(root)
            study = root / "frozen"
            plan_path = freeze(pair, maximum, study, start=date(2026, 1, 2), end=date(2026, 1, 3),
                               threshold="10", score="brier")
            plan = json.loads(plan_path.read_bytes())
            self.assertEqual(plan["calibration"]["cases"], 365)
            self.assertIn("temperature_regression.py", plan["source_sha256"])
            self.assertEqual((study / "matched-extrema.csv").read_bytes(), (pair / "matched-extrema.csv").read_bytes())
            source = root / "later.csv"
            self.later_fixture(source, "11.2")
            summary_path = evaluate(study, source, root / "first")
            summary = json.loads(summary_path.read_bytes())
            self.assertEqual(summary["observed_events"], 1)
            self.assertEqual(summary["draws_per_case"], {"independent": 365**2, "same_day": 365})
            cases = _csv_rows(summary_path.parent / "matched-range.csv")
            self.assertEqual(cases[0]["date"], "2026-01-02")
            self.assertEqual(cases[0]["issued_at"], "2026-01-01T00:00:00+00:00")
            self.assertEqual(cases[0]["actual_range_celsius"], "10.0")
            for row in cases:
                for method in ("independent", "same_day"):
                    self.assertAlmostEqual(float(row[f"{method}_brier"]),
                                           (float(row[f"{method}_probability"]) - int(row["event"]))**2)
            self.assertAlmostEqual(summary["mean_brier_difference_same_day_minus_independent"],
                                   sum(float(row["brier_difference"]) for row in cases) / len(cases))
            replay = evaluate(study, source, root / "replay")
            self.assertEqual(replay.read_bytes(), summary_path.read_bytes())
            self.assertEqual((replay.parent / "matched-range.csv").read_bytes(),
                             (summary_path.parent / "matched-range.csv").read_bytes())

            self.later_fixture(source, "20.2")
            changed = evaluate(study, source, root / "changed")
            changed_cases = _csv_rows(changed.parent / "matched-range.csv")
            self.assertEqual(changed_cases[0], cases[0])
            self.assertEqual(changed_cases[1]["event"], "1")
            for field in ("predicted_range_celsius", "independent_probability", "same_day_probability"):
                self.assertEqual(changed_cases[1][field], cases[1][field])

            missing_history = root / "missing-history"
            source.write_text(source.read_text().replace("2026-01-01", "2026-01-04"))
            with self.assertRaisesRegex(ValueError, "complete NOAA daily calendar"):
                evaluate(study, source, missing_history)
            self.assertFalse(missing_history.exists())
            with patch("scripts.study_noaa_range._source_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "frozen range study sources changed"):
                    evaluate(study, root / "unread.csv", root / "changed-sources")
            with (study / "matched-extrema.csv").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "frozen input changed"):
                evaluate(study, root / "unread.csv", root / "tampered")
            self.assertFalse((root / "tampered").exists())
