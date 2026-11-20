"""Check crossed forecasts and temporal matching across scalar runs."""

import csv
from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.study_noaa_extrema import ORIGIN_FIELDS, _paired_scores, _require_matched_origins


class NoaaExtremaTests(unittest.TestCase):
    def test_scores_count_a_crossing_and_preserve_its_negative_range(self):
        dates = [date(2025, 1, 1), date(2025, 1, 2)]
        result = _paired_scores(dates, np.array([8., 1.]), np.array([12., 3.]),
                                np.array([5., 7.]), np.array([8., 6.]))
        self.assertEqual(result["crossings"], 1)
        self.assertEqual(result["observed_crossings"], 0)
        self.assertEqual(result["tightest_predicted_range"], {"date": "2025-01-02", "celsius": -1.})
        self.assertEqual(result["daily_range"]["mae"], 2.)

    def test_pairing_rejects_same_date_with_different_eligible_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, counterpart = root / "max", root / "min"
            for run in (reference, counterpart):
                run.mkdir()

            def write(run, training_count):
                row = dict(zip(ORIGIN_FIELDS, ("730", "2025-01-01T00:00:00Z", "2024-12-31T00:00:00Z",
                    "2024-12-31T00:00:00Z", "2023-01-01T00:00:00Z", "2024-12-31T00:00:00Z",
                    str(training_count)), strict=True), status="passed", benchmark="last-value")
                for name in ("predictions.csv", "benchmark_predictions.csv"):
                    with (run / name).open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)

            write(reference, 731)
            write(counterpart, 731)
            _require_matched_origins(reference, counterpart, [date(2025, 1, 1)])
            write(counterpart, 730)
            with self.assertRaisesRegex(ValueError, "matching origins"):
                _require_matched_origins(reference, counterpart, [date(2025, 1, 1)])
