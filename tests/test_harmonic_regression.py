"""The weekly harmonic adapter recovers a known curve through the shared runner."""

import csv
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from model_pipelining.harmonic_regression import WeeklyHarmonicRegression
from model_pipelining.models import ForecastRequest, Observation
from scripts.run_harmonic_example import main as run_example


START = datetime(2020, 1, 1, tzinfo=UTC)


def curve(day):
    return 4 + 2 * math.sin(2 * math.pi * day / 7) - math.cos(2 * math.pi * day / 7)


def request(days):
    history = tuple(Observation(START + timedelta(days=day), curve(day)) for day in days)
    return ForecastRequest(history=history, issued_at=history[-1].timestamp,
                           target_time=history[-1].timestamp + timedelta(days=1))


class WeeklyHarmonicRegressionTests(unittest.TestCase):
    def test_noiseless_weekly_curve_is_recovered_at_the_next_day(self):
        self.assertAlmostEqual(WeeklyHarmonicRegression().forecast(request(range(10))), curve(10), places=10)

    def test_history_without_three_distinct_phases_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not identify"):
            WeeklyHarmonicRegression().forecast(request((0, 7)))

    def test_configuration_is_fixed(self):
        model = WeeklyHarmonicRegression.from_configuration({"period_days": 7, "harmonics": 1})
        self.assertEqual(model, WeeklyHarmonicRegression())
        with self.assertRaisesRegex(ValueError, "period_days=7"):
            WeeklyHarmonicRegression.from_configuration({"period_days": 14, "harmonics": 1})

    def test_example_command_runs_matched_cases_with_the_src_adapter(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "example"
            run_example(["--output-directory", str(output)])
            manifest = json.loads((output / "harmonic" / "manifest.json").read_text(encoding="utf-8"))
            with (output / "harmonic" / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["model"]["class"], "model_pipelining.harmonic_regression.WeeklyHarmonicRegression")
        self.assertIn("harmonic_regression.py", manifest["source_sha256"])
        self.assertEqual(len(rows), 72)
        self.assertTrue(all(row["status"] == "passed" for row in rows))


if __name__ == "__main__":
    unittest.main()
