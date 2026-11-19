"""Check one-step regression behavior and reject unmatched saved cases."""

from datetime import UTC, datetime, timedelta
import math
from pathlib import Path
import tempfile
import unittest

from model_pipelining.models import ForecastRequest, Observation
from scripts.study_noaa_seasonality import TemperatureRegression, _evaluated


class NoaaSeasonalityTests(unittest.TestCase):
    def test_annual_and_lagged_regression_recovers_a_known_next_value(self):
        start = datetime(2023, 1, 1, tzinfo=UTC)
        values = [42.]
        for day in range(1, 367):
            phase = 2 * math.pi * day / 365.2425
            values.append(7 + 3 * math.sin(phase) - 2 * math.cos(phase) + .4 * values[-1])
        history = tuple(Observation(start + timedelta(days=index), value)
                        for index, value in enumerate(values[:-1]))
        model = TemperatureRegression("annual-plus-lag", True, True)
        cutoff = history[-1].timestamp
        result = model.forecast(ForecastRequest(history, cutoff, cutoff + timedelta(days=1)))
        self.assertAlmostEqual(result, values[-1], places=10)
        with self.assertRaisesRegex(ValueError, "contiguous one-day"):
            model.forecast(ForecastRequest(history, cutoff, cutoff + timedelta(days=2)))

    def test_saved_evaluation_requires_matching_dates_and_actuals(self):
        day = datetime(2024, 1, 1, tzinfo=UTC).date()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            path.write_text("target_time,actual,prediction,status\n2024-01-01T00:00:00Z,8,7,passed\n")
            self.assertEqual(_evaluated(path, [day], {day: 8.}).tolist(), [7.])
            with self.assertRaisesRegex(ValueError, "unmatched"):
                _evaluated(path, [day], {day: 9.})
            with self.assertRaisesRegex(ValueError, "frozen cohort"):
                _evaluated(path, [day, day + timedelta(days=1)], {day: 8.})
