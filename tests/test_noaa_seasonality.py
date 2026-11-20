"""Check one-step regression behavior and reject unmatched saved cases."""

from datetime import UTC, date, datetime, timedelta
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model_pipelining.models import ForecastRequest, Observation
from model_pipelining.temperature_evaluation import read_evaluated_predictions, temperature_error_metrics
from model_pipelining.temperature_regression import TemperatureRegression
from scripts.study_noaa_seasonality import MODELS, TemperatureRegression as FrozenTemperatureRegression
from scripts.study_noaa_seasonality import _evaluated, _metrics


class NoaaSeasonalityTests(unittest.TestCase):
    def test_annual_and_lagged_regression_recovers_a_known_next_value(self):
        start = datetime(2023, 1, 1, tzinfo=UTC)
        values = [42.]
        for day in range(1, 367):
            phase = 2 * math.pi * day / 365.2425
            values.append(7 + 3 * math.sin(phase) - 2 * math.cos(phase) + .4 * values[-1])
        history = tuple(Observation(start + timedelta(days=index), value)
                        for index, value in enumerate(values[:-1]))
        model = TemperatureRegression("annual-plus-lag")
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
            for read in (_evaluated, read_evaluated_predictions):
                with self.subTest(read=read):
                    self.assertEqual(read(path, [day], {day: 8.}).tolist(), [7.])
                    with self.assertRaisesRegex(ValueError, "unmatched"):
                        read(path, [day], {day: 9.})
                    with self.assertRaisesRegex(ValueError, "frozen cohort"):
                        read(path, [day, day + timedelta(days=1)], {day: 8.})

    def test_all_variants_restore_and_match_the_frozen_regressions(self):
        start = datetime(2023, 12, 28, tzinfo=UTC)
        history = tuple(Observation(start + timedelta(days=index), value)
                        for index, value in enumerate((2., 8., 5., 10., -1., 3., 3., 5.)))
        cutoff = history[-1].timestamp
        request = ForecastRequest(history, cutoff, cutoff + timedelta(days=1))
        for name, features in MODELS.items():
            with self.subTest(name=name):
                frozen = FrozenTemperatureRegression(name, *features)
                saved = json.loads(json.dumps(frozen.configuration))
                restored = TemperatureRegression.from_configuration(saved)
                self.assertEqual(restored, TemperatureRegression(name))
                self.assertEqual(restored.configuration, saved)
                self.assertEqual(restored.forecast(request), frozen.forecast(request))
        with self.assertRaisesRegex(ValueError, "supported temperature regression"):
            TemperatureRegression.from_configuration({
                **TemperatureRegression("annual-plus-lag").configuration, "lag_days": 2,
            })

    def test_grouped_error_metrics_equal_the_frozen_summary(self):
        days = [date(2024, 2, 28) + timedelta(days=index) for index in range(4)]
        values = dict(zip(days, [1., 2., 7., 4.], strict=True))
        predictions = {"seasonal": np.array([1., 5., 5.]), "persistence": np.array([1., 2., 7.])}
        self.assertEqual(temperature_error_metrics(days[1:], values, predictions),
                         _metrics(days[1:], values, predictions))
