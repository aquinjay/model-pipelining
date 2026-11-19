"""Regression checks for the public example's input validation."""

import csv
from datetime import timedelta
import io
import unittest

from scripts.run_noaa_temperature import END, START, STATION, daily_maxima


def _fixture(*, gap=False, flagged=False, duplicate=False):
    result = io.StringIO()
    writer = csv.writer(result)
    writer.writerow(["STATION", "DATE", "TMAX", "TMAX_ATTRIBUTES"])
    for index in reversed(range((END - START).days + 1)):
        day = START + timedelta(days=index)
        if gap and index == 101:
            continue
        row = [STATION, day.isoformat(), "8.5", ",X,W" if flagged and index == 101 else ",,W"]
        writer.writerow(row)
        if duplicate and index == 101:
            writer.writerow(row)
    return result.getvalue().encode()


class NoaaTemperatureTests(unittest.TestCase):
    def test_calendar_is_sorted_with_leap_day_and_unchanged_units(self):
        rows = daily_maxima(_fixture())
        self.assertEqual(len(rows), 731)
        self.assertEqual(rows[0], ("2023-01-01T00:00:00Z", 8.5))
        self.assertIn(("2024-02-29T00:00:00Z", 8.5), rows)
        self.assertEqual(rows[-1], ("2024-12-31T00:00:00Z", 8.5))

    def test_quality_flag_requires_caller_to_reconcile_the_input(self):
        with self.assertRaisesRegex(ValueError, "flagged"):
            daily_maxima(_fixture(flagged=True))

    def test_gap_and_duplicate_fail_instead_of_changing_the_target_calendar(self):
        with self.assertRaisesRegex(ValueError, "complete NOAA daily calendar"):
            daily_maxima(_fixture(gap=True))
        with self.assertRaisesRegex(ValueError, "Duplicate NOAA date"):
            daily_maxima(_fixture(duplicate=True))
