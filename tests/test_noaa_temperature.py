"""Regression checks for the public example's input validation."""

import csv
from datetime import timedelta
import io
import unittest

from model_pipelining.noaa import daily_temperatures
from scripts.run_noaa_temperature import END, START, STATION, daily_maxima


def _fixture(*, gap=False, flagged=False, duplicate=False, flagged_minimum=False):
    result = io.StringIO()
    writer = csv.writer(result)
    writer.writerow(["STATION", "DATE", "TMAX", "TMAX_ATTRIBUTES", "TMIN", "TMIN_ATTRIBUTES"])
    for index in reversed(range((END - START).days + 1)):
        day = START + timedelta(days=index)
        if gap and index == 101:
            continue
        row = [STATION, day.isoformat(), "8.5", ",X,W" if flagged and index == 101 else ",,W",
               "1.5", ",X,W" if flagged_minimum and index == 101 else ",,W,0700"]
        writer.writerow(row)
        if duplicate and index == 101:
            writer.writerow(row)
    return result.getvalue().encode()


class NoaaTemperatureTests(unittest.TestCase):
    def test_calendar_is_sorted_with_leap_day_and_unchanged_units(self):
        rows = daily_maxima(_fixture())
        self.assertEqual(rows, daily_temperatures(_fixture(), station=STATION, start=START, end=END, element="TMAX"))
        self.assertEqual(len(rows), 731)
        self.assertEqual(rows[0], ("2023-01-01T00:00:00Z", 8.5))
        self.assertIn(("2024-02-29T00:00:00Z", 8.5), rows)
        self.assertEqual(rows[-1], ("2024-12-31T00:00:00Z", 8.5))

    def test_quality_flag_requires_caller_to_reconcile_the_input(self):
        source = _fixture(flagged=True)
        for read in (daily_maxima, self.parse_maximum):
            with self.subTest(read=read), self.assertRaisesRegex(ValueError, "flagged"):
                read(source)

    def test_gap_and_duplicate_fail_instead_of_changing_the_target_calendar(self):
        for read in (daily_maxima, self.parse_maximum):
            with self.subTest(read=read):
                with self.assertRaisesRegex(ValueError, "complete NOAA daily calendar"):
                    read(_fixture(gap=True))
                with self.assertRaisesRegex(ValueError, "Duplicate NOAA date"):
                    read(_fixture(duplicate=True))

    def parse_maximum(self, source):
        return daily_temperatures(source, station=STATION, start=START, end=END, element="TMAX")

    def test_minimum_uses_its_own_quality_flags(self):
        rows = daily_temperatures(_fixture(flagged=True), station=STATION, start=START, end=END, element="TMIN")
        self.assertEqual(len(rows), 731)
        self.assertIn(("2024-02-29T00:00:00Z", 1.5), rows)
        with self.assertRaisesRegex(ValueError, "flagged NOAA TMIN"):
            daily_temperatures(_fixture(flagged_minimum=True), station=STATION, start=START, end=END, element="TMIN")
