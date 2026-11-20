"""Validate temperatures from a caller-supplied metric NOAA daily archive."""

import csv
from datetime import date, timedelta
import io
import math


def daily_temperatures(source: bytes, *, station: str, start: date, end: date,
                       element: str) -> list[tuple[str, float]]:
    """Return date-sorted UTC labels and Celsius values for TMIN or TMAX.

    The source must already use metric units and include the element's
    attributes. The date labels support simulated availability.
    """
    if element not in ("TMIN", "TMAX"):
        raise ValueError("NOAA temperature element must be TMIN or TMAX")
    if start > end:
        raise ValueError("NOAA calendar start must not follow its end")
    attributes = f"{element}_ATTRIBUTES"
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig")))
    columns = {"STATION", "DATE", element, attributes}
    if reader.fieldnames is None or not columns.issubset(reader.fieldnames):
        raise ValueError(f"Expected NOAA station, date, {element} and {element} attributes")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Duplicate NOAA CSV column")
    values = {}
    for row in reader:
        if row["STATION"] != station:
            raise ValueError(f"Expected NOAA station {station}")
        try:
            day = date.fromisoformat(row["DATE"])
            value = float(row[element])
            flags = row[attributes].split(",")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"Invalid NOAA {element} value at line {reader.line_num}") from error
        if not math.isfinite(value) or value == -9999:
            raise ValueError(f"Missing or invalid {element} on {day}")
        if len(flags) not in (3, 4) or flags[1].strip():
            raise ValueError(f"Missing or flagged NOAA {element} quality information on {day}")
        if day in values:
            raise ValueError(f"Duplicate NOAA date {day}")
        values[day] = value
    expected = [start + timedelta(days=index) for index in range((end - start).days + 1)]
    if sorted(values) != expected:
        raise ValueError(f"Expected a complete NOAA daily calendar from {start} to {end}")
    return [(f"{day}T00:00:00Z", values[day]) for day in expected]
