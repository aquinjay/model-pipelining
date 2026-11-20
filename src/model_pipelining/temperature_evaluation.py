"""Verify and summarize saved scalar temperature forecasts on matching dates."""

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


ORIGIN_FIELDS = ("origin_index", "target_time", "issued_at", "training_cutoff",
                 "training_start", "training_end", "training_count")
_SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def read_evaluated_predictions(path: Path, dates, values, *, benchmark: bool = False) -> np.ndarray:
    """Return finite, successful saved predictions in the declared date order."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(dates):
        raise ValueError(f"Expected the frozen cohort in {path.name}")
    predictions = []
    for row, day in zip(rows, dates, strict=True):
        if (row["status"] != "passed" or datetime.fromisoformat(row["target_time"]).date() != day
                or float(row["actual"]) != values[day] or (benchmark and row["benchmark"] != "last-value")):
            raise ValueError(f"Failed or unmatched {path.name} case for {day}")
        predicted = float(row["prediction"])
        if not np.isfinite(predicted):
            raise ValueError(f"Nonfinite forecast for {day}")
        predictions.append(predicted)
    return np.array(predictions)


def _origins(path: Path, *, benchmark: bool = False) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["status"] != "passed" or (benchmark and row["benchmark"] != "last-value") for row in rows):
        raise ValueError(f"Failed or unexpected forecast case in {path}")
    return [tuple(row[key] for key in ORIGIN_FIELDS) for row in rows]


def require_matched_origins(reference: Path, counterpart: Path, dates) -> None:
    """Require matching saved origins for the two scalar runs and their last-value benchmarks."""
    expected = [(day, day - timedelta(days=1)) for day in dates]
    reference_origins = _origins(reference / "predictions.csv")
    counterpart_origins = _origins(counterpart / "predictions.csv")
    for run, origins in ((reference, reference_origins), (counterpart, counterpart_origins)):
        if origins != _origins(run / "benchmark_predictions.csv", benchmark=True):
            raise ValueError("Model and persistence benchmark origins must match")
    if (counterpart_origins != reference_origins or
            [(datetime.fromisoformat(row[1]).date(), datetime.fromisoformat(row[5]).date())
             for row in reference_origins] != expected):
        raise ValueError("Temperature forecasts must use matching origins and eligible dates")


def temperature_error_metrics(dates, values, predictions) -> dict:
    """Summarize point errors overall, by season/month, and by realized daily change."""
    actual = np.array([values[day] for day in dates])
    previous = np.array([values[day - timedelta(days=1)] for day in dates])
    month = np.array([day.month for day in dates])
    absolute_change = np.abs(actual - previous)
    groups = {
        "seasons": {name: np.isin(month, months) for name, months in _SEASONS.items()},
        "months": {f"{item:02d}": month == item for item in range(1, 13)},
        "realized_change": {
            "under_2": absolute_change < 2,
            "2_to_under_5": (absolute_change >= 2) & (absolute_change < 5),
            "5_or_more": absolute_change >= 5,
        },
    }

    def summarize(errors, mask):
        subset = errors[mask]
        if not len(subset):
            return dict(cases=0, bias=None, mae=None, rmse=None)
        return dict(cases=len(subset), bias=float(subset.mean()),
                    mae=float(np.abs(subset).mean()), rmse=float(np.sqrt(np.mean(subset**2))))

    summaries = {}
    for name, prediction in predictions.items():
        errors = prediction - actual
        squared_error = float(np.sum(errors**2))
        summaries[name] = dict(
            overall=summarize(errors, np.ones(len(actual), dtype=bool)),
            **{group: {label: summarize(errors, mask) for label, mask in masks.items()}
               for group, masks in groups.items()},
            squared_error_share_on_5_or_more=(
                float(np.sum(errors[absolute_change >= 5]**2) / squared_error) if squared_error else None),
        )
    return summaries
