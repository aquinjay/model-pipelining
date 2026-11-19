"""Diagnose saved NOAA errors and compare three small point regressions."""

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.reporting import compare_runs
from model_pipelining.snapshots import retrieve_snapshot


YEAR_DAYS = 365.2425
PHASE_ORIGIN = date(2023, 1, 1).toordinal()
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MODELS = {"annual-only": (True, False), "lag-only": (False, True), "annual-plus-lag": (True, True)}


@dataclass(frozen=True)
class TemperatureRegression:
    """Ordinary least squares with a fixed annual pair, a one-day lag, or both."""

    name: str
    annual: bool
    lag: bool
    supports_quantiles = False
    supports_highest_density = False

    @property
    def configuration(self):
        return dict(intercept=True, annual_period_days=YEAR_DAYS if self.annual else None,
                    phase_origin="2023-01-01" if self.annual else None, lag_days=1 if self.lag else None)

    def forecast(self, request):
        values = np.array([row.value for row in request.history])
        ordinals = np.array([row.timestamp.toordinal() for row in request.history])
        target = request.target_time.toordinal()
        if len(values) < 2 or target != ordinals[-1] + 1 or not np.all(np.diff(ordinals) == 1):
            raise ValueError("Temperature regression requires a contiguous one-day forecast")
        # Use the same response dates for all three variants. The first
        # eligible observation can supply a lag but has no earlier value.
        design = [np.ones(len(values) - 1)]
        future = [1.]
        if self.annual:
            phase = 2 * np.pi * (ordinals[1:] - PHASE_ORIGIN) / YEAR_DAYS
            target_phase = 2 * np.pi * (target - PHASE_ORIGIN) / YEAR_DAYS
            design.extend((np.sin(phase), np.cos(phase)))
            future.extend((np.sin(target_phase), np.cos(target_phase)))
        if self.lag:
            design.append(values[:-1])
            future.append(values[-1])
        coefficients, _, rank, _ = np.linalg.lstsq(np.column_stack(design), values[1:], rcond=None)
        if rank != len(design):
            raise ValueError("Temperature history does not identify the regression coefficients")
        return float(np.dot(future, coefficients))


def _evaluated(path, dates, values, *, benchmark=False):
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


def _metrics(dates, values, predictions):
    actual = np.array([values[day] for day in dates])
    previous = np.array([values[day - timedelta(days=1)] for day in dates])
    change = actual - previous
    month = np.array([day.month for day in dates])
    absolute_change = np.abs(change)
    groups = {
        "seasons": {name: np.isin(month, months) for name, months in SEASONS.items()},
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


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_study(source, output):
    source, output = Path(source), Path(output)
    baseline = source / "forecast"
    design = json.loads((source / "design.json").read_text(encoding="utf-8"))
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    dataset, snapshot_id = design["dataset"], design["snapshot_id"]
    if (manifest["status"] != "completed" or manifest["model"]["name"] != "local-level"
            or dataset != "noaa-central-park-tmax"
            or (dataset, snapshot_id) != (manifest["input"]["dataset"], manifest["input"]["snapshot_id"])
            or design["experiment"] != manifest["experiment"]):
        raise ValueError("Expected a completed Central Park Gaussian example and its frozen design")

    store = source / "snapshots"
    path = retrieve_snapshot(snapshot_id, dataset=dataset, store=store)
    with path.open(encoding="utf-8", newline="") as stream:
        values = {datetime.fromisoformat(row["timestamp"]).date(): float(row["value"])
                  for row in csv.DictReader(stream)}
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(366)]
    predictions = {
        "local-level": _evaluated(baseline / "evaluation.csv", dates, values),
        "last-value": _evaluated(baseline / "benchmark_evaluation.csv", dates, values, benchmark=True),
    }
    if not np.array_equal(predictions["last-value"], [values[day - timedelta(days=1)] for day in dates]):
        raise ValueError("The saved last-value benchmark differs from the previous dated observation")

    output.mkdir(parents=True, exist_ok=False)
    models = [TemperatureRegression(name, *features) for name, features in MODELS.items()]
    _write_json(output / "design.json", dict(source=str(source), dataset=dataset, snapshot_id=snapshot_id,
        target_dates=[str(dates[0]), str(dates[-1])], cases=len(dates),
        experiment=design["experiment"], models={model.name: model.configuration for model in models},
        movement_thresholds_celsius=[2, 5],
        scope="Exploratory development comparison after baseline error inspection",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    _write_json(output / "baseline-diagnostics.json", _metrics(dates, values, predictions))

    experiment = ExperimentSpec.from_dict(design["experiment"])
    for model in models:
        run = run_backtest(dataset=dataset, snapshot_id=snapshot_id, store=store,
            experiment=experiment, model=model, output_directory=output / model.name)
        # Also compares the entire frozen history identity and scoring vintage.
        compare_runs(baseline, run, output_directory=output / f"compare-{model.name}")
        predictions[model.name] = _evaluated(run / "evaluation.csv", dates, values)

    _write_json(output / "summary.json", _metrics(dates, values, predictions))
    with (output / "matched-cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "actual", "previous_actual", *predictions])
        for index, day in enumerate(dates):
            writer.writerow([day, values[day], values[day - timedelta(days=1)],
                             *(forecast[index] for forecast in predictions.values())])
    return output / "summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    options = parser.parse_args()
    print(run_study(options.source_directory, options.output_directory))


if __name__ == "__main__":
    main()
