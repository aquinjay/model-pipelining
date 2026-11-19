"""Compare separately fitted Central Park minimum and maximum forecasts."""

import argparse
import csv
from dataclasses import replace
from datetime import date, datetime, timedelta
import hashlib
import io
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.snapshots import register_snapshot, retrieve_snapshot
from scripts.run_noaa_temperature import START, daily_maxima
from scripts.study_noaa_seasonality import MODELS, TemperatureRegression, _evaluated, _write_json


FIRST_TARGET = date(2025, 1, 1)
LAST_TARGET = date(2025, 12, 31)
SOURCE_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1?"
    "dataset=daily-summaries&stations=USW00094728&startDate=2023-01-01&endDate=2025-12-31"
    "&dataTypes=TMIN,TMAX&units=metric&includeAttributes=true&format=csv"
)
ORIGIN_FIELDS = ("origin_index", "target_time", "issued_at", "training_cutoff",
                 "training_start", "training_end", "training_count")


def _archive_minimum(raw):
    """Map the parallel NOAA fields into the existing daily temperature validator."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = ("STATION", "DATE", "TMIN", "TMIN_ATTRIBUTES")
    if reader.fieldnames is None or not set(fields).issubset(reader.fieldnames):
        raise ValueError("Expected NOAA TMIN and TMIN attributes")
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("STATION", "DATE", "TMAX", "TMAX_ATTRIBUTES"))
    writer.writerows([row[key] for key in fields] for row in reader)
    try:
        return daily_maxima(output.getvalue().encode(), start=START, end=LAST_TARGET)
    except ValueError as error:
        raise ValueError("Invalid NOAA minimum-temperature archive") from error


def _values(rows):
    return {datetime.fromisoformat(timestamp).date(): value for timestamp, value in rows}


def _origins(path, *, benchmark=False):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["status"] != "passed" or (benchmark and row["benchmark"] != "last-value") for row in rows):
        raise ValueError(f"Failed or unexpected forecast case in {path}")
    return [tuple(row[key] for key in ORIGIN_FIELDS) for row in rows]


def _require_matched_origins(reference, counterpart, dates):
    expected = [(day, day - timedelta(days=1)) for day in dates]
    max_origins = _origins(reference / "predictions.csv")
    min_origins = _origins(counterpart / "predictions.csv")
    for run, origins in ((reference, max_origins), (counterpart, min_origins)):
        if origins != _origins(run / "benchmark_predictions.csv", benchmark=True):
            raise ValueError("Model and persistence benchmark origins must match")
    if (min_origins != max_origins or
            [(datetime.fromisoformat(row[1]).date(), datetime.fromisoformat(row[5]).date())
             for row in max_origins] != expected):
        raise ValueError("Minimum and maximum forecasts must use matching origins and eligible dates")


def _paired_scores(dates, minimum_actual, maximum_actual, minimum_forecast, maximum_forecast):
    actual_range = maximum_actual - minimum_actual
    predicted_range = maximum_forecast - minimum_forecast
    min_error, max_error = minimum_forecast - minimum_actual, maximum_forecast - maximum_actual

    def scores(error):
        return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                    bias=float(error.mean()))

    tightest = int(predicted_range.argmin())
    correlation = (float(np.corrcoef(min_error, max_error)[0, 1])
                   if np.ptp(min_error) and np.ptp(max_error) else None)
    return dict(cases=len(dates), crossings=int(np.count_nonzero(predicted_range < 0)),
        observed_crossings=int(np.count_nonzero(actual_range < 0)),
        tightest_predicted_range=dict(date=str(dates[tightest]), celsius=float(predicted_range[tightest])),
        minimum=scores(min_error), maximum=scores(max_error), daily_range=scores(predicted_range - actual_range),
        paired_error_correlation=correlation)


def _plot(path, minimum, maximum, forecasts):
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    spread, paired_errors = figure.subplots(1, 2)
    actual = maximum - minimum
    span = float(max(actual.max(), *((maxima - minima).max() for minima, maxima in forecasts.values())))
    spread.plot([0, span], [0, span], linestyle="--", color="0.6", linewidth=1)
    spread.axhline(0., color="0.5", linewidth=1)
    for name, (minima, maxima) in forecasts.items():
        spread.scatter(actual, maxima - minima, s=10, alpha=.45, label=name)
    spread.set(xlabel="Observed range (°C)", ylabel="Forecast range (°C)", title="Daily maximum minus minimum")
    spread.legend()
    minima, maxima = forecasts["annual-plus-lag"]
    paired_errors.axhline(0., color="0.5", linewidth=1)
    paired_errors.axvline(0., color="0.5", linewidth=1)
    paired_errors.scatter(minima - minimum, maxima - maximum, s=12, alpha=.55)
    paired_errors.set(xlabel="Minimum forecast error (°C)", ylabel="Maximum forecast error (°C)",
                      title="Errors from the two seasonal regressions")
    figure.savefig(path, dpi=150)


def run_study(maximum_study, source, output):
    maximum_study, source, output = Path(maximum_study), Path(source), Path(output)
    evaluation = maximum_study / "evaluation"
    reference = evaluation / "annual-plus-lag"
    plan_bytes = (maximum_study / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    inputs = json.loads((evaluation / "inputs.json").read_bytes())
    manifest = json.loads((reference / "manifest.json").read_bytes())
    snapshot_id = inputs["normalized_snapshot_id"]
    model = TemperatureRegression("annual-plus-lag", *MODELS["annual-plus-lag"])
    if (hashlib.sha256(plan_bytes).hexdigest() != inputs["plan_sha256"]
            or manifest["status"] != "completed" or manifest["experiment"] != plan["experiment"]
            or manifest["input"]["snapshot_id"] != snapshot_id
            or manifest["model"]["configuration"] != model.configuration):
        raise ValueError("Expected the completed frozen annual-plus-lag maximum-temperature run")

    maximum_path = retrieve_snapshot(snapshot_id, dataset=manifest["input"]["dataset"], store=evaluation / "snapshots")
    with maximum_path.open(encoding="utf-8", newline="") as stream:
        maxima = _values((row["timestamp"], float(row["value"])) for row in csv.DictReader(stream))
    raw = source.read_bytes()
    minimum_rows = _archive_minimum(raw)
    minima = _values(minimum_rows)
    new_maxima = _values(daily_maxima(raw, start=START, end=LAST_TARGET))
    if maxima.keys() != minima.keys():
        raise ValueError("The maximum-temperature snapshot and minimum archive have different calendars")
    dates = [FIRST_TARGET + timedelta(days=i) for i in range((LAST_TARGET - FIRST_TARGET).days + 1)]
    initial_experiment = ExperimentSpec.from_dict(plan["experiment"])
    if (initial_experiment.evaluation.role != "final"
            or initial_experiment.evaluation.final_start.date() != FIRST_TARGET
            or initial_experiment.evaluation.final_end.date() != LAST_TARGET + timedelta(days=1)
            or initial_experiment.evaluation.benchmarks != ("last-value",)):
        raise ValueError("Expected the frozen 2025 targets and last-value benchmark")
    experiment = replace(initial_experiment, task=replace(initial_experiment.task,
        name="central-park-daily-tmin", target_description="Next dated observation of daily minimum temperature at Central Park"))

    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "noaa-extrema-source.csv"
    raw_path.write_bytes(raw)
    minimum_path = output / "central-park-tmin-2023-2025.csv"
    with minimum_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("timestamp", "value"))
        writer.writerows(minimum_rows)
    store = output / "snapshots"
    receipt = register_snapshot(raw_path, dataset="noaa-central-park-extrema-raw", store=store)
    minimum = register_snapshot(minimum_path, dataset="noaa-central-park-tmin", store=store,
                                note=f"TMIN extracted from raw snapshot {receipt.snapshot_id}")
    _write_json(output / "design.json", dict(
        maximum_study=str(maximum_study), maximum_snapshot=snapshot_id,
        minimum_snapshot=minimum.snapshot_id, source_snapshot=receipt.snapshot_id, source_url=SOURCE_URL,
        same_archive_maximum_values=sum(new_maxima[day] == maxima[day] for day in maxima),
        archive_calendar_days=len(maxima), observed_crossings_in_archive=sum(minima[day] > maxima[day] for day in maxima),
        minimum_experiment=experiment.to_dict(), model=model.configuration,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    run = run_backtest(dataset=minimum.dataset, snapshot_id=minimum.snapshot_id, store=store,
                        experiment=experiment, model=model, output_directory=output / "minimum")
    _require_matched_origins(reference, run, dates)
    observed_min, observed_max = (np.array([values[day] for day in dates]) for values in (minima, maxima))
    forecasts = {}
    for name, file in (("annual-plus-lag", "evaluation.csv"), ("last-value", "benchmark_evaluation.csv")):
        benchmark = name == "last-value"
        forecasts[name] = (_evaluated(run / file, dates, minima, benchmark=benchmark),
                           _evaluated(reference / file, dates, maxima, benchmark=benchmark))
    summary = {name: _paired_scores(dates, observed_min, observed_max, *predictions)
               for name, predictions in forecasts.items()}
    _write_json(output / "summary.json", summary)
    with (output / "matched-extrema.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "actual_min", "actual_max",
                         *(f"{name}_{extreme}" for name in forecasts for extreme in ("min", "max"))])
        for index, day in enumerate(dates):
            writer.writerow([day, minima[day], maxima[day],
                             *(prediction[index] for series in forecasts.values() for prediction in series)])
    _plot(output / "paired-forecasts.png", observed_min, observed_max, forecasts)
    return output / "summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Local NOAA metric CSV with TMIN, TMAX and attributes")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(run_study(args.maximum_study, args.source, args.output_directory))


if __name__ == "__main__":
    main()
