"""Freeze and evaluate a range-probability comparison on saved scalar forecasts."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
from statistics import fmean
from urllib.parse import urlencode

from model_pipelining import artifacts, backtest, experiments, noaa, snapshots
from model_pipelining import temperature_evaluation, temperature_regression
from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.evidence import source_hashes
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.temperature_evaluation import read_evaluated_predictions, require_matched_origins
from model_pipelining.temperature_regression import TemperatureRegression


STATION = "USW00094728"
HISTORY_START = date(2023, 1, 1)
CALIBRATION_START = date(2025, 1, 1)
ARCHIVE_START = date(2026, 1, 1)
EXTREMES = ("TMIN", "TMAX")
METHODS = ("independent", "same_day")
MODEL_NAME = "annual-plus-lag"
CALIBRATION_TABLE = "matched-extrema.csv"
HISTORY_FILES = {element: f"{element.lower()}-history.csv" for element in EXTREMES}


def _dates(start, end):
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _midnight(day):
    return datetime.combine(day, time.min, UTC)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _source_hashes():
    return source_hashes(__file__, *(module.__file__ for module in (
        artifacts, backtest, experiments, noaa, snapshots, temperature_evaluation, temperature_regression,
    )))


def _decimal(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Expected a finite decimal value, got {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"Expected a finite decimal value, got {value!r}")
    return result


def _csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _input_rows(raw):
    with io.StringIO(raw.decode("utf-8"), newline="") as stream:
        return list(csv.DictReader(stream))


def _history_values(raw):
    rows = _input_rows(raw)
    values = {datetime.fromisoformat(row["timestamp"]).date(): _decimal(row["value"]) for row in rows}
    if len(values) != len(rows):
        raise ValueError("Duplicate date in the calibration history")
    return values


def _calibration(raw):
    rows = _input_rows(raw)
    dates = _dates(CALIBRATION_START, ARCHIVE_START - timedelta(days=1))
    if [date.fromisoformat(row["date"]) for row in rows] != dates:
        raise ValueError("Calibration must cover every 2025 date once in order")
    values = {element: [(_decimal(row[f"actual_{suffix}"]), _decimal(row[f"{MODEL_NAME}_{suffix}"]))
                        for row in rows] for element, suffix in zip(EXTREMES, ("min", "max"), strict=True)}
    return dates, values


def _archive_url(end):
    query = urlencode(dict(dataset="daily-summaries", stations=STATION, startDate=ARCHIVE_START.isoformat(),
                          endDate=end.isoformat(), dataTypes="TMIN,TMAX", units="metric",
                          includeAttributes="true", format="csv"))
    return f"https://www.ncei.noaa.gov/access/services/data/v1?{query}"


def freeze(pair, maximum, output, *, start, end, threshold, score):
    """Copy verified 2025 calibration inputs and pin the later study design."""
    threshold = _decimal(threshold)
    if start < ARCHIVE_START or end < start or threshold <= 0 or score != "brier":
        raise ValueError("Choose an ordered post-2025 evaluation window, a positive threshold and Brier scoring")
    pair, maximum, output = map(Path, (pair, maximum, output))
    design_bytes = (pair / "design.json").read_bytes()
    design = json.loads(design_bytes)
    model = TemperatureRegression.from_configuration(design["model"])
    if model.name != MODEL_NAME:
        raise ValueError("Calibration requires the saved annual-plus-lag regressions")
    table_bytes = (pair / CALIBRATION_TABLE).read_bytes()
    dates, calibration = _calibration(table_bytes)
    maximum_evaluation = maximum / "evaluation"
    runs = {"TMIN": pair / "minimum", "TMAX": maximum_evaluation / MODEL_NAME}
    require_matched_origins(runs["TMAX"], runs["TMIN"], dates)

    files = {CALIBRATION_TABLE: table_bytes}
    tasks = {}
    for element, (store, snapshot_key) in zip(EXTREMES, (
        (pair / "snapshots", "minimum_snapshot"), (maximum_evaluation / "snapshots", "maximum_snapshot"),
    ), strict=True):
        run = runs[element]
        manifest = json.loads((run / "manifest.json").read_bytes())
        dataset = f"noaa-central-park-{element.lower()}"
        snapshot_id = design[snapshot_key]
        if (manifest["status"] != "completed" or manifest["model"]["configuration"] != model.configuration
                or manifest["model"]["name"] != MODEL_NAME or manifest["input"]["dataset"] != dataset
                or manifest["input"]["snapshot_id"] != snapshot_id):
            raise ValueError(f"Expected the completed calibration forecast and matching {element} snapshot")
        snapshot = snapshots.retrieve_snapshot(snapshot_id, dataset=dataset, store=store)
        history = snapshot.read_bytes()
        if _sha(history) != snapshot_id:
            raise ValueError(f"The {element} history differs from its snapshot ID")
        values = _history_values(history)
        if list(values) != _dates(HISTORY_START, ARCHIVE_START - timedelta(days=1)):
            raise ValueError(f"Incomplete {element} history in the calibration snapshot")
        predictions = read_evaluated_predictions(run / "evaluation.csv", dates, {day: float(v) for day, v in values.items()})
        if any(actual != values[day] or prediction != _decimal(saved)
               for day, (actual, prediction), saved in zip(dates, calibration[element], predictions, strict=True)):
            raise ValueError(f"The 2025 calibration table differs from the saved {element} run")
        files[HISTORY_FILES[element]] = history
        task = ExperimentSpec.from_dict(manifest["experiment"]).task
        if (task.calendar != "daily_utc" or task.horizon_days != 1 or task.issuance_delay_seconds != 0
                or task.availability_column is not None or task.revision_policy != "reject"
                or (task.timestamp_column, task.target_column, task.units) != ("timestamp", "value", "degrees Celsius")):
            raise ValueError(f"Expected the original daily availability and columns for {element}")
        tasks[element] = asdict(task)

    event_count = sum(maximum[0] - minimum[0] >= threshold
                      for minimum, maximum in zip(calibration["TMIN"], calibration["TMAX"], strict=True))
    plan = dict(
        format_version=1, frozen_at=datetime.now(UTC).isoformat(), station=STATION,
        evaluation_start=start.isoformat(), evaluation_end=end.isoformat(), source_url=_archive_url(end),
        threshold_celsius=str(threshold), score=score, methods=METHODS, model=model.configuration, tasks=tasks,
        calibration=dict(cases=len(dates), observed_events=event_count, pair_design_sha256=_sha(design_bytes)),
        input_sha256={name: _sha(data) for name, data in files.items()}, source_sha256=_source_hashes(),
        availability="Daily date labels simulate availability at midnight UTC",
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    plan_path = output / "plan.json"
    write_json(plan_path, plan)
    (output / "plan.sha256").write_text(_sha(plan_path.read_bytes()) + "\n", encoding="utf-8")
    return plan_path


def _read_plan(study):
    raw = (study / "plan.json").read_bytes()
    digest = _sha(raw)
    if digest != (study / "plan.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("The frozen range plan changed")
    plan = json.loads(raw)
    if plan["format_version"] != 1 or plan["source_sha256"] != _source_hashes():
        raise ValueError("The frozen range study sources changed")
    files = {}
    for name in (CALIBRATION_TABLE, *HISTORY_FILES.values()):
        data = (study / name).read_bytes()
        if _sha(data) != plan["input_sha256"].get(name):
            raise ValueError(f"The frozen input changed: {name}")
        files[name] = data
    return plan, digest, files


def _residual_differences(calibration):
    minimum, maximum = ([actual - prediction for actual, prediction in calibration[element]] for element in EXTREMES)
    return {
        "independent": sorted(high - low for low in minimum for high in maximum),
        "same_day": sorted(high - low for low, high in zip(minimum, maximum, strict=True)),
    }


def _case_probabilities(point_range, threshold, differences):
    events = len(differences) - bisect_left(differences, threshold - point_range)
    crossings = bisect_left(differences, -point_range)
    return events / len(differences), crossings / len(differences)


def _summarize(cases, method):
    bins = defaultdict(list)
    for row in cases:
        probability = row[f"{method}_probability"]
        bins[min(9, int(probability * 10))].append(row)
    return dict(
        brier=fmean(row[f"{method}_brier"] for row in cases),
        mean_probability=fmean(row[f"{method}_probability"] for row in cases),
        mean_negative_range_probability=fmean(row[f"{method}_negative_range_probability"] for row in cases),
        cases_with_negative_range_probability=sum(row[f"{method}_negative_range_probability"] > 0 for row in cases),
        calibration_bins=[dict(lower=index / 10, upper=(index + 1) / 10, cases=len(rows),
                               mean_probability=fmean(row[f"{method}_probability"] for row in rows),
                               event_rate=fmean(row["event"] for row in rows))
                          for index, rows in sorted(bins.items())],
    )


def _prepare_later_archive(files, output, plan, raw, end):
    values = {}
    for element in EXTREMES:
        values[element] = _history_values(files[HISTORY_FILES[element]])
        for timestamp, value in noaa.daily_temperatures(raw, station=plan["station"], start=ARCHIVE_START,
                                                       end=end, element=element):
            values[element][datetime.fromisoformat(timestamp).date()] = _decimal(value)
    if any(values["TMIN"][day] > values["TMAX"][day] for day in values["TMIN"]):
        raise ValueError("Observed minimum temperature exceeds maximum in the archive")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "noaa-later-source.csv"
    archive.write_bytes(raw)
    store = output / "snapshots"
    receipt = snapshots.register_snapshot(archive, dataset="noaa-central-park-later-raw", store=store)
    return values, receipt.snapshot_id


def evaluate(study, source, output):
    """Run scalar forecasts over the frozen dates, then compare saved probabilities."""
    study, source, output = map(Path, (study, source, output))
    plan, plan_sha256, files = _read_plan(study)
    _, calibration = _calibration(files[CALIBRATION_TABLE])
    start, end = (date.fromisoformat(plan[key]) for key in ("evaluation_start", "evaluation_end"))
    dates = _dates(start, end)
    values, raw_snapshot = _prepare_later_archive(files, output, plan, source.read_bytes(), end)
    write_json(output / "inputs.json", dict(plan_sha256=plan_sha256, raw_snapshot_id=raw_snapshot))
    store = output / "snapshots"
    model = TemperatureRegression.from_configuration(plan["model"])
    runs = {}
    for element in EXTREMES:
        dataset = f"noaa-central-park-{element.lower()}"
        normalized = output / f"{element.lower()}.csv"
        write_csv(normalized, ("timestamp", "value"), (dict(timestamp=_midnight(day).isoformat(), value=value)
                  for day, value in values[element].items()))
        receipt = snapshots.register_snapshot(normalized, dataset=dataset, store=store)
        experiment = ExperimentSpec(task=ForecastTask(**plan["tasks"][element]), evaluation=EvaluationPlan(
            role="final", final_start=_midnight(start), final_end=_midnight(end + timedelta(days=1)),
            benchmarks=("last-value",),
        ))
        runs[element] = backtest.run_backtest(dataset=dataset, snapshot_id=receipt.snapshot_id,
            store=store, experiment=experiment, model=model, output_directory=output / element.lower())
        if json.loads((runs[element] / "manifest.json").read_bytes())["status"] != "completed":
            raise ValueError(f"The {element} scalar run failed. See its manifest and checks.")

    require_matched_origins(runs["TMAX"], runs["TMIN"], dates)
    predictions = [read_evaluated_predictions(runs[element] / "evaluation.csv", dates,
                                              {day: float(values[element][day]) for day in dates})
                   for element in EXTREMES]
    origins = _csv_rows(runs["TMAX"] / "predictions.csv")
    differences = _residual_differences(calibration)
    threshold = _decimal(plan["threshold_celsius"])
    cases = []
    for day, minimum, maximum, origin in zip(dates, *predictions, origins, strict=True):
        issuance = _midnight(day - timedelta(days=1))
        if any(datetime.fromisoformat(origin[field]) != issuance for field in ("issued_at", "training_cutoff")):
            raise ValueError(f"Unexpected issuance or cutoff for {day}")
        actual_range = values["TMAX"][day] - values["TMIN"][day]
        point_range = _decimal(maximum) - _decimal(minimum)
        event = int(actual_range >= threshold)
        row = dict(date=day, issued_at=origin["issued_at"], training_cutoff=origin["training_cutoff"],
                   actual_range_celsius=actual_range, predicted_range_celsius=point_range, event=event)
        for method in METHODS:
            probability, crossings = _case_probabilities(point_range, threshold, differences[method])
            row.update({f"{method}_probability": probability, f"{method}_negative_range_probability": crossings,
                        f"{method}_brier": (probability - event)**2})
        row["brier_difference"] = row["same_day_brier"] - row["independent_brier"]
        cases.append(row)

    write_csv(output / "matched-range.csv", cases[0], cases)
    summary = output / "summary.json"
    events = sum(row["event"] for row in cases)
    write_json(summary, dict(evaluation_start=start.isoformat(), evaluation_end=end.isoformat(),
        cases=len(cases), observed_events=events, observed_event_rate=events / len(cases),
        threshold_celsius=plan["threshold_celsius"], calibration=plan["calibration"],
        draws_per_case={method: len(differences[method]) for method in METHODS},
        methods={method: _summarize(cases, method) for method in METHODS},
        mean_brier_difference_same_day_minus_independent=fmean(row["brier_difference"] for row in cases)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--pair-study", type=Path, required=True)
    prepare.add_argument("--maximum-study", type=Path, required=True)
    prepare.add_argument("--start", type=date.fromisoformat, required=True)
    prepare.add_argument("--end", type=date.fromisoformat, required=True)
    prepare.add_argument("--threshold", required=True)
    prepare.add_argument("--score", choices=("brier",), required=True)
    assess = commands.add_parser("evaluate")
    assess.add_argument("--study", type=Path, required=True)
    assess.add_argument("--source", type=Path, required=True)
    for command in (prepare, assess):
        command.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze(args.pair_study, args.maximum_study, args.output_directory, start=args.start,
                     end=args.end, threshold=args.threshold, score=args.score))
    else:
        print(evaluate(args.study, args.source, args.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
