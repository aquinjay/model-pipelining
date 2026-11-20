"""Diagnose and evaluate seasonal residual pools for Central Park range forecasts."""

import argparse
import csv
from datetime import UTC, date, datetime, timedelta
import io
import json
from pathlib import Path
from statistics import StatisticsError, correlation, fmean, pstdev
from urllib.parse import parse_qsl, urlencode, urlsplit

from model_pipelining import noaa
from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.snapshots import retrieve_snapshot
from scripts import study_noaa_range as annual


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_BY_MONTH = {month: name for name, months in SEASONS.items() for month in months}
METHODS = (*annual.METHODS, "seasonal_independent", "seasonal_same_day")
ARCHIVE_FIELDS = ("STATION", "DATE", "TMIN", "TMIN_ATTRIBUTES", "TMAX", "TMAX_ATTRIBUTES")


def _json(path):
    return json.loads(path.read_bytes())


def _write_plan(directory, plan):
    path = directory / "plan.json"
    write_json(path, plan)
    (directory / "plan.sha256").write_text(annual._sha(path.read_bytes()) + "\n", encoding="utf-8")
    return path


def _development_archive(study, run):
    plan, digest, files = annual._read_plan(study)
    inputs = _json(run / "inputs.json")
    if inputs["plan_sha256"] != digest:
        raise ValueError("Development run does not match its frozen range plan")
    snapshot = retrieve_snapshot(inputs["raw_snapshot_id"], dataset="noaa-central-park-later-raw", store=run / "snapshots")
    raw = snapshot.read_bytes()
    if annual._sha(raw) != inputs["raw_snapshot_id"]:
        raise ValueError("Development archive differs from its snapshot ID")
    return plan, files, raw


def _ranges(calibration):
    return [high[0] - low[0] for low, high in zip(calibration["TMIN"], calibration["TMAX"], strict=True)]


def _seasonal_pools(dates, calibration):
    return {name: {element: [row for day, row in zip(dates, calibration[element], strict=True) if day.month in months]
                   for element in annual.EXTREMES} for name, months in SEASONS.items()}


def _score(dates, cases, calibration_dates, calibration, threshold):
    if [date.fromisoformat(row["date"]) for row in cases] != dates:
        raise ValueError("Range forecasts do not match the declared dates")
    seasonal = {name: annual._residual_differences(pool)
                for name, pool in _seasonal_pools(calibration_dates, calibration).items()}
    full_year = annual._residual_differences(calibration)
    output = []
    for day, saved in zip(dates, cases, strict=True):
        season = SEASON_BY_MONTH[day.month]
        point = annual._decimal(saved["predicted_range_celsius"])
        event = int(annual._decimal(saved["actual_range_celsius"]) >= threshold)
        if event != int(saved["event"]):
            raise ValueError(f"Saved event disagrees with the frozen threshold for {day}")
        row = {field: saved[field] for field in ("date", "issued_at", "training_cutoff",
                                                "actual_range_celsius", "predicted_range_celsius")}
        row.update(season=season, event=event)
        for name in annual.METHODS:
            for prefix, pool in (("", full_year), ("seasonal_", seasonal[season])):
                method = prefix + name
                probability, crossings = annual._case_probabilities(point, threshold, pool[name])
                computed = {f"{method}_probability": probability, f"{method}_negative_range_probability": crossings,
                            f"{method}_brier": (probability - event)**2}
                if not prefix and any(value != float(saved[key]) for key, value in computed.items()):
                    raise ValueError(f"Full-year probabilities or scores disagree with the saved run for {day}")
                row.update(computed)
        output.append(row)
    return output


def _summary(rows, *, include_bins):
    methods = {}
    event_rate = fmean(row["event"] for row in rows)
    for method in METHODS:
        value = annual._summarize(rows, method)
        value["mean_probability_minus_event_rate"] = value["mean_probability"] - event_rate
        if not include_bins:
            del value["calibration_bins"]
        methods[method] = value
    return dict(cases=len(rows), observed_events=sum(row["event"] for row in rows),
                observed_event_rate=event_rate, methods=methods,
                mean_paired_brier_differences={
                    "annual_same_day_minus_independent": fmean(row["same_day_brier"] - row["independent_brier"] for row in rows),
                    "seasonal_same_day_minus_independent": fmean(row["seasonal_same_day_brier"] - row["seasonal_independent_brier"] for row in rows),
                    "seasonal_minus_annual_independent": fmean(row["seasonal_independent_brier"] - row["independent_brier"] for row in rows),
                    "seasonal_minus_annual_same_day": fmean(row["seasonal_same_day_brier"] - row["same_day_brier"] for row in rows),
                })


def _residual_summary(pool):
    minimum, maximum = ([float(actual - fitted) for actual, fitted in pool[element]] for element in annual.EXTREMES)
    range_errors = [high - low for low, high in zip(minimum, maximum, strict=True)]
    try:
        error_correlation = correlation(minimum, maximum)
    except StatisticsError:
        error_correlation = None
    return dict(cases=len(minimum), minimum_bias=fmean(minimum), maximum_bias=fmean(maximum),
                error_correlation=error_correlation, range_error_bias=fmean(range_errors),
                range_error_sd=pstdev(range_errors))


def diagnose(study, run, output):
    """Recheck the saved annual scores and describe monthly performance and residuals."""
    study, run, output = map(Path, (study, run, output))
    plan, files, _ = _development_archive(study, run)
    start, end = (date.fromisoformat(plan[field]) for field in ("evaluation_start", "evaluation_end"))
    dates = annual._dates(start, end)
    calibration_dates, calibration = annual._calibration(files[annual.CALIBRATION_TABLE])
    threshold = annual._decimal(plan["threshold_celsius"])

    def cohort_summary(group):
        return dict(events=sum(value >= threshold for value in _ranges(group)), **_residual_summary(group))

    matches = run / "matched-range.csv"
    cases = _score(dates, annual._csv_rows(matches), calibration_dates, calibration, threshold)
    months = {}
    for month in sorted({day.month for day in dates}):
        selected = [row for day, row in zip(dates, cases, strict=True) if day.month == month]
        pool = {element: [row for day, row in zip(calibration_dates, calibration[element], strict=True)
                          if day.month == month] for element in annual.EXTREMES}
        months[f"{month:02d}"] = dict(calibration=cohort_summary(pool), evaluation=_summary(selected, include_bins=False))
    evaluation_residuals = {}
    for element in annual.EXTREMES:
        scored = annual._csv_rows(run / element.lower() / "evaluation.csv")
        if ([datetime.fromisoformat(row["target_time"]).date() for row in scored] != dates
                or any(row["status"] != "passed" for row in scored)):
            raise ValueError(f"Development {element} evaluations must match the range dates")
        evaluation_residuals[element] = [(annual._decimal(row["actual"]), annual._decimal(row["prediction"])) for row in scored]
    for row, minimum, maximum in zip(cases, evaluation_residuals["TMIN"], evaluation_residuals["TMAX"], strict=True):
        if (maximum[0] - minimum[0] != annual._decimal(row["actual_range_celsius"])
                or maximum[1] - minimum[1] != annual._decimal(row["predicted_range_celsius"])):
            raise ValueError(f"Saved scalar forecasts disagree with the range for {row['date']}")
    for month, summary in months.items():
        summary["evaluation_residuals"] = _residual_summary({
            element: [row for day, row in zip(dates, evaluation_residuals[element], strict=True) if day.month == int(month)]
            for element in annual.EXTREMES
        })
    write_json(output, dict(scope="development on previously scored dates", evaluation_start=start.isoformat(),
                evaluation_end=end.isoformat(),
                threshold_celsius=plan["threshold_celsius"], matched_cases_sha256=annual._sha(matches.read_bytes()),
                calendar_seasons=SEASONS, calibration_full_year=cohort_summary(calibration),
                calibration_seasons={name: cohort_summary(pool) for name, pool in _seasonal_pools(calibration_dates, calibration).items()},
                evaluation=_summary(cases, include_bins=True), months=months))
    return output


def freeze(study, run, output, *, start, end):
    """Freeze the annual baseline and seasonal scoring rule before obtaining fresh values."""
    study, run, output = map(Path, (study, run, output))
    original, files, history = _development_archive(study, run)
    archive_start = date.fromisoformat(original["evaluation_end"]) + timedelta(days=1)
    if start < archive_start or end < start:
        raise ValueError("Choose dates strictly after the development cohort")
    if annual._decimal(original["threshold_celsius"]) != 10 or original["score"] != "brier":
        raise ValueError("Seasonal follow-up requires the frozen 10 °C Brier design")
    source_url = urlsplit(annual._archive_url(end))
    query = dict(parse_qsl(source_url.query), startDate=archive_start.isoformat())
    frozen_at = datetime.now(UTC).isoformat()
    baseline = dict(original, evaluation_start=start.isoformat(), evaluation_end=end.isoformat(),
                    frozen_at=frozen_at, source_url=annual._archive_url(end))
    output.mkdir(parents=True, exist_ok=False)
    annual_directory = output / "annual-plan"
    annual_directory.mkdir()
    for name, data in files.items():
        (annual_directory / name).write_bytes(data)
    baseline_path = _write_plan(annual_directory, baseline)
    (output / "development-archive.csv").write_bytes(history)
    return _write_plan(output, dict(format_version=1, frozen_at=frozen_at, station=baseline["station"],
        evaluation_start=start.isoformat(), evaluation_end=end.isoformat(), archive_start=archive_start.isoformat(),
        source_url=source_url._replace(query=urlencode(query)).geturl(), threshold_celsius=baseline["threshold_celsius"],
        score=baseline["score"], methods=METHODS, residual_pool="same meteorological season from 2025",
        calendar_seasons=SEASONS, annual_plan_sha256=annual._sha(baseline_path.read_bytes()),
        development_archive_sha256=annual._sha(history), source_sha256=annual._sha(Path(__file__).read_bytes())))


def _archive_rows(raw, station, start, end):
    for element in annual.EXTREMES:
        noaa.daily_temperatures(raw, station=station, start=start, end=end, element=element)
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        return [{field: row[field] for field in ARCHIVE_FIELDS} for row in csv.DictReader(stream)]


def evaluate(study, source, output):
    """Evaluate four probability methods using the same fresh scalar forecasts."""
    study, source, output = map(Path, (study, source, output))
    raw_plan = (study / "plan.json").read_bytes()
    digest = annual._sha(raw_plan)
    if digest != (study / "plan.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("The frozen seasonal plan changed")
    plan = json.loads(raw_plan)
    if plan["format_version"] != 1 or plan["source_sha256"] != annual._sha(Path(__file__).read_bytes()):
        raise ValueError("The frozen seasonal study source changed")
    baseline, baseline_digest, files = annual._read_plan(study / "annual-plan")
    if baseline_digest != plan["annual_plan_sha256"]:
        raise ValueError("The frozen annual baseline changed")
    history = (study / "development-archive.csv").read_bytes()
    if annual._sha(history) != plan["development_archive_sha256"]:
        raise ValueError("The frozen development archive changed")
    start, end, archive_start = (date.fromisoformat(plan[key]) for key in ("evaluation_start", "evaluation_end", "archive_start"))
    history_rows = _archive_rows(history, plan["station"], annual.ARCHIVE_START, archive_start - timedelta(days=1))
    raw = source.read_bytes()
    later_rows = _archive_rows(raw, plan["station"], archive_start, end)
    output.mkdir(parents=True, exist_ok=False)
    (output / "fresh-archive.csv").write_bytes(raw)
    combined = output / "combined-archive.csv"
    write_csv(combined, ARCHIVE_FIELDS, (*history_rows, *later_rows))
    annual_run = output / "annual"
    annual.evaluate(study / "annual-plan", combined, annual_run)
    calibration_dates, calibration = annual._calibration(files[annual.CALIBRATION_TABLE])
    cases = _score(annual._dates(start, end), annual._csv_rows(annual_run / "matched-range.csv"), calibration_dates,
                   calibration, annual._decimal(baseline["threshold_celsius"]))
    write_csv(output / "matched-range.csv", cases[0], cases)
    summary = output / "summary.json"
    write_json(summary, dict(plan_sha256=digest, fresh_archive_sha256=annual._sha(raw),
        evaluation_start=start.isoformat(), evaluation_end=end.isoformat(), threshold_celsius=plan["threshold_celsius"],
        seasonal_pool_sizes={name: len(pool["TMIN"]) for name, pool in _seasonal_pools(calibration_dates, calibration).items()},
        **_summary(cases, include_bins=True)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("diagnose", "freeze", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--study", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--source", type=Path, required=True)
        else:
            command.add_argument("--run", type=Path, required=True)
        if name == "freeze":
            command.add_argument("--start", type=date.fromisoformat, required=True)
            command.add_argument("--end", type=date.fromisoformat, required=True)
    args = vars(parser.parse_args(argv))
    operation = {"diagnose": diagnose, "freeze": freeze, "evaluate": evaluate}[args.pop("command")]
    print(operation(**args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
