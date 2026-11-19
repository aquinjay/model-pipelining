"""Run the Central Park daily maximum temperature development forecast."""

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.local_level import LocalLevel
from model_pipelining.reporting import plot_run
from model_pipelining.snapshots import register_snapshot


STATION = "USW00094728"
START = date(2023, 1, 1)
END = date(2024, 12, 31)
SOURCE_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1?"
    "dataset=daily-summaries&stations=USW00094728&startDate=2023-01-01&endDate=2024-12-31"
    "&dataTypes=TMAX&units=metric&includeAttributes=true&format=csv"
)


def daily_maxima(source):
    """Validate the requested station, quality flags and full daily calendar."""
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig")))
    columns = {"STATION", "DATE", "TMAX", "TMAX_ATTRIBUTES"}
    if reader.fieldnames is None or not columns.issubset(reader.fieldnames):
        raise ValueError("Expected NOAA station, date, TMAX and TMAX attributes")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Duplicate NOAA CSV column")
    values = {}
    for row in reader:
        if row["STATION"] != STATION:
            raise ValueError(f"Expected NOAA station {STATION}")
        try:
            day = date.fromisoformat(row["DATE"])
            value = float(row["TMAX"])
            flags = row["TMAX_ATTRIBUTES"].split(",")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"Invalid NOAA value at line {reader.line_num}") from error
        if not math.isfinite(value) or value == -9999:
            raise ValueError(f"Missing or invalid TMAX on {day}")
        if len(flags) not in (3, 4) or flags[1].strip():
            raise ValueError(f"Missing or flagged NOAA quality information on {day}")
        if day in values:
            raise ValueError(f"Duplicate NOAA date {day}")
        values[day] = value
    expected = [START + timedelta(days=index) for index in range((END - START).days + 1)]
    if sorted(values) != expected:
        raise ValueError(f"Expected a complete NOAA daily calendar from {START} to {END}")
    return [(f"{day}T00:00:00Z", values[day]) for day in expected]


def run_example(source_path, output_directory):
    source = Path(source_path).read_bytes()
    rows = daily_maxima(source)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "noaa-source.csv"
    raw_path.write_bytes(source)
    prepared = output / "central-park-tmax.csv"
    with prepared.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("timestamp", "value"))
        writer.writerows(rows)

    store = output / "snapshots"
    raw = register_snapshot(raw_path, dataset="noaa-central-park-raw", store=store)
    data = register_snapshot(prepared, dataset="noaa-central-park-tmax", store=store,
                             note=f"Normalized NOAA raw snapshot {raw.snapshot_id}")
    experiment = ExperimentSpec(
        task=ForecastTask(name="central-park-daily-tmax",
            target_description="Next dated observation of daily maximum temperature at Central Park",
            source_description="NOAA GHCN-Daily archive request for USW00094728, using date labels and simulated availability",
            units="degrees Celsius", minimum_history=365, actual_policy="latest_in_snapshot"),
        evaluation=EvaluationPlan(benchmarks=("last-value",)),
    )
    model = LocalLevel(initial_mean=15, initial_variance=100,
                       observation_variance=4, evolution_variance=9)
    design = dict(source_url=SOURCE_URL, raw_dataset=raw.dataset, raw_snapshot_id=raw.snapshot_id,
        dataset=data.dataset, snapshot_id=data.snapshot_id, start=START.isoformat(), end=END.isoformat(),
        station=STATION, rows=len(rows), model=model.configuration,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        availability="Simulate the archived daily value being known at its UTC date label. These labels do not establish real historical publication times.",
        experiment=experiment.to_dict())
    (output / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")

    run = run_backtest(dataset=data.dataset, snapshot_id=data.snapshot_id, store=store,
        experiment=experiment, output_directory=output / "forecast", model=model)
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    if (not scores["complete"] or scores["scored_count"] != 366
            or any(scores[kind]["status"] != "completed"
                   for kind in ("benchmarks", "distribution", "uncertainty"))):
        raise ValueError("Expected complete forecasts, benchmarks and scores for all 366 Central Park cases")
    return plot_run(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local metric NOAA CSV with attributes")
    parser.add_argument("--output-directory", type=Path, required=True)
    options = parser.parse_args()
    print(run_example(options.source, options.output_directory))


if __name__ == "__main__":
    main()
