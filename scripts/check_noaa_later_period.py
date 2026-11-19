"""Freeze the Central Park model comparison, then evaluate a separate NOAA year."""

import argparse
import csv
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.reporting import compare_runs
from model_pipelining.snapshots import register_snapshot, retrieve_snapshot
from scripts.run_noaa_temperature import daily_maxima
from scripts.study_noaa_seasonality import MODELS, TemperatureRegression, _evaluated, _metrics, _write_json


START = date(2025, 1, 1)
END = date(2025, 12, 31)
SOURCE_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1?"
    "dataset=daily-summaries&stations=USW00094728&startDate=2025-01-01&endDate=2025-12-31"
    "&dataTypes=TMAX&units=metric&includeAttributes=true&format=csv"
)
PRINCIPAL = "annual-plus-lag"
CONTROLS = ("lag-only", "last-value")
INPUT_FILES = ("check_noaa_later_period.py", "run_noaa_temperature.py", "study_noaa_seasonality.py")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _script_hashes():
    directory = Path(__file__).resolve().parent
    return {name: _sha((directory / name).read_bytes()) for name in INPUT_FILES}


def _regressions():
    return {name: TemperatureRegression(name, *MODELS[name]) for name in (PRINCIPAL, CONTROLS[0])}


def freeze(development, output):
    """Save the design and historical snapshot before accepting any later data."""
    development, output = Path(development), Path(output)
    raw_design = (development / "design.json").read_bytes()
    design = json.loads(raw_design)
    models = {name: model.configuration for name, model in _regressions().items()}
    if (design["dataset"] != "noaa-central-park-tmax" or design["target_dates"] != ["2024-01-01", "2024-12-31"]
            or any(design["models"][name] != model for name, model in models.items())):
        raise ValueError("Expected the Central Park 2024 development design and its exact regression settings")
    earlier = ExperimentSpec.from_dict(design["experiment"])
    if earlier.evaluation.role != "development" or earlier.evaluation.benchmarks != ("last-value",):
        raise ValueError("Expected the original Central Park development settings")
    original_directory = Path(design["source"])
    input_file = retrieve_snapshot(design["snapshot_id"], dataset=design["dataset"],
                                   store=original_directory / "snapshots")
    history = input_file.read_bytes()
    if _sha(history) != design["snapshot_id"] or not history.endswith(b"\n"):
        raise ValueError("The original normalized history must match its verified snapshot ID")

    first = datetime(2025, 1, 1, tzinfo=UTC)
    exclusive_end = datetime(2026, 1, 1, tzinfo=UTC)
    experiment = replace(earlier, evaluation=replace(earlier.evaluation, role="final",
                         final_start=first, final_end=exclusive_end))
    plan = dict(
        frozen_at=datetime.now(UTC).isoformat(),
        development_history_snapshot=design["snapshot_id"],
        development_design_sha256=_sha(raw_design),
        target_start=str(START), target_end=str(END), cases=(END - START).days + 1,
        experiment=experiment.to_dict(), models=models, target_source_url=SOURCE_URL,
        metrics=["mae", "rmse"], success="Annual-plus-lag must beat both controls on overall MAE and RMSE",
        controls=list(CONTROLS), movement_thresholds_celsius=[2, 5],
        secondary_comparisons="Season and realized change groups are descriptive",
        scripts_sha256=_script_hashes(),
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "history-2023-2024.csv").write_bytes(history)
    (output / "development-design.json").write_bytes(raw_design)
    plan_path = output / "plan.json"
    _write_json(plan_path, plan)
    (output / "plan.sha256").write_text(_sha(plan_path.read_bytes()) + "\n", encoding="utf-8")
    return plan_path


def evaluate(study, source):
    """Use the frozen plan to evaluate all matched targets on the later archive."""
    study, source = Path(study), Path(source)
    plan_bytes = (study / "plan.json").read_bytes()
    if _sha(plan_bytes) != (study / "plan.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("The frozen plan changed")
    plan = json.loads(plan_bytes)
    models = _regressions()
    if (plan["scripts_sha256"] != _script_hashes()
            or any(plan["models"][name] != model.configuration for name, model in models.items())):
        raise ValueError("The frozen scripts or model settings changed")
    history = (study / "history-2023-2024.csv").read_bytes()
    if (_sha(history) != plan["development_history_snapshot"]
            or _sha((study / "development-design.json").read_bytes()) != plan["development_design_sha256"]):
        raise ValueError("The frozen development inputs changed")
    experiment = ExperimentSpec.from_dict(plan["experiment"])
    raw = source.read_bytes()
    later = daily_maxima(raw, start=START, end=END)
    dates = [START + timedelta(days=offset) for offset in range(plan["cases"])]
    output = study / "evaluation"
    output.mkdir(exist_ok=False)
    raw_path = output / "noaa-2025-source.csv"
    raw_path.write_bytes(raw)
    prepared = output / "central-park-tmax-2023-2025.csv"
    prepared.write_bytes(history)
    with prepared.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(later)
    store = output / "snapshots"
    raw_snapshot = register_snapshot(raw_path, dataset="noaa-central-park-2025-raw", store=store)
    data = register_snapshot(prepared, dataset="noaa-central-park-tmax", store=store,
        note=f"Append raw snapshot {raw_snapshot.snapshot_id} to {plan['development_history_snapshot']}")
    _write_json(output / "inputs.json", dict(evaluated_at=datetime.now(UTC).isoformat(),
        plan_sha256=_sha(plan_bytes), prefix_sha256=_sha(history), raw_snapshot_id=raw_snapshot.snapshot_id,
        normalized_snapshot_id=data.snapshot_id))
    with prepared.open(encoding="utf-8", newline="") as stream:
        values = {datetime.fromisoformat(row["timestamp"]).date(): float(row["value"])
                  for row in csv.DictReader(stream)}

    runs = {name: run_backtest(dataset=data.dataset, snapshot_id=data.snapshot_id, store=store,
                experiment=experiment, model=model, output_directory=output / name)
            for name, model in models.items()}
    compare_runs(runs[CONTROLS[0]], runs[PRINCIPAL], output_directory=output / "comparison")
    predictions = {name: _evaluated(run / "evaluation.csv", dates, values)
                   for name, run in runs.items()}
    predictions["last-value"] = _evaluated(runs[PRINCIPAL] / "benchmark_evaluation.csv", dates, values, benchmark=True)
    if any(prediction != values[day - timedelta(days=1)] for day, prediction in zip(dates, predictions["last-value"], strict=True)):
        raise ValueError("Persistence must use the previous dated observation")
    results = _metrics(dates, values, predictions)
    improved = {name: {metric: results[PRINCIPAL]["overall"][metric] < results[name]["overall"][metric]
                       for metric in plan["metrics"]} for name in CONTROLS}
    _write_json(output / "summary.json", dict(cases=len(dates), results=results,
        improved=improved, replicated=all(all(metrics.values()) for metrics in improved.values())))
    with (output / "matched-cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "actual", "previous_actual", *predictions])
        for index, day in enumerate(dates):
            writer.writerow([day, values[day], values[day - timedelta(days=1)],
                             *(forecast[index] for forecast in predictions.values())])
    return output / "summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="command", required=True)
    plan = operations.add_parser("freeze")
    plan.add_argument("--development-directory", type=Path, required=True)
    plan.add_argument("--output-directory", type=Path, required=True)
    run = operations.add_parser("evaluate")
    run.add_argument("--study-directory", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    options = parser.parse_args()
    if options.command == "freeze":
        print(freeze(options.development_directory, options.output_directory))
    else:
        print(evaluate(options.study_directory, options.source))


if __name__ == "__main__":
    main()
