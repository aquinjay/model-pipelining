"""Execute a declared daily experiment with release-aware, frozen forecasts."""

import csv
from datetime import UTC, datetime, timedelta
import hashlib
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path
import platform
from statistics import mean

from model_pipelining.experiments import ExperimentSpec, ForecastTask, plan_cases, read_task_data, select_actuals
from model_pipelining.models import ForecastRequest, Forecaster, LastValue, Observation, TrainingMean
from model_pipelining.snapshots import retrieve_snapshot
from model_pipelining.tags import resolve_tag
from model_pipelining.uncertainty import (
    evaluate_highest_density, evaluate_intervals, interval_probabilities, required_quantiles, summarize_uncertainty,
    validate_forecast, validate_interval_levels,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_observations(path: Path) -> tuple[Observation, ...]:
    """The original immediate, unrevised daily reader used by diagnostics."""
    return tuple(Observation(item.timestamp, item.value) for item in read_task_data(path, ForecastTask()))


def _evaluate_points(predictions, actuals):
    evaluations = []
    for prediction in predictions:
        actual = actuals[prediction["target_time"]]
        row = {"target_time": prediction["target_time"], "actual": actual.value,
               "actual_available_at": actual.available_at.isoformat(),
               "prediction": prediction["prediction"], "error": None,
               "absolute_error": None, "squared_error": None,
               "status": prediction["status"], "failure": prediction["error"]}
        if "benchmark" in prediction:
            row["benchmark"] = prediction["benchmark"]
        if prediction["status"] == "passed":
            try:
                error = prediction["prediction"] - actual.value
                squared_error = error ** 2
                if not math.isfinite(error) or not math.isfinite(squared_error):
                    raise ValueError("Nonfinite scoring result")
                row.update(error=error, absolute_error=abs(error), squared_error=squared_error)
            except (ValueError, OverflowError) as error:
                row.update(status="failed", failure=f"{type(error).__name__}: {error}")
        evaluations.append(row)
    return evaluations


def _point_scores(evaluations):
    scored = [row for row in evaluations if row["status"] == "passed"]
    return {
        "expected_count": len(evaluations), "scored_count": len(scored),
        "failed_count": len(evaluations) - len(scored), "complete": len(scored) == len(evaluations),
        # statistics.mean avoids overflow of a sum of finite squared errors.
        "mae": mean(row["absolute_error"] for row in scored) if scored else None,
        "rmse": math.sqrt(mean(row["squared_error"] for row in scored)) if scored else None,
        "scope": "Successful cases only; partial scores are not a complete-run comparison",
    }


def run_backtest(
    *,
    dataset: str,
    output_directory: str | Path,
    snapshot_id: str | None = None,
    tag: str | None = None,
    minimum_history: int | None = None,
    experiment: ExperimentSpec | None = None,
    model: Forecaster | None = None,
    interval_levels: tuple[float, ...] = (0.8, 0.95),
    store: str | Path = "data/versioned",
) -> Path:
    """Save a daily experiment in a new directory and return its path.

    With no experiment, use the original immediate/unrevised development task
    and minimum_history=30. With an ExperimentSpec, choose minimum history there
    and omit the legacy argument. Supply exactly one of snapshot_id and tag.
    A tag resolves once; the manifest
    pins its permanent ID. The default model is TrainingMean. Forecast failures
    are saved per origin and execution continues. Failed runs return normally;
    inspect manifest.json's status and checks.json before interpreting scores.

    Models declaring supports_quantiles=True must supply every requested
    observation quantile. Point-only models have no invented uncertainty.
    Invalid input/configuration raises before creating the output directory.
    Filesystem errors propagate; an interrupted directory may remain with status
    running. Existing outputs are never reused. Reporting is a separate call.
    """
    if (snapshot_id is None) == (tag is None):
        raise ValueError("Supply exactly one of snapshot_id or tag")
    if experiment is None:
        experiment = ExperimentSpec(task=ForecastTask(minimum_history=30 if minimum_history is None else minimum_history))
    elif not isinstance(experiment, ExperimentSpec):
        raise ValueError("experiment must be an ExperimentSpec")
    elif minimum_history is not None:
        raise ValueError("Choose minimum_history in the experiment, not both interfaces")
    task = experiment.task
    levels = validate_interval_levels(interval_levels)
    selected_id = resolve_tag(dataset=dataset, tag=tag, store=store) if tag is not None else snapshot_id
    path = retrieve_snapshot(selected_id, dataset=dataset, store=store)
    versions = read_task_data(path, task)
    cases, schedule = plan_cases(versions, experiment)
    selected_model = TrainingMean() if model is None else model
    supports_quantiles = getattr(selected_model, "supports_quantiles", False)
    supports_highest_density = getattr(selected_model, "supports_highest_density", False)
    if type(supports_quantiles) is not bool:
        raise ValueError("model.supports_quantiles must be a boolean")
    if type(supports_highest_density) is not bool:
        raise ValueError("model.supports_highest_density must be a boolean")
    probabilities = required_quantiles(levels) if supports_quantiles else ()
    density_levels = levels if supports_highest_density else ()
    if not isinstance(selected_model.name, str) or not selected_model.name.strip():
        raise ValueError("model.name must be a nonempty string")
    configuration = selected_model.configuration
    if not isinstance(configuration, dict):
        raise ValueError("model.configuration must be a JSON dictionary")
    # Round-trip before execution so a model cannot later mutate saved settings.
    configuration = json.loads(json.dumps(configuration, allow_nan=False))
    source_files = [Path(__file__), Path(__file__).with_name("models.py"),
                    Path(__file__).with_name("uncertainty.py"), Path(__file__).with_name("experiments.py")]
    model_source = inspect.getsourcefile(type(selected_model))
    if model_source is not None:
        source_files.append(Path(model_source))
    source_hashes = {
        str(source.name): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
    }
    manifest = {
        "format_version": 3,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "input": {"dataset": dataset, "snapshot_id": selected_id, "selected_tag": tag},
        "experiment": experiment.to_dict(),
        "experiment_sha256": hashlib.sha256(json.dumps(experiment.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "task": {
            "timestamp_column": task.timestamp_column, "target_column": task.target_column,
            "frequency": "one UTC day", "horizon_days": task.horizon_days,
            "minimum_history": task.minimum_history, "training_window": "expanding",
            "point_statistic": "predictive mean", "evaluation_role": experiment.evaluation.role,
            "availability": ("Each observation is available at its timestamp; no delays or revisions" if task.availability_column is None
                             else f"Explicit {task.availability_column}; revisions={task.revision_policy}"),
            "issuance": ("Immediately after observing the current value" if task.issuance_delay_seconds == 0 and task.availability_column is None
                         else f"Origin period timestamp plus {task.issuance_delay_seconds} seconds; current value need not be available"),
            "units": task.units,
        },
        "model": {"name": selected_model.name, "configuration": configuration,
                  "supports_quantiles": supports_quantiles,
                  "supports_highest_density": supports_highest_density,
                  "class": f"{type(selected_model).__module__}.{type(selected_model).__qualname__}"},
        "evaluation": {"interval_levels": list(levels), "quantile_levels": list(probabilities),
                       "highest_density_levels": list(density_levels),
                       "uncertainty_target": "future_observation",
                       "primary_interval_method": ("highest_density" if supports_highest_density else
                                                   "equal_tail" if supports_quantiles else None),
                       "quantile_loss": "unscaled_pinball", "interval_score": "winkler_equal_tail_only",
                       "highest_density_evaluation": "coverage_and_total_set_width"},
        "environment": {"python": platform.python_version(), "pandas": version("pandas")},
        "source_sha256": source_hashes,
    }
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", manifest)
    _write_csv(output / "case_schedule.csv", schedule)

    predictions = []
    quantile_predictions = []
    quantile_outputs = {}
    density_predictions = []
    density_outputs = {}
    benchmark_models = {name: {"training-mean": TrainingMean, "last-value": LastValue}[name]() for name in experiment.evaluation.benchmarks}
    benchmark_predictions = []
    history_checks_passed = True
    for case in cases:
        history = case.history
        issued_at, target_time = case.issued_at, case.target_time
        # Coordinates come from the task, never from the model. No future value
        # is included in this immutable request.
        row = {
            "origin_index": case.origin_index,
            "issued_at": issued_at.isoformat(), "training_cutoff": issued_at.isoformat(),
            "training_start": history[0].timestamp.isoformat(), "training_count": len(history),
            "training_end": history[-1].timestamp.isoformat(), "history_sha256": case.history_sha256,
            "target_time": target_time.isoformat(), "prediction": None,
            "status": "failed", "error": None,
        }
        try:
            if (
                len(history) < task.minimum_history
                or any(observation.timestamp > issued_at for observation in history)
                or any(item.available_at > issued_at for item in case.versions)
                or issued_at >= target_time
                or target_time - issued_at != timedelta(days=task.horizon_days, seconds=-task.issuance_delay_seconds)
            ):
                history_checks_passed = False
                raise ValueError("Training history or target time violates the forecast task")
            model_output = selected_model.forecast(ForecastRequest(history, issued_at, target_time, probabilities, density_levels))
            forecast = validate_forecast(model_output, probabilities, density_levels)
            row.update(prediction=forecast.mean, status="passed")
            quantile_outputs[row["target_time"]] = {item.probability: item.value for item in forecast.quantiles}
            density_outputs[row["target_time"]] = forecast.highest_density_regions
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        predictions.append(row)
        for name, benchmark in benchmark_models.items():
            benchmark_row = {**row, "benchmark": name, "prediction": None, "status": "failed", "error": None}
            try:
                value = validate_forecast(benchmark.forecast(ForecastRequest(history, issued_at, target_time)), ())
                benchmark_row.update(prediction=value.mean, status="passed")
            except Exception as error:
                benchmark_row["error"] = f"{type(error).__name__}: {error}"
            benchmark_predictions.append(benchmark_row)
        for probability in probabilities:
            quantile_predictions.append({
                "target_time": row["target_time"], "probability": probability,
                "value": quantile_outputs.get(row["target_time"], {}).get(probability),
                "status": row["status"], "error": row["error"],
            })
        regions = {region.level: region for region in density_outputs.get(row["target_time"], ())}
        for level in density_levels:
            region = regions.get(level)
            for index, segment in enumerate(region.segments if region is not None else (None,)):
                density_predictions.append({
                    "target_time": row["target_time"], "level": level,
                    "segment_index": index if segment is not None else None,
                    "lower": segment.lower if segment is not None else None,
                    "upper": segment.upper if segment is not None else None,
                    "status": row["status"], "error": row["error"],
                })

    # Close the forecast file before consulting actual targets for evaluation.
    # It is never rewritten by evaluation or reporting.
    _write_csv(output / "predictions.csv", predictions)
    if supports_quantiles:
        _write_csv(output / "quantiles.csv", quantile_predictions)
    if supports_highest_density:
        _write_csv(output / "highest_density.csv", density_predictions)
    if benchmark_models:
        _write_csv(output / "benchmark_predictions.csv", benchmark_predictions)
    actuals = select_actuals(versions, task)
    evaluations = _evaluate_points(predictions, actuals)
    _write_csv(output / "evaluation.csv", evaluations)
    benchmark_scores = {"status": "not_applicable", "runs": []}
    if benchmark_models:
        benchmark_evaluations = _evaluate_points(benchmark_predictions, actuals)
        _write_csv(output / "benchmark_evaluation.csv", benchmark_evaluations)
        benchmark_scores["runs"] = [{"name": name, **_point_scores([row for row in benchmark_evaluations if row["benchmark"] == name])}
                                    for name in benchmark_models]
        benchmark_scores["status"] = "completed" if all(row["complete"] for row in benchmark_scores["runs"]) else "failed"
    interval_evaluations = []
    density_evaluations = []
    if supports_quantiles or supports_highest_density:
        for prediction in predictions:
            target_time = prediction["target_time"]
            actual = actuals[target_time].value
            quantiles = quantile_outputs.get(target_time, {})
            rows = []
            for level in levels if supports_quantiles else ():
                lower_probability, upper_probability = interval_probabilities(level)
                rows.append({
                    "target_time": target_time, "actual": actual, "level": level,
                    "lower_probability": lower_probability, "upper_probability": upper_probability,
                    "lower": quantiles.get(lower_probability), "upper": quantiles.get(upper_probability),
                    "covered": None, "width": None, "interval_score": None,
                    "lower_pinball_loss": None, "upper_pinball_loss": None,
                    "status": "failed", "failure": prediction["error"],
                })
            region_rows = [{"target_time": target_time, "actual": actual, "level": level,
                            "covered": None, "total_width": None, "segment_count": None,
                            "status": "failed", "failure": prediction["error"]} for level in density_levels]
            if prediction["status"] == "passed":
                try:
                    # Complete all levels before accepting this uncertainty
                    # case, so summaries always refer to the same cases.
                    results = evaluate_intervals(quantiles, actual, levels) if supports_quantiles else []
                    region_results = evaluate_highest_density(density_outputs[target_time], actual) if supports_highest_density else []
                    for row, result in zip(rows, results):
                        row.update(result, status="passed", failure=None)
                    for row, result in zip(region_rows, region_results):
                        row.update(result, status="passed", failure=None)
                except (ValueError, OverflowError) as error:
                    for row in rows + region_rows:
                        row["failure"] = f"{type(error).__name__}: {error}"
            interval_evaluations.extend(rows)
            density_evaluations.extend(region_rows)
    if supports_quantiles:
        _write_csv(output / "interval_evaluation.csv", interval_evaluations)
    if supports_highest_density:
        _write_csv(output / "highest_density_evaluation.csv", density_evaluations)
    scores = {
        **_point_scores(evaluations), "benchmarks": benchmark_scores,
        "uncertainty": summarize_uncertainty(interval_evaluations, levels,
                                             expected_count=len(predictions), supported=supports_quantiles,
                                             highest_density_rows=density_evaluations,
                                             supports_highest_density=supports_highest_density),
    }
    _write_json(output / "scores.json", scores)
    checks = [
        {"name": "input_schema_and_daily_calendar", "required": True, "status": "passed"},
        {"name": "eligible_history_and_target_alignment", "required": True,
         "status": "passed" if history_checks_passed else "failed"},
        {"name": "forecast_outputs_valid", "required": True,
         "status": "passed" if all(row["status"] == "passed" for row in predictions) else "failed"},
        {"name": "all_expected_cases_scored", "required": True,
         "status": "passed" if scores["complete"] else "failed"},
        {"name": "declared_case_plan_and_release_eligibility", "required": True,
         "status": "passed" if history_checks_passed else "failed"},
        {"name": "all_requested_benchmarks_scored", "required": bool(benchmark_models),
         "status": benchmark_scores["status"].replace("completed", "passed")},
        {"name": "requested_observation_quantiles_valid", "required": supports_quantiles,
         "status": ("passed" if all(row["status"] == "passed" for row in predictions) else "failed")
                   if supports_quantiles else "not_applicable"},
        {"name": "requested_highest_density_regions_valid", "required": supports_highest_density,
         "status": ("passed" if all(row["status"] == "passed" for row in predictions) else "failed")
                   if supports_highest_density else "not_applicable"},
        {"name": "all_uncertainty_cases_scored", "required": supports_quantiles or supports_highest_density,
         "status": ("passed" if scores["uncertainty"]["status"] == "completed" else "failed")
                   if supports_quantiles or supports_highest_density else "not_applicable"},
    ]
    _write_json(output / "checks.json", checks)
    _write_json(output / "review.json", {
        "status": "pending", "decision": None,
        "limitations": [f"Declared {experiment.evaluation.role} evaluation; boundary selection does not certify untouched data",
                        "One daily series; release timestamps are caller-supplied claims, not independently verified",
                        "Final evaluation learns from earlier final observations only when available at that origin; settings must remain predeclared",
                        "Quantiles describe future observations conditional on model assumptions",
                        "Highest-density region optimality is the adapter's responsibility, not proven by structural validation",
                        "Coverage and scores do not automatically approve uncertainty calibration",
                        "Check success is not human approval or evidence of real-world usefulness"],
    })
    checks_passed = all(check["status"] == "passed" for check in checks if check["required"])
    manifest.update(status="completed" if checks_passed else "failed", finished_at=datetime.now(UTC).isoformat())
    _write_json(output / "manifest.json", manifest)
    return output
