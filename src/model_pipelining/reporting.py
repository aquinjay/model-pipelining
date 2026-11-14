"""Draw a chart from saved evaluation results; never fit or call a model."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from model_pipelining.uncertainty import interval_probabilities


def _read_run(run_directory: str | Path) -> tuple[dict, dict, list[dict]]:
    """Read finished evidence without touching the input store or any model."""
    run = Path(run_directory).expanduser()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] not in (1, 2):
        raise ValueError("Unsupported run format version")
    if manifest["status"] not in ("completed", "failed"):
        raise ValueError("Reporting requires a finished run")
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    with (run / "evaluation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return manifest, scores, rows


def _saved_interval(run: str | Path, manifest: dict, rows: list[dict], level: float | None = None):
    """Read one central interval from frozen quantiles, including failed gaps."""
    if not manifest["model"].get("supports_quantiles", False):
        if level is not None:
            raise ValueError("This saved run has no observation quantiles")
        return None
    levels = manifest["evaluation"]["interval_levels"]
    chosen = max(levels) if level is None else level
    if chosen not in levels:
        raise ValueError("The requested interval level was not saved by this run")
    lower_probability, upper_probability = interval_probabilities(chosen)
    with (Path(run).expanduser() / "quantiles.csv").open(encoding="utf-8", newline="") as stream:
        quantiles = {(row["target_time"], float(row["probability"])): row for row in csv.DictReader(stream)}
    lower, upper = [], []
    for row in rows:
        first = quantiles[(row["target_time"], lower_probability)]
        second = quantiles[(row["target_time"], upper_probability)]
        if first["status"] == second["status"] == "passed":
            first_value, second_value = float(first["value"]), float(second["value"])
            if not math.isfinite(first_value) or not math.isfinite(second_value) or first_value > second_value:
                raise ValueError("Invalid saved interval bounds")
            lower.append(first_value)
            upper.append(second_value)
        else:
            lower.append(float("nan"))
            upper.append(float("nan"))
    return chosen, lower, upper


def _draw_uncertainty(axes, times, run, manifest, rows, *, level=None, method="auto", prefix="", color=None):
    """Prefer supplied highest-density regions; never fill their internal gaps."""
    if method not in ("auto", "highest_density", "equal_tail"):
        raise ValueError("interval_method must be auto, highest_density, or equal_tail")
    model = manifest["model"]
    if method == "auto":
        if model.get("supports_highest_density", False):
            method = "highest_density"
        elif model.get("supports_quantiles", False):
            method = "equal_tail"
        elif level is None:
            return
        else:
            raise ValueError("This saved run has no observation uncertainty")
    if method == "equal_tail":
        if not model.get("supports_quantiles", False):
            raise ValueError("This saved run has no observation quantiles")
        chosen, lower, upper = _saved_interval(run, manifest, rows, level)
        axes.fill_between(times, lower, upper, alpha=0.15, color=color,
                          label=f"{prefix}{100 * chosen:g}% equal-tail prediction interval")
        return
    if not model.get("supports_highest_density", False):
        raise ValueError("This saved run has no highest-density prediction regions")
    levels = manifest["evaluation"]["highest_density_levels"]
    chosen = max(levels) if level is None else level
    if chosen not in levels:
        raise ValueError("The requested interval level was not saved by this run")
    regions = {row["target_time"]: {} for row in rows}
    with (Path(run).expanduser() / "highest_density.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["level"]) != chosen or row["status"] != "passed":
                continue
            lower, upper = float(row["lower"]), float(row["upper"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("Invalid saved highest-density segment")
            regions[row["target_time"]][int(row["segment_index"])] = (lower, upper)
    for index in range(max((len(segments) for segments in regions.values()), default=0)):
        pairs = [regions[row["target_time"]].get(index, (float("nan"), float("nan"))) for row in rows]
        axes.fill_between(times, [pair[0] for pair in pairs], [pair[1] for pair in pairs], alpha=0.15, color=color,
                          label=f"{prefix}{100 * chosen:g}% highest-density prediction region" if index == 0 else None)


def plot_run(
    run_directory: str | Path, *, interval_level: float | None = None, interval_method: str = "auto",
) -> Path:
    """Save forecast.png from an existing run, refusing to overwrite a chart.

    Failed predictions appear as gaps. The title includes run status and scored
    coverage so a partial run is not presented as a complete success. Draw the
    widest saved highest-density region by default, falling back to equal-tail
    intervals only when highest-density output is unavailable.
    """
    run = Path(run_directory).expanduser()
    manifest, scores, rows = _read_run(run)
    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    times = [datetime.fromisoformat(row["target_time"]) for row in rows]
    axes.plot(times, [float(row["actual"]) for row in rows], label="Actual", linewidth=1)
    axes.plot(times, [float(row["prediction"]) if row["prediction"] else float("nan")
                      for row in rows], label="Forecast", linewidth=1.5)
    _draw_uncertainty(axes, times, run, manifest, rows, level=interval_level, method=interval_method, color="C1")
    case_counts = f"points {scores['scored_count']}/{scores['expected_count']}"
    uncertainty = scores.get("uncertainty", {})
    if uncertainty.get("status") not in (None, "not_applicable"):
        case_counts += f"; uncertainty {uncertainty['scored_count']}/{uncertainty['expected_count']}"
    axes.set(title=f"{manifest['model']['name']} — {manifest['status']} ({case_counts})",
             xlabel="Target date (UTC)", ylabel="Value (input units)")
    axes.legend()
    figure.autofmt_xdate()
    destination = run / "forecast.png"
    with destination.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return destination


def compare_runs(
    first_run: str | Path,
    second_run: str | Path,
    *,
    output_directory: str | Path,
) -> Path:
    """Compare complete runs on identical inputs and cases; support old point runs.

    Save comparison.json and comparison.png in a new directory and return its
    absolute path. Require the same permanent input, task, training selections,
    target dates, and actuals. Refuse failed/partial runs rather than comparing
    selectively successful cases. Source artifacts are never rewritten.
    """
    manifests, scores, evaluations, coordinates = [], [], [], []
    for run in (first_run, second_run):
        manifest, score, rows = _read_run(run)
        if (
            manifest["status"] != "completed" or not score["complete"]
            or score["failed_count"] != 0 or not rows
            or score["scored_count"] != len(rows)
            or score["expected_count"] != len(rows)
            or any(row["status"] != "passed" for row in rows)
        ):
            raise ValueError("Comparisons require two complete runs with all cases scored")
        with (Path(run).expanduser() / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        keys = ("origin_index", "issued_at", "training_cutoff", "training_start", "training_count", "target_time")
        coordinates.append([tuple(row[key] for key in keys) for row in predictions])
        if (
            [row["target_time"] for row in predictions] != [row["target_time"] for row in rows]
            or len({row["target_time"] for row in rows}) != len(rows)
        ):
            raise ValueError("Saved forecast and evaluation cases do not align")
        manifests.append(manifest)
        scores.append(score)
        evaluations.append(rows)
    first_input, second_input = (manifest["input"] for manifest in manifests)
    if (
        (first_input["dataset"], first_input["snapshot_id"])
        != (second_input["dataset"], second_input["snapshot_id"])
        or manifests[0]["task"] != manifests[1]["task"]
        or coordinates[0] != coordinates[1]
        or [row["actual"] for row in evaluations[0]] != [row["actual"] for row in evaluations[1]]
    ):
        raise ValueError("Comparison inputs, tasks, training selections, and actuals must match")

    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    times = [datetime.fromisoformat(row["target_time"]) for row in evaluations[0]]
    axes.plot(times, [float(row["actual"]) for row in evaluations[0]], label="Actual", linewidth=1, alpha=0.65)
    for index, (manifest, score, rows) in enumerate(zip(manifests, scores, evaluations), start=1):
        line, = axes.plot(times, [float(row["prediction"]) for row in rows], linewidth=1.5,
                          label=f"{index}: {manifest['model']['name']}")
        _draw_uncertainty(axes, times, (first_run, second_run)[index - 1], manifest, rows,
                          prefix=f"{index}: ", color=line.get_color())
    axes.set(title=f"Development comparison — {len(times)} common forecast dates",
             xlabel="Target date (UTC)", ylabel="Value (input units)")
    axes.legend()
    figure.autofmt_xdate()
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    uncertainty = {"status": "not_applicable", "intervals": [], "highest_density_regions": [],
                   "reason": "Both runs must supply complete uncertainty with common methods and levels"}
    uncertainty_scores = [score.get("uncertainty", {}) for score in scores]
    if all(score.get("status") == "completed" for score in uncertainty_scores):
        for kind in ("highest_density_regions", "intervals"):
            first_levels = {row["level"]: row for row in uncertainty_scores[0].get(kind, [])}
            second_levels = {row["level"]: row for row in uncertainty_scores[1].get(kind, [])}
            common = sorted(first_levels.keys() & second_levels.keys())
            uncertainty[kind] = [{"level": level, "runs": [first_levels[level], second_levels[level]]}
                                 for level in common]
        if uncertainty["intervals"] or uncertainty["highest_density_regions"]:
            uncertainty["status"] = "completed"
            uncertainty.pop("reason")
    summary = {
        "format_version": 2,
        "input": {"dataset": first_input["dataset"], "snapshot_id": first_input["snapshot_id"]},
        "task": manifests[0]["task"],
        "case_count": len(times),
        "runs": [{"path": str(Path(run).expanduser().absolute()), "model": manifest["model"], "scores": score}
                 for run, manifest, score in zip((first_run, second_run), manifests, scores)],
        "uncertainty_comparison": uncertainty,
        "interpretation": "Development comparison only; no automatic selection, tuning, or human approval",
    }
    with (output / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "comparison.png").open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return output
