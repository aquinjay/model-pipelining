"""Local-level assumption influence and sequential model criticism.

These are development diagnostics, separate from backtest scores and approval.
Strong influence is evidence to interpret, never a computational failure.
"""

import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist, StatisticsError, correlation, mean

from model_pipelining.artifacts import write_json
from model_pipelining.experiments import read_observations
from model_pipelining.local_level import LocalLevel
from model_pipelining.snapshots import retrieve_snapshot
from model_pipelining.tags import resolve_tag


_PATH_FIELDS = (
    "scenario", "observation_index", "timestamp", "observation", "observation_mean",
    "observation_variance", "innovation", "standardized_innovation", "pit",
    "update_gain", "level_mean", "level_sd", "next_observation_sd",
    "initial_mean_weight", "level_mean_difference", "level_sd_difference",
    "next_observation_sd_difference",
)


def _path(history, model, name):
    rows = []
    weight = 1.0
    previous_variance = model.initial_variance
    for index, (observation, step) in enumerate(zip(history, model.filter_history(history)), start=1):
        gain = (previous_variance + model.evolution_variance) / step.observation_variance
        # In exact arithmetic V/Q = 1-K. This avoids cancellation at K near 1.
        weight *= model.observation_variance / step.observation_variance
        standardized = step.innovation / math.sqrt(step.observation_variance)
        row = {
            "scenario": name, "observation_index": index, "timestamp": step.timestamp.isoformat(),
            "observation": observation.value, "observation_mean": step.observation_mean,
            "observation_variance": step.observation_variance, "innovation": step.innovation,
            "standardized_innovation": standardized, "pit": NormalDist().cdf(standardized),
            "update_gain": gain, "level_mean": step.level_mean, "level_sd": math.sqrt(step.level_variance),
            "next_observation_sd": math.sqrt(step.level_variance + model.evolution_variance + model.observation_variance),
            "initial_mean_weight": weight,
        }
        if not all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
            raise ValueError("Nonfinite diagnostic calculation")
        rows.append(row)
        previous_variance = step.level_variance
    return rows


def _criticism(rows):
    residuals = [row["standardized_innovation"] for row in rows]
    try:
        lag_one = correlation(residuals[:-1], residuals[1:]) if len(residuals) >= 3 else None
    except StatisticsError:
        lag_one = None
    result = {
        "mean_standardized_innovation": mean(residuals),
        "mean_squared_standardized_innovation": mean(value ** 2 for value in residuals),
        "lag_one_correlation": lag_one,
        "lag_one_unavailable_reason": "Need at least three observations and nonconstant lagged sequences" if lag_one is None else None,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
        raise ValueError("Nonfinite criticism summary")
    return result


def run_local_level_diagnostics(
    *, dataset: str, reference: LocalLevel, scenarios: dict[str, LocalLevel],
    output_directory: str | Path, snapshot_id: str | None = None,
    tag: str | None = None, store: str | Path = "data/versioned",
) -> Path:
    """Save full forward update paths and criticism for caller-chosen settings.

    Supply exactly one input selector. Scenario names are nonblank strings other
    than 'reference'; models must be LocalLevel instances (not subclasses).
    No settings are selected/tuned automatically. Input/options are validated
    before output creation. Numerical failures are retained per configuration;
    inspect diagnostics.json.status. I/O errors propagate. Plotting is separate.
    """
    if (snapshot_id is None) == (tag is None):
        raise ValueError("Supply exactly one of snapshot_id or tag")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a nonempty dictionary of named LocalLevel models")
    if any(not isinstance(name, str) or not name.strip() or name == "reference" for name in scenarios):
        raise ValueError("Scenario names must be nonblank strings other than reference")
    models = {"reference": reference, **{name: scenarios[name] for name in sorted(scenarios)}}
    if any(type(model) is not LocalLevel for model in models.values()):
        raise ValueError("All configurations must be LocalLevel instances, not subclasses")
    selected_id = resolve_tag(dataset=dataset, tag=tag, store=store) if tag is not None else snapshot_id
    history = read_observations(retrieve_snapshot(selected_id, dataset=dataset, store=store))
    if not history:
        raise ValueError("Diagnostics require at least one observation")
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "kind": "local_level_diagnostics", "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "input": {"dataset": dataset, "snapshot_id": selected_id, "selected_tag": tag},
        "scope": {
            "role": "development model diagnostics; not backtest scoring or model approval",
            "selection": "All observations, including initialization; forward filtering only",
            "availability": "Immediate observation availability; no delays or revisions",
            "initial_mean_weight": "Derivative of updated level mean with respect to initial mean, holding all variance settings fixed; not total prior influence",
            "criticism": "Pre-observation sequential innovations and conditional normal PIT; no smoothing or final-data posterior replications",
            "policy": "Configuration agreement, prior influence, and residual patterns have no pass/fail thresholds",
        },
        "observation_count": len(history), "python": platform.python_version(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("diagnostics.py", "local_level.py", "experiments.py", "ingest.py", "artifacts.py")},
    }
    write_json(output / "diagnostics.json", manifest)
    paths, summaries = {}, []
    for name, model in models.items():
        changed = [key for key, value in model.configuration.items() if value != reference.configuration[key]]
        summary = {
            "name": name, "configuration": model.configuration,
            "changed_prior_settings": [key for key in changed if key.startswith("initial_")],
            "changed_noise_settings": [key for key in changed if not key.startswith("initial_")],
            "status": "failed", "error": None, "criticism": None, "influence": None,
        }
        try:
            rows = _path(history, model, name)
            criticism = _criticism(rows)
            reference_rows = rows if name == "reference" else paths.get("reference")
            for index, row in enumerate(rows):
                baseline = reference_rows[index] if reference_rows is not None else None
                for key in ("level_mean", "level_sd", "next_observation_sd"):
                    difference = row[key] - baseline[key] if baseline is not None else None
                    if difference is not None and not math.isfinite(difference):
                        raise ValueError("Nonfinite difference from reference")
                    row[f"{key}_difference"] = difference
            influence = None
            if rows[0]["level_mean_difference"] is not None:
                influence = {f"maximum_absolute_{key}_difference": max(abs(row[f"{key}_difference"]) for row in rows)
                             for key in ("level_mean", "level_sd", "next_observation_sd")}
                influence["final_level_mean_difference"] = rows[-1]["level_mean_difference"]
            paths[name] = rows
            summary.update(status="completed", criticism=criticism, influence=influence)
        except (ValueError, OverflowError) as error:
            summary["error"] = f"{type(error).__name__}: {error}"
        summaries.append(summary)
    with (output / "update_paths.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PATH_FIELDS)
        writer.writeheader()
        writer.writerows(row for rows in paths.values() for row in rows)
    write_json(output / "scenario_summaries.json", summaries)
    write_json(output / "review.json", {
        "status": "pending", "decision": None,
        "questions": ["How do the stated assumptions shape the update path and uncertainty?",
                      "Which effects matter for the intended question, and why?",
                      "What sequential prediction discrepancies merit further investigation?"],
        "limitations": ["Influence is descriptive; smaller influence is not an objective",
                        "Initial mean weight is conditional on all variance settings; it is not total prior influence",
                        "Initial variance and noise changes alter observation gains and uncertainty throughout the path",
                        "Sequential diagnostics include initialization and are not an untouched final evaluation",
                        "No parameter uncertainty, posterior predictive replication, or automatic model selection"],
    })
    manifest.update(status="completed" if len(paths) == len(models) else "failed",
                    finished_at=datetime.now(UTC).isoformat(), expected_configurations=len(models),
                    completed_configurations=len(paths), failed_configurations=len(models) - len(paths))
    # The initial manifest is the only artifact updated during execution.
    write_json(output / "diagnostics.json", manifest, overwrite=True)
    return output


def plot_local_level_diagnostics(diagnostics_directory: str | Path) -> tuple[Path, Path]:
    """Draw sensitivity and innovation charts from saved diagnostics only.

    Failed configurations are named in titles and have no invented paths. Both
    output paths must be unused. No input retrieval or model calls occur here.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(diagnostics_directory).expanduser()
    manifest = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "local_level_diagnostics"
            or manifest.get("status") not in ("completed", "failed")):
        raise ValueError("Plotting requires finished version-1 local-level diagnostics")
    destinations = (directory / "sensitivity.png", directory / "innovations.png")
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Diagnostic charts already exist")
    summaries = json.loads((directory / "scenario_summaries.json").read_text(encoding="utf-8"))
    with (directory / "update_paths.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    figures = [Figure(figsize=(11, 10), layout="constrained"), Figure(figsize=(11, 9), layout="constrained")]
    for figure in figures:
        FigureCanvasAgg(figure)
    axes = figures[0].subplots(4, 1, sharex=True)
    residual_axes = figures[1].subplots(3, 1)
    failed = [item["name"] for item in summaries if item["status"] == "failed"]
    suffix = f"; failed configurations: {', '.join(failed)}" if failed else ""
    figures[0].suptitle(f"Assumption influence — {manifest['status']}{suffix}")
    figures[1].suptitle(f"Sequential model criticism — {manifest['status']}{suffix}")
    observed = next((item["name"] for item in summaries if item["status"] == "completed"), None)
    if observed is not None:
        actuals = [row for row in rows if row["scenario"] == observed]
        axes[0].plot([int(row["observation_index"]) for row in actuals],
                     [float(row["observation"]) for row in actuals], color="0.7", alpha=0.6, label="Observations")
    for item in summaries:
        if item["status"] != "completed":
            continue
        selected = [row for row in rows if row["scenario"] == item["name"]]
        times = [int(row["observation_index"]) for row in selected]
        line, = axes[0].plot(times, [float(row["level_mean"]) for row in selected], label=item["name"])
        color = line.get_color()
        for axis, key in zip(axes[1:], ("level_mean_difference", "level_sd", "initial_mean_weight")):
            axis.plot(times, [float(row[key]) if row[key] else float("nan") for row in selected], color=color)
        standardized = [float(row["standardized_innovation"]) for row in selected]
        residual_axes[0].plot(times, standardized, color=color, label=item["name"])
        residual_axes[1].hist([float(row["pit"]) for row in selected], bins=10, range=(0, 1),
                              histtype="step", density=True, color=color, label=item["name"])
        residual_axes[2].scatter(standardized[:-1], standardized[1:], s=10, alpha=0.5, color=color)
    for axis, label in zip(axes, ("Updated level mean", "Mean difference\nfrom reference", "Updated level SD", "Initial mean weight\n(conditional derivative)")):
        axis.set_ylabel(label)
    axes[1].axhline(0, color="0.5", linewidth=0.7)
    axes[3].set_yscale("symlog", linthresh=1e-12)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([0, 1e-9, 1e-6, 1e-3, 1], ["0", "$10^{-9}$", "$10^{-6}$", "$10^{-3}$", "1"])
    axes[-1].set_xlabel("Observations assimilated (includes initialization)")
    residual_axes[0].axhline(0, color="0.5", linewidth=0.7)
    residual_axes[0].set(xlabel="Observation index", ylabel="Standardized innovation")
    residual_axes[1].axhline(1, color="0.5", linewidth=0.7, linestyle="--")
    residual_axes[1].set(xlabel="Conditional normal PIT", ylabel="Density")
    residual_axes[2].set(xlabel="Previous standardized innovation", ylabel="Next standardized innovation")
    if observed is not None:
        axes[0].legend()
        residual_axes[0].legend()
    for figure, destination in zip(figures, destinations):
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=150)
    return destinations
