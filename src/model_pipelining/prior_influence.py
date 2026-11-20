"""Descriptive prior comparisons on identical, historically eligible histories."""

import csv
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform


from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.evidence import get_evidence_provider, saved_presentation, source_hashes
from model_pipelining.experiments import (ExperimentSpec, ForecastCase,
    select_history_as_of, select_development_history as _select_development_history)
from model_pipelining.models import Forecaster
from model_pipelining.uncertainty import interval_probabilities


_QUANTITIES = (
    "mean_location", "mean_precision", "shape", "scale", "degrees_of_freedom", "prior_mean_weight",
    "mean_lower", "mean_upper", "mean_variance", "mean_moment_status",
    "observation_variance_mean", "observation_variance_median", "variance_lower", "variance_upper",
    "variance_moment_status", "predictive_mean", "predictive_variance", "predictive_scale",
    "predictive_lower", "predictive_upper", "predictive_width", "predictive_moment_status",
    "prior_scale_component", "data_scatter_component", "prior_data_disagreement_component",
    "state_interval_method", "predictive_intervals_json", "predictive_segment_count",
    "quadrature_nodes", "quadrature_tail_bound",
    "evolution_variance_mean", "evolution_variance_median", "evolution_variance_lower", "evolution_variance_upper",
    "evolution_probabilities_json", "variance_covariance",
)
_DIFFERENCES = ("mean_location", "observation_variance_median", "observation_variance_mean",
                "predictive_variance", "predictive_lower", "predictive_upper", "predictive_width",
                "evolution_variance_mean", "evolution_variance_median")
_FIELDS = ("configuration", "update_index", "stage", "issued_at", "history_count", "history_sha256",
           "status", "error", *_QUANTITIES, "comparison_status",
           *(f"{key}_difference" for key in _DIFFERENCES))


def _summary(model, case, level):
    result = get_evidence_provider(model, "prior_influence").summarize(case, level)
    required = ("mean_location", "observation_variance_median", "predictive_lower",
                "predictive_upper", "predictive_width")
    if (not isinstance(result, dict) or set(result) - {*_QUANTITIES, "_integration_evidence"}
            or any(not isinstance(result.get(name), (int, float)) or isinstance(result[name], bool)
                   or not math.isfinite(result[name]) for name in required)
            or result["observation_variance_median"] <= 0 or result["predictive_width"] <= 0
            or result["predictive_lower"] >= result["predictive_upper"]):
        raise ValueError("Malformed prior summary from evidence provider")
    json.dumps(result, allow_nan=False)
    return result


def _learning_cases(versions, specification, end_case):
    periods = tuple(dict.fromkeys(item.timestamp for item in versions if item.timestamp <= end_case.issued_at))
    cases, schedule = [], []
    task = specification.task
    for index, period in enumerate(periods):
        issuance = period + timedelta(seconds=task.issuance_delay_seconds)
        if issuance > end_case.issued_at:
            continue
        history = select_history_as_of(versions, issued_at=issuance, period_end=period)
        selected = bool(history)
        schedule.append({"issued_at": issuance.isoformat(), "period_end": period.isoformat(),
                         "history_count": len(history), "selection": "selected" if selected else "no_available_history"})
        if selected:
            cases.append(ForecastCase(index, issuance, issuance+timedelta(days=task.horizon_days), history))
    # Explicit cutoff can include releases after the last daily issuance.
    if not cases or cases[-1].issued_at != end_case.issued_at:
        cases.append(end_case)
        schedule.append({"issued_at": end_case.issued_at.isoformat(), "period_end": end_case.issued_at.isoformat(),
                         "history_count": len(end_case.history), "selection": "explicit_cutoff"})
    return tuple(cases), schedule


def run_prior_influence(*, dataset: str, reference: Forecaster,
                       scenarios: dict[str, Forecaster], issued_at: datetime,
                       output_directory: str | Path, snapshot_id: str | None = None,
                       tag: str | None = None, store: str | Path = "data/versioned",
                       experiment: ExperimentSpec | None = None, interval_level: float = 0.95) -> Path:
    """Compare explicit priors from the empty-history prior through a development cutoff.

    The provider validates which prior settings may vary across models. Every
    configuration uses identical as-of histories, including eligible revisions.
    Early paths start with the first available observation; minimum history is
    enforced at the final cutoff. No tuning, ranking or influence threshold is
    introduced. Numerical failures are saved per update; later updates/configs
    continue. I/O errors propagate. Plotting is a separate explicit operation.
    """
    interval_probabilities(interval_level)
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a nonempty dictionary of named models")
    if any(not isinstance(name, str) or not name.strip() or name == "reference" for name in scenarios):
        raise ValueError("Scenario names must be nonblank strings other than reference")
    models = {"reference": reference, **{name: scenarios[name] for name in sorted(scenarios)}}
    provider = get_evidence_provider(reference, "prior_influence")
    provider.validate_scenarios(tuple(models.values()))
    metadata = provider.metadata
    specification, versions, end_case, context = _select_development_history(dataset=dataset, issued_at=issued_at,
        snapshot_id=snapshot_id, tag=tag, store=store, experiment=experiment)
    cases, schedule = _learning_cases(versions, specification, end_case)
    context = {**context, "unused_forecast_settings": "Actual vintage and benchmarks do not score comparisons; horizon supplies future request coordinates only",
               "selection": "Daily period+issuance_delay as-of histories plus explicit final cutoff, before final_start; minimum_history applies at cutoff only"}
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "kind": f"{metadata.family}_prior_influence", "family": metadata.family, "presentation": metadata.presentation, "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "context": context, "interval_level": float(interval_level),
        "configurations": {name: model.configuration for name, model in models.items()},
        "expected_updates_per_configuration": len(cases)+1,
        "scope": {
            **metadata.influence_scope,
            "stage_zero": "Empty-history joint prior, before any observations",
            "histories": "Identical eligible releases per issuance; refit from scratch after revisions",
            "policy": "No prior influence, interval agreement, predictive performance, tuning or approval threshold",
        },
        "environment": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy")},
        "source_sha256": source_hashes(*(Path(__file__).with_name(name) for name in
            ("prior_influence.py", "evidence.py", "experiments.py", "ingest.py", "models.py", "uncertainty.py", "artifacts.py")),
            *metadata.implementation_files),
    }
    write_json(output / "prior_influence.json", manifest)
    write_csv(output / "learning_schedule.csv", ("issued_at", "period_end", "history_count", "selection"), schedule)
    write_csv(output / "selected_histories.csv", ("update_index", "timestamp", "available_at", "value"),
               ({"update_index": update, "timestamp": item.timestamp.isoformat(),
                 "available_at": item.available_at.isoformat(), "value": item.value}
                for update, case in enumerate(cases, start=1) for item in case.versions))
    paths, summaries, reference_rows, integration_checks = [], [], {}, []
    for name, model in models.items():
        rows = []
        for index, case in enumerate((None, *cases)):
            row = {field: None for field in _FIELDS}
            row.update(configuration=name, update_index=index, stage="prior" if case is None else "posterior",
                       issued_at=case.issued_at.isoformat() if case else None,
                       history_count=len(case.history) if case else 0,
                       history_sha256=case.history_sha256 if case else None, status="failed")
            try:
                summary = _summary(model, case, interval_level)
                if "_integration_evidence" in summary:
                    integration_checks.append({"configuration": name, "update_index": index, **summary.pop("_integration_evidence")})
                row.update(summary)
                row["status"] = "completed"
            except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                if getattr(error, "evidence", None) is not None:
                    integration_checks.append({"configuration": name, "update_index": index,
                                               "status": "failed", "quadrature": error.evidence})
            baseline = row if name == "reference" else reference_rows[index]
            if row["status"] != "completed":
                row["comparison_status"] = "configuration_unavailable"
            elif baseline["status"] != "completed":
                row["comparison_status"] = "reference_unavailable"
            else:
                try:
                    differences = {f"{key}_difference": row[key]-baseline[key] if row[key] is not None and baseline[key] is not None else None
                                   for key in _DIFFERENCES}
                    if any(x is not None and not math.isfinite(x) for x in differences.values()):
                        raise ValueError("Unrepresentable difference from reference")
                    row.update(differences, comparison_status="available")
                except (ValueError, OverflowError) as error:
                    row.update(status="failed", comparison_status="difference_failed", error=f"{type(error).__name__}: {error}")
            if name == "reference":
                reference_rows[index] = row
            rows.append(row)
        changed = [key for key, value in model.configuration.items() if value != reference.configuration[key]]
        summaries.append({"name": name, "configuration": model.configuration, "changed_settings": changed,
                          "changed_mean_prior_settings": [key for key in changed if key in metadata.mean_prior_settings],
                          "changed_variance_prior_settings": [key for key in changed if key in metadata.variance_prior_settings],
                          "status": "completed" if all(row["status"] == "completed" for row in rows) else "failed",
                          "completed_updates": sum(row["status"] == "completed" for row in rows),
                          "failed_updates": sum(row["status"] == "failed" for row in rows),
                          "final": rows[-1],
                          "maximum_absolute_differences": {key: max((abs(row[f"{key}_difference"]) for row in rows if row[f"{key}_difference"] is not None), default=None)
                                                           for key in _DIFFERENCES}})
        paths.extend(rows)
    write_csv(output / "learning_paths.csv", _FIELDS, paths)
    write_json(output / "configuration_summaries.json", summaries)
    if metadata.integration_checks:
        write_json(output/"integration_checks.json", integration_checks)
    passed = all(summary["status"] == "completed" for summary in summaries)
    write_json(output / "checks.json", [
        {"name": "development_history_selection", "required": True, "status": "passed"},
        {"name": "all_requested_prior_and_posterior_updates", "required": True, "status": "passed" if passed else "failed"},
    ])
    write_json(output / "review.json", {
        "status": "pending", "decision": None,
        "questions": ["How do mean location/precision and variance shape/scale alter learning and prediction?",
                      "Are observed differences meaningful in the target units and at relevant sample sizes?",
                      "Are the chosen priors plausible given their predictive implications?"],
        "limitations": ["Strong or persistent influence is not a computational failure or a target to minimize",
                        "The prior-location coefficient is conditional and does not measure total prior influence",
                        "Variance parameter bounds are equal-tail credible intervals, not HPD prediction intervals",
                        "No automatic tuning, ranking, model approval or held-out evaluation is performed"],
    })
    manifest.update(status="completed" if passed else "failed", finished_at=datetime.now(UTC).isoformat(),
                    completed_configurations=sum(summary["status"] == "completed" for summary in summaries),
                    failed_configurations=sum(summary["status"] == "failed" for summary in summaries))
    write_json(output / "prior_influence.json", manifest, overwrite=True)
    return output


def plot_prior_influence(comparison_directory: str | Path) -> tuple[Path, ...]:
    """Draw saved learning paths/differences without input retrieval or model calls."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(comparison_directory).expanduser()
    manifest = json.loads((directory / "prior_influence.json").read_text(encoding="utf-8"))
    presentation = saved_presentation(manifest, "prior_influence")
    destinations = (directory / "prior_learning.png", directory / "prior_differences.png")
    if presentation.get("evolution_variance",False):
        destinations += (directory/"evolution_priors.png",)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Prior influence charts already exist")
    summaries = json.loads((directory / "configuration_summaries.json").read_text(encoding="utf-8"))
    with (directory / "learning_paths.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    figures = [Figure(figsize=(12, 12), layout="constrained"), Figure(figsize=(11, 9), layout="constrained")]
    for figure in figures:
        FigureCanvasAgg(figure)
    axes = figures[0].subplots(3, 2, sharex=True)
    differences = figures[1].subplots(3, 1, sharex=True)
    failed = [item["name"] for item in summaries if item["status"] == "failed"]
    suffix = f"; failed: {', '.join(failed)}" if failed else ""
    level = 100*manifest["interval_level"]
    interval_title = f"{level:g}% " + presentation["intervals"]
    figures[0].suptitle(f"Prior influence — computation {manifest['status']}{suffix}\n{interval_title}; review pending")
    figures[1].suptitle(f"Signed differences from reference — {manifest['status']}{suffix}\n"
                        "Descriptive differences; no agreement or influence threshold")
    for item in summaries:
        selected = [row for row in rows if row["configuration"] == item["name"]]
        times = [int(row["update_index"]) for row in selected]
        def values(key):
            result = []
            for row in selected:
                raw = row[key] if row["status"] == "completed" else ""
                if raw and not math.isfinite(float(raw)):
                    raise ValueError("Nonfinite saved learning-path value")
                result.append(float(raw) if raw else float("nan"))
            return result
        line, = axes[0, 0].plot(times, values("mean_location"), label=item["name"])
        color = line.get_color()
        axes[0, 0].fill_between(times, values("mean_lower"), values("mean_upper"), color=color, alpha=0.07)
        axes[0, 1].plot(times, values("observation_variance_median"), color=color)
        axes[0, 1].fill_between(times, values("variance_lower"), values("variance_upper"), color=color, alpha=0.07)
        axes[1, 0].plot(times, values("predictive_width"), color=color)
        axes[1, 1].plot(times, values("prior_mean_weight"), color=color)
        axes[2, 0].plot(times, values("mean_location"), color=color)
        if presentation["predictive_segments"]:
            regions = [json.loads(row["predictive_intervals_json"]) if row["status"] == "completed" else [] for row in selected]
            for segment in range(max(map(len, regions), default=0)):
                bounds = [[region[segment][key] if len(region) > segment else float("nan") for region in regions]
                          for key in ("lower", "upper")]
                axes[2, 0].fill_between(times, *bounds, color=color, alpha=0.07)
        else:
            axes[2, 0].fill_between(times, values("predictive_lower"), values("predictive_upper"), color=color, alpha=0.07)
        axes[2, 1].plot(times, values("degrees_of_freedom"), color=color)
        if presentation.get("evolution_variance",False):
            axes[1,1].plot(times,values("evolution_variance_mean"),color=color)
            axes[2,1].plot(times,values("variance_covariance"),color=color)
        for axis, key in zip(differences, ("mean_location", "observation_variance_median", "predictive_width")):
            axis.plot(times, values(f"{key}_difference"), color=color, label=item["name"])
    labels = (("Mean location (target units)", "Variance median (squared units)"),
              ("Predictive HPD width (target units)", "Prior location coefficient k0/(k0+n)"),
              ("Observation forecast (target units)", "Student-t degrees of freedom"))
    for row_axes, row_labels in zip(axes, labels):
        for axis, label in zip(row_axes, row_labels):
            axis.set_ylabel(label)
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_yscale("log")
    if presentation.get("evolution_variance",False):
        axes[1,1].set_ylabel("Expected evolution variance")
        axes[2,1].set_ylabel("Posterior covariance of V and W")
    elif not presentation["prior_location_coefficient"]:
        axes[1, 1].text(0.5, 0.5, "No single prior-location coefficient\nfor this model", ha="center", transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].set_yscale("log")
    if not presentation["degrees_of_freedom"] and not presentation.get("evolution_variance",False):
        axes[2, 1].text(0.5, 0.5, "No Student-t degrees of freedom\nfor this predictive law", ha="center", transform=axes[2, 1].transAxes)
    axes[0, 0].legend(fontsize="small")
    for axis in axes[-1]:
        axis.set_xlabel("Information update (0 = prior)")
    for axis, label in zip(differences, ("Mean location difference\n(target units)",
                                        "Variance median difference\n(squared units)", "Predictive width difference\n(target units)")):
        axis.axhline(0, color="0.4", linewidth=0.7)
        axis.set_yscale("symlog", linthresh=0.01)
        axis.set_ylabel(label)
    differences[0].legend(fontsize="small")
    differences[-1].set_xlabel("Information update (0 = prior)")
    if presentation.get("evolution_variance",False):
        extra=Figure(figsize=(11,max(4,3*len(summaries))),layout="constrained")
        FigureCanvasAgg(extra)
        extra_axes=extra.subplots(len(summaries),1,squeeze=False).flat
        for item,axis in zip(summaries,extra_axes):
            selected=[row for row in rows if row["configuration"]==item["name"]]
            laws=[json.loads(row["evolution_probabilities_json"]) if row["status"]=="completed" else [] for row in selected]
            support=next((law for law in laws if law),[])
            for index,(value,_) in enumerate(support):
                axis.plot([int(row["update_index"]) for row in selected],
                    [law[index][1] if law else float('nan') for law in laws],label=f'W={value:g}')
            axis.set(title=item["name"],xlabel="Information update (0 = prior)",ylabel="Posterior mass",ylim=(0,1))
            if support:axis.legend()
        extra.suptitle("Evolution-variance prior influence; fixed support; review pending")
        figures.append(extra)
    for figure, destination in zip(figures, destinations):
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=150)
    return destinations
