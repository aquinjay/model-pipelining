"""Joint prior/posterior replications and descriptive model criticism.

Constant-mean replications share a (mu, variance) pair. Dynamic replications
share an initial level/variance pair and draw fresh process innovations.
Discrepancies never decide approval. Draws are independent, not MCMC; dynamic
posterior variance integration has explicit numerical precision checks.
"""

import csv
from datetime import UTC, datetime
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np

from model_pipelining.evidence import (PredictiveReplications, ReplicatedDataset,
    get_evidence_provider, source_hashes, saved_presentation, validate_replications, _write_csv, _write_json)
from model_pipelining.experiments import ExperimentSpec, select_development_history as _select_development_history
from model_pipelining.models import Forecaster, ForecastRequest


def _validate(model, replications, seed):
    provider = get_evidence_provider(model, "predictive_checks")
    if type(replications) is not int or replications < 1:
        raise ValueError("replications must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return provider


def simulate_prior_predictive(model: Forecaster, *, observation_count: int,
                              replications: int = 1000, seed: int = 0) -> PredictiveReplications:
    """Delegate joint prior datasets to the model's evidence provider, without data."""
    provider = _validate(model, replications, seed)
    if type(observation_count) is not int or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    result = provider.simulate(observation_count=observation_count, replications=replications, seed=seed)
    return validate_replications(result, observation_count=observation_count, replications=replications,
                                 seed=seed, kind="prior", latent_paths=provider.metadata.latent_paths)


def simulate_posterior_predictive(model: Forecaster, request: ForecastRequest, *,
                                  replications: int = 1000, seed: int = 0) -> PredictiveReplications:
    """Delegate whole-history replication to the provider; this is in-sample criticism.

    Providers record exact or numerical inference semantics in their metadata.
    This does not make held-out forecasts or reuse a fitted latent path.
    """
    provider = _validate(model, replications, seed)
    result = provider.simulate(observation_count=len(request.history), replications=replications,
                               seed=seed, request=request)
    return validate_replications(result, observation_count=len(request.history), replications=replications,
                                 seed=seed, kind="posterior", latent_paths=provider.metadata.latent_paths)


_STATISTICS = (
    "mean", "sample_sd", "minimum", "maximum", "q05", "median", "q95",
    "maximum_standardized_deviation", "lag_one_correlation", "half_mean_difference", "max_cusum",
)


def dataset_statistics(values) -> dict:
    """Fixed location/spread/tail/order discrepancies, with explicit undefineds.

    Quantiles use NumPy's linear interpolation. SD uses n-1. CUSUM is
    max(abs(cumsum(y-mean))) / (sample_sd*sqrt(n)). Half difference is second
    half mean minus first half mean; an odd middle observation joins the second.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Statistics require a nonempty finite one-dimensional dataset")
    n = len(array)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        average = float(array.mean())
        centered = array - average
        sd = float(array.std(ddof=1)) if n > 1 else None
        result = {"mean": average, "sample_sd": sd, "minimum": float(array.min()),
                  "maximum": float(array.max()), "q05": float(np.quantile(array, 0.05)),
                  "median": float(np.quantile(array, 0.5)), "q95": float(np.quantile(array, 0.95)),
                  "maximum_standardized_deviation": float(np.max(np.abs(centered)) / sd) if sd else None,
                  "lag_one_correlation": None,
                  "half_mean_difference": float(array[n // 2:].mean() - array[:n // 2].mean()) if n > 1 else None,
                  "max_cusum": float(np.max(np.abs(np.cumsum(centered))) / (sd * math.sqrt(n))) if sd else None}
        if n >= 3 and np.ptp(array[:-1]) > 0 and np.ptp(array[1:]) > 0:
            result["lag_one_correlation"] = float(np.corrcoef(array[:-1], array[1:])[0, 1])
    reasons = {
        "sample_sd": "Need at least two observations",
        "half_mean_difference": "Need at least two observations",
        "maximum_standardized_deviation": "Need at least two observations and positive sample SD",
        "max_cusum": "Need at least two observations and positive sample SD",
        "lag_one_correlation": "Need at least three observations and nonconstant lagged sequences",
    }
    if any(x is not None and not math.isfinite(x) for x in result.values()):
        raise ValueError("Unrepresentable dataset statistic")
    return {name: {"value": result[name], "unavailable_reason": reasons.get(name) if result[name] is None else None}
            for name in _STATISTICS}


def _tail_summary(count, total):
    if not total:
        return {"count": count, "fraction": None, "mcse": None, "wilson_95": None}
    fraction = count / total
    z = 1.959963984540054
    denominator = 1 + z*z / total
    center = (fraction + z*z / (2*total)) / denominator
    half = z * math.sqrt(fraction*(1-fraction)/total + z*z/(4*total*total)) / denominator
    return {"count": count, "fraction": fraction, "mcse": math.sqrt(fraction*(1-fraction)/total),
            "wilson_95": [max(0.0, center-half), min(1.0, center+half)]}


def _replicated_quantiles(values):
    # Interpolate as a convex sum, avoiding overflow in high-low for finite
    # opposite-sign extremes. These are empirical, not distribution moments.
    if not values:
        return None
    ordered = sorted(values)
    result = {}
    for probability in (0.025, 0.5, 0.975):
        position = probability * (len(ordered)-1)
        lower = math.floor(position)
        fraction = position-lower
        result[str(probability)] = math.fsum(((1-fraction)*ordered[lower],
                                            fraction*ordered[min(lower+1, len(ordered)-1)]))
    return result


def _run_checks(*, model, observation_count, replications, seed, output_directory,
                context=None, request=None):
    metadata = get_evidence_provider(model, "predictive_checks").metadata
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "kind": f"{metadata.family}_predictive_checks", "family": metadata.family, "presentation": metadata.presentation, "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "configuration": model.configuration,
        "replications": replications, "observation_count": observation_count, "seed": seed,
        "context": context,
        "scope": {
            "purpose": "Prior plausibility and in-sample model criticism; human interpretation pending",
            **metadata.predictive_scope,
            "seed_derivation": "PCG64(SeedSequence([seed, 0 for prior or 1 for posterior, replication_index]))",
            "statistics": list(_STATISTICS), "quantiles": "linear interpolation", "sample_sd_ddof": 1,
            "half_mean_difference": "second half mean minus first; odd middle joins second half",
            "max_cusum": "max(abs(cumsum(y-mean))) / (sample_sd*sqrt(n))",
            "tail_fractions": "Inclusive comparisons against observed statistics, descriptive, not calibrated p-values",
            "precision": "Independent-replication Bernoulli MCSE plus Wilson 95% intervals; zero MCSE at 0/1 is not certainty",
            "policy": "No discrepancy, coverage, prior plausibility or model-approval thresholds",
        },
        "environment": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy")},
        "source_sha256": source_hashes(*(Path(__file__).with_name(name) for name in
            ("predictive_checks.py", "evidence.py", "experiments.py", "ingest.py", "models.py")),
            *metadata.implementation_files),
    }
    _write_json(output / "predictive_checks.json", manifest)
    observations = context["selected_history"] if context else []
    _write_csv(output / "observations.csv", ("observation_index", "timestamp", "available_at", "value"),
               ({"observation_index": i, **item} for i, item in enumerate(observations)))
    observed = {"status": "not_applicable", "statistics": {}, "error": None}
    if request is not None:
        try:
            observed.update(status="completed", statistics=dataset_statistics([item.value for item in request.history]))
        except (ValueError, OverflowError, FloatingPointError) as error:
            observed.update(status="failed", error=f"{type(error).__name__}: {error}")
    _write_json(output / "observed_statistics.json", observed)
    summaries = {}
    for kind in (("prior", "posterior") if request is not None else ("prior",)):
        directory = output / kind
        directory.mkdir()
        simulation = None
        error, inference_failure = None, None
        try:
            simulation = (simulate_prior_predictive(model, observation_count=observation_count,
                                                   replications=replications, seed=seed) if kind == "prior" else
                          simulate_posterior_predictive(model, request, replications=replications, seed=seed))
        except (ValueError, OverflowError, FloatingPointError) as failure:
            error = f"{type(failure).__name__}: {failure}"
            inference_failure = getattr(failure, "evidence", None)
        datasets = simulation.datasets if simulation else ()
        failures = simulation.failures if simulation else ()
        parameter_rows = [{"replication": item.replication,
                           **{name: getattr(item, name) for name in metadata.parameter_fields},
                           "status": "completed", "error": None} for item in datasets]
        parameter_rows.extend({"replication": item["replication"],
                               **{name: None for name in metadata.parameter_fields},
                               "status": "failed", "error": item["error"]} for item in failures)
        # Preserve the original static/dynamic CSV column order.
        parameter_fields = ("replication", "mean", "observation_variance", "status", "error",
                            *(name for name in metadata.parameter_fields if name not in ("mean", "observation_variance")))
        if metadata.latent_paths:
            _write_csv(directory/"latent_levels.csv", ("replication", "observation_index", "level"),
                ({"replication": item.replication, "observation_index": i, "level": value}
                 for item in datasets for i, value in enumerate(item.latent_levels)))
        _write_csv(directory / "parameter_draws.csv", parameter_fields,
                   sorted(parameter_rows, key=lambda row: row["replication"]))
        _write_csv(directory / "replicated.csv", ("replication", "observation_index", "value"),
                   ({"replication": item.replication, "observation_index": i, "value": value}
                    for item in datasets for i, value in enumerate(item.values)))
        statistic_rows, statistic_failures = [], []
        for item in datasets:
            try:
                statistics = dataset_statistics(item.values)
                statistic_rows.extend({"replication": item.replication, "statistic": name, **value}
                                      for name, value in statistics.items())
            except (ValueError, OverflowError, FloatingPointError) as failure:
                statistic_failures.append({"replication": item.replication, "error": f"{type(failure).__name__}: {failure}"})
        _write_csv(directory / "statistics.csv", ("replication", "statistic", "value", "unavailable_reason"), statistic_rows)
        comparisons = {}
        for name in _STATISTICS:
            valid = [row["value"] for row in statistic_rows if row["statistic"] == name and row["value"] is not None]
            observed_entry = observed["statistics"].get(name, {})
            observed_value = observed_entry.get("value")
            total = len(valid)
            comparisons[name] = {
                "observed": observed_value,
                "comparison_unavailable_reason": (observed_entry.get("unavailable_reason") or "No valid replicated statistic")
                    if observed_value is None or not total else None,
                "valid_replications": total, "missing_replications": replications-total,
                "replicated_quantiles": _replicated_quantiles(valid),
                "greater_equal": _tail_summary(sum(x >= observed_value for x in valid), total) if observed_value is not None else None,
                "less_equal": _tail_summary(sum(x <= observed_value for x in valid), total) if observed_value is not None else None,
                "equal_count": sum(x == observed_value for x in valid) if observed_value is not None else None,
            }
            if request is None:
                comparisons[name]["comparison_unavailable_reason"] = "Prior-only simulation has no observed comparison"
            elif observed["status"] == "failed":
                comparisons[name]["comparison_unavailable_reason"] = observed["error"]
        summary = {"kind": kind, "status": "completed" if simulation and not failures and not statistic_failures else "failed",
                   "expected_replications": replications, "completed_replications": len(datasets),
                   "distribution": simulation.distribution if simulation else None,
                   "error": error, "simulation_failures": failures, "statistic_failures": statistic_failures,
                   "comparisons": comparisons}
        if metadata.integration_checks:
            summary["inference_failure"] = inference_failure
        _write_json(directory / "summary.json", summary)
        summaries[kind] = summary
    checks = [{"name": f"{kind}_simulation_and_statistics", "required": True,
               "status": "passed" if summary["status"] == "completed" else "failed"}
              for kind, summary in summaries.items()]
    checks.append({"name": "observed_statistics_computed", "required": request is not None,
                   "status": {"completed": "passed", "failed": "failed", "not_applicable": "not_applicable"}[observed["status"]]})
    _write_json(output / "checks.json", checks)
    _write_json(output / "review.json", {
        "status": "pending", "decision": None,
        "questions": ["Are prior-generated levels, variability and extremes plausible in the target units?",
                      "Which observed tail, dependence or level-change patterns are poorly reproduced?",
                      "Do discrepancies motivate changing the model or further investigation?"],
        "limitations": ["In-sample replication is not held-out calibration or model approval",
                        "The model assumes Gaussian observations and the configured mean/state structure",
                        "Finite-replication tail fractions are descriptive and have no automatic acceptance threshold",
                        "Prior influence comparisons are a separate explicit workflow"],
    })
    manifest.update(status="completed" if all(c["status"] == "passed" for c in checks if c["required"]) else "failed",
                    finished_at=datetime.now(UTC).isoformat(),
                    phase_status={kind: summary["status"] for kind, summary in summaries.items()})
    (output / "predictive_checks.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def run_prior_predictive_checks(*, model: Forecaster, observation_count: int,
                                output_directory: str | Path, replications: int = 1000, seed: int = 0) -> Path:
    """Persist prior datasets/statistics before observing data. Plot separately."""
    _validate(model, replications, seed)
    if type(observation_count) is not int or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    return _run_checks(model=model, observation_count=observation_count, replications=replications,
                       seed=seed, output_directory=output_directory)


def run_predictive_checks(*, dataset: str, model: Forecaster, issued_at: datetime,
                         output_directory: str | Path, snapshot_id: str | None = None,
                         tag: str | None = None, store: str | Path = "data/versioned",
                         experiment: ExperimentSpec | None = None, replications: int = 1000, seed: int = 0) -> Path:
    """Save prior and posterior checks for one explicitly selected as-of history.

    Only development role is allowed. Issuance must precede any final boundary.
    Latest eligible releases per period are selected; final/future values never
    enter the posterior, observed discrepancies, or seeds. Invalid options/input
    raise before output creation; numerical failures are saved; I/O propagates.
    """
    _validate(model, replications, seed)
    _, _, case, context = _select_development_history(dataset=dataset, issued_at=issued_at,
        snapshot_id=snapshot_id, tag=tag, store=store, experiment=experiment)
    request = ForecastRequest(case.history, case.issued_at, case.target_time, (), ())
    selected = case.versions
    return _run_checks(model=model, observation_count=len(selected), replications=replications, seed=seed,
                       output_directory=output_directory, context=context, request=request)


def plot_predictive_checks(check_directory: str | Path) -> tuple[Path, ...]:
    """Draw replicated paths, ECDFs and discrepancy histograms from saved files.

    Finished failed phases can be plotted; all destinations must be unused.
    This function never reads the input store, fits a model, or simulates data.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(check_directory).expanduser()
    manifest = json.loads((directory / "predictive_checks.json").read_text(encoding="utf-8"))
    presentation = saved_presentation(manifest, "predictive_checks")
    kinds = tuple(manifest["phase_status"])
    destinations = tuple(directory / f"{kind}_predictive.png" for kind in kinds)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Predictive-check charts already exist")
    with (directory / "observations.csv").open(encoding="utf-8", newline="") as stream:
        observations = [float(row["value"]) for row in csv.DictReader(stream)]
    for kind, destination in zip(kinds, destinations):
        summary = json.loads((directory / kind / "summary.json").read_text(encoding="utf-8"))
        with (directory / kind / "replicated.csv").open(encoding="utf-8", newline="") as stream:
            paths = {}
            for row in csv.DictReader(stream):
                index = int(row["replication"])
                if index not in paths and len(paths) >= 20:
                    continue
                paths.setdefault(index, []).append(float(row["value"]))
        with (directory / kind / "statistics.csv").open(encoding="utf-8", newline="") as stream:
            statistics = list(csv.DictReader(stream))
        if not all(math.isfinite(value) for values in paths.values() for value in values) or not all(math.isfinite(x) for x in observations):
            raise ValueError("Nonfinite saved predictive values")
        figure = Figure(figsize=(12, 13), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(4, 2)
        label = "Prior predictive" if kind == "prior" else "Posterior predictive (in-sample replication)"
        figure.suptitle(f"{label} — computation {summary['status']}; human review pending\n"
                       f"{summary['completed_replications']}/{summary['expected_replications']} datasets generated; " +
                       presentation["generation"])
        for index, values in paths.items():
            axes[0, 0].plot(range(len(values)), values, color="tab:blue", alpha=0.12, linewidth=0.8,
                            label="First up to 20 replications" if index == next(iter(paths)) else None)
            ordered = np.sort(values)
            axes[0, 1].plot(ordered, np.arange(1, len(ordered)+1)/len(ordered), color="tab:blue", alpha=0.15)
        if observations:
            axes[0, 0].plot(range(len(observations)), observations, color="black", linewidth=1, label="Selected observations")
            axes[0, 1].plot(np.sort(observations), np.arange(1, len(observations)+1)/len(observations), color="black", label="Selected observations")
        axes[0, 0].set(xlabel="Observation index", ylabel="Target units", title="Replicated datasets")
        axes[0, 1].set(xlabel="Target units", ylabel="Empirical cumulative probability", title="Distribution shape")
        if paths or observations:
            axes[0, 0].legend(fontsize="small")
        displayed = ("mean", "sample_sd", "maximum_standardized_deviation", "lag_one_correlation",
                     "half_mean_difference", "max_cusum")
        for axis, name in zip(axes[1:].flat, displayed):
            values = [float(row["value"]) for row in statistics if row["statistic"] == name and row["value"]]
            comparison = summary["comparisons"][name]
            if values:
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("Nonfinite saved predictive statistic")
                axis.hist(values, bins=30, color="tab:blue", alpha=0.65, label="Replicated statistic")
            else:
                axis.text(0.5, 0.5, "No defined replicated statistic", ha="center", transform=axis.transAxes)
            if comparison["observed"] is not None:
                axis.axvline(comparison["observed"], color="black", linewidth=1.6, label="Observed statistic")
                tail = comparison["greater_equal"]
                if tail and tail["fraction"] is not None:
                    low, high = tail["wilson_95"]
                    axis.text(0.98, 0.95, f"Fraction ≥ observed: {tail['fraction']:.3f}\nWilson 95%: [{low:.3f}, {high:.3f}]",
                              transform=axis.transAxes, ha="right", va="top", fontsize="small")
            axis.set(xlabel=name.replace("_", " "), ylabel="Replications")
            if values or comparison["observed"] is not None:
                axis.legend(fontsize="small", loc="upper left")
        if summary["error"]:
            axes[0, 0].text(0.5, 0.5, summary["error"], ha="center", wrap=True, transform=axes[0, 0].transAxes)
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=150)
    return destinations
