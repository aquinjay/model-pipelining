"""Exact joint prior/posterior replications and descriptive model criticism.

One (mu, variance) pair generates a whole replicated dataset. Discrepancies
never decide model approval. These simulations are independent draws, not MCMC.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np

from model_pipelining.experiments import ExperimentSpec, ForecastCase, read_task_data, select_history_as_of
from model_pipelining.models import ForecastRequest
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.snapshots import retrieve_snapshot
from model_pipelining.tags import resolve_tag


@dataclass(frozen=True)
class ReplicatedDataset:
    replication: int
    mean: float
    observation_variance: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class PredictiveReplications:
    kind: str
    seed: int
    requested_replications: int
    observation_count: int
    distribution: dict
    datasets: tuple[ReplicatedDataset, ...]
    failures: tuple[dict, ...]


def _validate(model, replications, seed):
    if type(model) is not NormalMeanVariance or model.mcmc is not None:
        raise ValueError("Checks require a concrete NormalMeanVariance with mcmc=None; draws are exact")
    if type(replications) is not int or replications < 1:
        raise ValueError("replications must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _simulate(*, kind, distribution, observation_count, replications, seed):
    datasets, failures = [], []
    for index in range(replications):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [seed, 0 if kind == "prior" else 1, index])))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Gamma is unit-scale precision: v=b/G, G~Gamma(a,1).
                gamma = float(generator.gamma(distribution["shape"]))
                if not math.isfinite(gamma) or gamma <= 0:
                    raise ValueError("Unrepresentable precision draw")
                variance = distribution["scale"] / gamma
                conditional_variance = variance / distribution["mean_precision"]
                if not all(math.isfinite(x) and x > 0 for x in (variance, conditional_variance)):
                    raise ValueError("Unrepresentable variance draw")
                mu = float(generator.normal(distribution["mean"], math.sqrt(conditional_variance)))
                values = generator.normal(mu, math.sqrt(variance), size=observation_count)
                if not math.isfinite(mu) or not np.isfinite(values).all():
                    raise ValueError("Nonfinite parameter or replicated dataset")
            datasets.append(ReplicatedDataset(index, mu, variance, tuple(map(float, values))))
        except (ValueError, OverflowError, FloatingPointError) as error:
            failures.append({"replication": index, "error": f"{type(error).__name__}: {error}"})
    return PredictiveReplications(kind, seed, replications, observation_count, distribution,
                                  tuple(datasets), tuple(failures))


def simulate_prior_predictive(model: NormalMeanVariance, *, observation_count: int,
                              replications: int = 1000, seed: int = 0) -> PredictiveReplications:
    """Draw joint prior parameters and whole datasets without observations.

    A failed replication is retained as an error and later draws continue.
    No model fit, MCMC, persistence, or global RNG mutation occurs.
    """
    _validate(model, replications, seed)
    if type(observation_count) is not int or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    distribution = {"mean": model.prior_mean, "mean_precision": model.prior_mean_precision,
                    "shape": model.prior_shape, "scale": model.prior_scale}
    return _simulate(kind="prior", distribution=distribution, observation_count=observation_count,
                     replications=replications, seed=seed)


def simulate_posterior_predictive(model: NormalMeanVariance, request: ForecastRequest, *,
                                  replications: int = 1000, seed: int = 0) -> PredictiveReplications:
    """Draw exact joint-posterior parameters, then replicate the fitted history.

    The dataset length equals len(request.history). This is in-sample criticism,
    not a sequence of held-out forecasts or draws from independent marginal t's.
    """
    _validate(model, replications, seed)
    posterior = model.posterior(request)
    distribution = {key: getattr(posterior, key) for key in ("mean", "mean_precision", "shape", "scale")}
    return _simulate(kind="posterior", distribution=distribution, observation_count=len(request.history),
                     replications=replications, seed=seed)


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


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_checks(*, model, observation_count, replications, seed, output_directory,
                context=None, request=None):
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "kind": "normal_mean_variance_predictive_checks", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "configuration": model.configuration,
        "replications": replications, "observation_count": observation_count, "seed": seed,
        "context": context,
        "scope": {
            "purpose": "Prior plausibility and in-sample model criticism; human interpretation pending",
            "generation": "One independent joint parameter pair per entire replicated dataset",
            "posterior": "Exact conjugate posterior draws; no MCMC, refitting per position, or held-out scoring",
            "seed_derivation": "PCG64(SeedSequence([seed, 0 for prior or 1 for posterior, replication_index]))",
            "statistics": list(_STATISTICS), "quantiles": "linear interpolation", "sample_sd_ddof": 1,
            "half_mean_difference": "second half mean minus first; odd middle joins second half",
            "max_cusum": "max(abs(cumsum(y-mean))) / (sample_sd*sqrt(n))",
            "tail_fractions": "Inclusive comparisons against observed statistics, descriptive, not calibrated p-values",
            "precision": "Independent-replication Bernoulli MCSE plus Wilson 95% intervals; zero MCSE at 0/1 is not certainty",
            "policy": "No discrepancy, coverage, prior plausibility or model-approval thresholds",
        },
        "environment": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy")},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("predictive_checks.py", "normal_mean_variance.py", "experiments.py", "ingest.py")},
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
        error = None
        try:
            simulation = (simulate_prior_predictive(model, observation_count=observation_count,
                                                   replications=replications, seed=seed) if kind == "prior" else
                          simulate_posterior_predictive(model, request, replications=replications, seed=seed))
        except (ValueError, OverflowError, FloatingPointError) as failure:
            error = f"{type(failure).__name__}: {failure}"
        datasets = simulation.datasets if simulation else ()
        failures = simulation.failures if simulation else ()
        parameter_rows = [{"replication": item.replication, "mean": item.mean,
                           "observation_variance": item.observation_variance, "status": "completed", "error": None}
                          for item in datasets]
        parameter_rows.extend({"replication": item["replication"], "mean": None, "observation_variance": None,
                               "status": "failed", "error": item["error"]} for item in failures)
        _write_csv(directory / "parameter_draws.csv", ("replication", "mean", "observation_variance", "status", "error"),
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
                        "The model assumes one constant mean/variance and conditionally independent observations",
                        "Finite-replication tail fractions are descriptive and have no automatic acceptance threshold",
                        "Prior influence comparisons are a separate future task"],
    })
    manifest.update(status="completed" if all(c["status"] == "passed" for c in checks if c["required"]) else "failed",
                    finished_at=datetime.now(UTC).isoformat(),
                    phase_status={kind: summary["status"] for kind, summary in summaries.items()})
    (output / "predictive_checks.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def run_prior_predictive_checks(*, model: NormalMeanVariance, observation_count: int,
                                output_directory: str | Path, replications: int = 1000, seed: int = 0) -> Path:
    """Persist prior datasets/statistics before observing data. Plot separately."""
    _validate(model, replications, seed)
    if type(observation_count) is not int or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    return _run_checks(model=model, observation_count=observation_count, replications=replications,
                       seed=seed, output_directory=output_directory)


def _select_development_history(*, dataset: str, issued_at: datetime, snapshot_id: str | None = None,
                         tag: str | None = None, store: str | Path = "data/versioned",
                         experiment: ExperimentSpec | None = None):
    """Common verified end-history/context for predictive and prior comparisons."""
    if (snapshot_id is None) == (tag is None):
        raise ValueError("Supply exactly one of snapshot_id or tag")
    if not isinstance(issued_at, datetime) or issued_at.utcoffset() is None:
        raise ValueError("issued_at must be a timezone-aware datetime")
    issued_at = issued_at.astimezone(UTC)
    specification = ExperimentSpec() if experiment is None else experiment
    if not isinstance(specification, ExperimentSpec) or specification.evaluation.role != "development":
        raise ValueError("A development ExperimentSpec is required")
    boundary = specification.evaluation.final_start
    if boundary is not None and issued_at >= boundary:
        raise ValueError("issued_at must precede the experiment's final_start")
    target_time = issued_at + timedelta(days=specification.task.horizon_days)
    selected_id = resolve_tag(dataset=dataset, tag=tag, store=store) if tag is not None else snapshot_id
    versions = read_task_data(retrieve_snapshot(selected_id, dataset=dataset, store=store), specification.task)
    selected = select_history_as_of(versions, issued_at=issued_at)
    if len(selected) < specification.task.minimum_history:
        raise ValueError("Insufficient available development history")
    case = ForecastCase(0, issued_at, target_time, selected)
    context = {"input": {"dataset": dataset, "snapshot_id": selected_id, "selected_tag": tag},
               "issued_at": issued_at.isoformat(), "history_sha256": case.history_sha256,
               "experiment": specification.to_dict(),
               "selection": "Latest available release per period with period and availability <= issued_at",
               "unused_forecast_settings": "horizon/issuance delay/actual policy/benchmarks do not score or select check datasets; issued_at is explicit",
               "selected_history": [{"timestamp": item.timestamp.isoformat(), "available_at": item.available_at.isoformat(), "value": item.value}
                                    for item in selected]}
    return specification, versions, case, context


def run_predictive_checks(*, dataset: str, model: NormalMeanVariance, issued_at: datetime,
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
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "normal_mean_variance_predictive_checks"
            or manifest.get("status") not in ("completed", "failed")):
        raise ValueError("Plotting requires finished version-1 predictive checks")
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
                       f"{summary['completed_replications']}/{summary['expected_replications']} datasets generated; "
                       "one mean/variance pair per dataset")
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
