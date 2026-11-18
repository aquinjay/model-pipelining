"""Release-aware development criticism of frozen historical predictive laws."""

from datetime import UTC, datetime
import json
import math
from pathlib import Path
from statistics import correlation, StatisticsError

from scipy.stats import t

from model_pipelining.evidence import get_evidence_provider, source_hashes, _write_json, _write_csv
from model_pipelining.experiments import plan_cases, select_development_history, select_actuals
from model_pipelining.mixtures import NormalMixture
from model_pipelining.models import ForecastRequest


def predictive_law(record):
    """Restore a saved observation distribution, without fitting or source data."""
    if record["family"] == "gaussian_mixture":
        return NormalMixture(tuple(record["weights"]), tuple(record["means"]), tuple(record["variances"]))
    if record["family"] == "student_t":
        if any(not math.isfinite(record[key]) for key in ("location", "scale", "df")) or min(record["scale"], record["df"]) <= 0:
            raise ValueError("Invalid saved Student-t distribution")
        return t(df=record["df"], loc=record["location"], scale=record["scale"])
    raise ValueError("Unsupported saved predictive distribution")


def _correlation(pairs):
    if len(pairs) < 3:
        return None
    try:
        return correlation(*zip(*pairs))
    except StatisticsError:
        return None


def summarize_sequential(rows):
    completed = [row for row in rows if row["status"] == "passed"]
    standardized = [row["standardized_innovation"] for row in rows]
    finite = [value for value in standardized if value is not None]
    acf = [{"lag": lag, "correlation": _correlation([(a, b) for a, b in
        zip(standardized[:-lag], standardized[lag:]) if a is not None and b is not None])}
        for lag in range(1, min(20, len(rows)-1)+1)]
    return {"completed_cases": len(completed), "failed_cases": len(rows)-len(completed),
        "mean_innovation": math.fsum(row["innovation"] for row in completed)/len(completed) if completed else None,
        "standardized_count": len(finite),
        "mean_standardized_innovation": math.fsum(finite)/len(finite) if finite else None,
        "mean_squared_standardized_innovation": math.fsum(value*value for value in finite)/len(finite) if finite else None,
        "pit_lower_tail_count": sum(row["pit"] <= .025 for row in completed),
        "pit_upper_tail_count": sum(row["pit"] >= .975 for row in completed),
        "residual_autocorrelation": acf,
        "interpretation": "Descriptive development criticism; no calibration or model-approval threshold"}


def run_sequential_checks(*, dataset, model, issued_at, output_directory, snapshot_id=None,
                          tag=None, store="data/versioned", experiment=None):
    """Freeze predictions before selecting scoring actuals; exclude final/future data.

    Histories use each historical issuance's eligible releases. Scoring releases
    must also be available by the explicit development cutoff. Failed origins
    and numerical evidence are retained, with later origins continuing.
    """
    provider = get_evidence_provider(model, "sequential_checks")
    spec, versions, end_case, context = select_development_history(dataset=dataset, issued_at=issued_at,
        snapshot_id=snapshot_id, tag=tag, store=store, experiment=experiment)
    cases, _ = plan_cases(versions, spec)
    cases = tuple(case for case in cases if case.target_time <= end_case.issued_at)
    if not cases:
        raise ValueError("No historical forecast targets before the development cutoff")
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format_version": 1, "kind": f"{provider.metadata.family}_sequential_checks",
        "family": provider.metadata.family, "status": "running", "context": context,
        "configuration": model.configuration, "started_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_hashes(Path(__file__), Path(__file__).with_name("experiments.py"),
            Path(__file__).with_name("evidence.py"), *provider.metadata.implementation_files),
        "scope": "Pre-target frozen predictive laws; as-of histories and cutoff-eligible scoring releases; descriptive development evidence"}
    _write_json(output/"sequential_checks.json", manifest)
    predictions = []
    for case in cases:
        record = {"origin_index": case.origin_index, "issued_at": case.issued_at.isoformat(),
                  "target_time": case.target_time.isoformat(), "history_sha256": case.history_sha256,
                  "history_count": len(case.history), "status": "failed", "distribution": None,
                  "error": None, "inference_failure": None}
        try:
            distribution = provider.predictive(ForecastRequest(case.history, case.issued_at, case.target_time))
            predictive_law(distribution)
            if (not math.isfinite(distribution["mean"]) or
                    (distribution["variance"] is not None and
                     (not math.isfinite(distribution["variance"]) or distribution["variance"] <= 0))):
                raise ValueError("Invalid predictive moments")
            json.dumps(distribution, allow_nan=False)
            record.update(status="passed", distribution=distribution)
        except (ValueError, OverflowError, FloatingPointError) as error:
            record.update(error=f"{type(error).__name__}: {error}", inference_failure=getattr(error, "evidence", None))
        predictions.append(record)
    _write_json(output/"predictive_distributions.json", predictions)
    # This happens only after all predictive distributions have been written.
    actuals = select_actuals(tuple(item for item in versions if item.available_at <= end_case.issued_at), spec.task)
    rows = []
    for prediction in predictions:
        row = {key: prediction[key] for key in ("origin_index", "issued_at", "target_time", "history_sha256", "status", "error")}
        row.update(actual=None, actual_available_at=None, prediction=None, predictive_variance=None,
                   innovation=None, standardized_innovation=None, standardization_reason=None, pit=None)
        if prediction["status"] == "passed":
            try:
                actual = actuals[prediction["target_time"]]
                distribution = prediction["distribution"]
                pit = float(predictive_law(distribution).cdf(actual.value))
                innovation = actual.value-distribution["mean"]
                variance = distribution["variance"]
                if not math.isfinite(pit) or not -1e-12 <= pit <= 1+1e-12 or not math.isfinite(innovation):
                    raise ValueError("Invalid scored predictive diagnostic")
                standardized = innovation/math.sqrt(variance) if variance is not None else None
                if standardized is not None and (not math.isfinite(standardized) or not math.isfinite(standardized*standardized)):
                    raise ValueError("Unrepresentable standardized residual moments")
                row.update(actual=actual.value, actual_available_at=actual.available_at.isoformat(),
                    prediction=distribution["mean"], predictive_variance=variance, innovation=innovation,
                    standardized_innovation=standardized,
                    standardization_reason="Predictive variance is undefined" if variance is None else None,
                    pit=min(1., max(0., pit)))
            except (KeyError, ValueError, OverflowError) as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    _write_csv(output/"sequential.csv", tuple(rows[0]), rows)
    _write_json(output/"summary.json", summarize_sequential(rows))
    passed = all(row["status"] == "passed" for row in rows)
    _write_json(output/"checks.json", [{"name": "all_sequential_predictions_and_scores", "required": True,
                                       "status": "passed" if passed else "failed"}])
    _write_json(output/"review.json", {"status": "pending", "decision": None,
        "questions": ["When do forecasts show bias, tail surprises or residual dependence?"],
        "limitations": ["Standardizing a mixture does not make its errors Gaussian",
                        "PIT uses the full predictive CDF, including parameter uncertainty",
                        "Multi-step overlapping errors can be dependent even under the model"]})
    manifest.update(status="completed" if passed else "failed", finished_at=datetime.now(UTC).isoformat())
    (output/"sequential_checks.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return output


def plot_sequential_checks(directory):
    """Draw errors, PIT and residual dependence from saved results only."""
    import csv
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    manifest = json.loads((directory/"sequential_checks.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("status") not in ("completed", "failed"):
        raise ValueError("Finished sequential checks required")
    destination = directory/"sequential.png"
    if destination.exists():
        raise FileExistsError(destination)
    with (directory/"sequential.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((directory/"summary.json").read_text())
    figure = Figure(figsize=(11, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for axis, key in zip(axes[0], ("innovation", "standardized_innovation")):
        axis.plot([float(row[key]) if row[key] else float("nan") for row in rows])
        axis.axhline(0, color="grey", linewidth=.7)
        axis.set(xlabel="Historical case", ylabel=key.replace("_", " "))
    pits = [float(row["pit"]) for row in rows if row["pit"]]
    if pits:
        axes[1, 0].hist(pits, bins=10, range=(0, 1), density=True)
    axes[1, 0].axhline(1, color="grey", linestyle="--")
    axes[1, 0].set(xlabel="Predictive CDF at actual (PIT)", ylabel="Density")
    acf = summary["residual_autocorrelation"]
    axes[1, 1].bar([item["lag"] for item in acf], [item["correlation"] if item["correlation"] is not None else float("nan") for item in acf])
    axes[1, 1].set(xlabel="Lag in forecast cases", ylabel="Standardized-error correlation", ylim=(-1, 1))
    figure.suptitle(f"Sequential development checks — {manifest['status']}; human review pending")
    with destination.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return (destination,)
