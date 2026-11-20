"""Controlled synthetic investigation of explicit evolution-variance settings."""

from datetime import UTC, datetime, timedelta
import csv
import json
import math
from pathlib import Path

import numpy as np

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.experiments import ExperimentSpec, ForecastTask, EvaluationPlan
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.sequential import run_sequential_checks
from model_pipelining.snapshots import register_snapshot


def run_dynamics_study(*, output_directory, evolution_variances=(0., .01, .1, 1.),
                       observation_count=120, development_count=100, minimum_history=20, seed=42):
    """Compare fixed dynamics on matched development cases; never tune or approve."""
    if (not isinstance(evolution_variances, tuple) or not evolution_variances
            or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in evolution_variances)
            or len(set(evolution_variances)) != len(evolution_variances)):
        raise ValueError("Choose unique finite nonnegative evolution variances")
    if (any(type(value) is not int for value in (observation_count, development_count, minimum_history, seed))
            or not 1 <= minimum_history < development_count < observation_count or not 0 <= seed < 2**32):
        raise ValueError("Need minimum_history < development_count < observation_count and a valid seed")
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output/"source").mkdir()
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=observation_count)
    start = datetime(2020,1,1,tzinfo=UTC)
    cutoff = start+timedelta(days=development_count)-timedelta(seconds=1)
    spec = ExperimentSpec(task=ForecastTask(minimum_history=minimum_history),
        evaluation=EvaluationPlan(final_start=start+timedelta(days=development_count),
                                 final_end=start+timedelta(days=observation_count)))
    manifest = {"format_version": 1, "kind": "dynamics_study", "status": "running", "seed": seed,
        "evolution_variances": evolution_variances, "experiment": spec.to_dict(),
        "generating_process": "Shared unit Gaussian noise; stable level 10, gradual 10+.05*t, abrupt 10+5*I[t>=development_count//2]",
        "observation_count": observation_count, "development_count": development_count,
        "purpose": "Controlled assumption investigation; no tuning, ranking or application/model approval"}
    write_json(output/"study.json", manifest)
    summaries = []
    for series in ("stable", "gradual", "abrupt"):
        values = [10+error+(.05*i if series == "gradual" else 5*(i >= development_count//2) if series == "abrupt" else 0)
                  for i,error in enumerate(noise)]
        source = output/"source"/f"{series}.csv"
        write_csv(source, ("timestamp", "value"), ({"timestamp": (start+timedelta(days=i)).isoformat(), "value": float(value)}
                                                   for i,value in enumerate(values)))
        snapshot = register_snapshot(source, dataset=series, store=output/"store")
        baseline = None
        for index, variance in enumerate(evolution_variances):
            model = LocalLevelVariance(evolution_variance=variance)
            path = run_sequential_checks(dataset=series, model=model, snapshot_id=snapshot.snapshot_id,
                store=output/"store", experiment=spec, issued_at=cutoff, output_directory=output/series/f"setting-{index:03d}")
            with (path/"sequential.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            identity = [(row["issued_at"],row["target_time"],row["history_sha256"],row["actual"]) for row in rows]
            if baseline is None:
                baseline = identity
            if identity != baseline:
                raise ValueError("Dynamics comparison cases or histories differ")
            summary = json.loads((path/"summary.json").read_text())
            complete = len(rows) == summary["completed_cases"]
            summaries.append({"series": series, "evolution_variance": variance, "path": str(path.relative_to(output)),
                "status": "completed" if complete else "failed", "cases": len(rows),
                "mean_squared_error": math.fsum(float(row["innovation"])**2 for row in rows)/len(rows) if complete else None,
                "mean_squared_standardized_innovation": summary["mean_squared_standardized_innovation"],
                "lag_one_correlation": summary["residual_autocorrelation"][0]["correlation"] if summary["residual_autocorrelation"] else None})
    write_json(output/"summary.json", summaries)
    write_json(output/"review.json", {"status": "pending", "decision": None,
        "limitations": ["Known synthetic development fixtures, not an untouched application evaluation",
                        "A global variance changes smoothness; abrupt changes may need a different transition law"]})
    manifest["status"] = "completed" if all(row["status"] == "completed" for row in summaries) else "failed"
    write_json(output/"study.json", manifest, overwrite=True)
    return output


def plot_dynamics_study(directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    manifest = json.loads((directory/"study.json").read_text())
    if manifest.get("kind") != "dynamics_study" or manifest.get("status") not in ("completed","failed"):
        raise ValueError("Finished dynamics study required")
    destination = directory/"dynamics.png"
    if destination.exists():
        raise FileExistsError(destination)
    rows = json.loads((directory/"summary.json").read_text())
    figure=Figure(figsize=(12,4),layout="constrained");FigureCanvasAgg(figure)
    axes=figure.subplots(1,3)
    for series in ("stable","gradual","abrupt"):
        selected=sorted((row for row in rows if row["series"]==series),key=lambda row:row["evolution_variance"])
        for axis,key in zip(axes,("mean_squared_error","mean_squared_standardized_innovation","lag_one_correlation")):
            axis.plot([row["evolution_variance"] for row in selected],
                      [row[key] if row[key] is not None else float('nan') for row in selected],marker='o',label=series)
            axis.set(xscale='symlog',xlabel='Fixed evolution variance',ylabel=key.replace('_',' '))
    axes[0].legend();figure.suptitle('Controlled development dynamics; human review pending')
    with destination.open('xb') as stream:figure.savefig(stream,format='png',dpi=150)
    return destination
