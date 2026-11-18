"""Data-free posterior-CDF simulation calibration and parameter recovery."""

from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import json
import math
import platform
from pathlib import Path

import numpy as np

from model_pipelining.evidence import get_evidence_provider, source_hashes, _write_json, _write_csv
from model_pipelining.models import ForecastRequest, Observation
from model_pipelining.predictive_checks import _tail_summary
from model_pipelining.uncertainty import interval_probabilities


def summarize_calibration(rows, *, trials, parameters, level):
    summaries = {}
    for name in parameters:
        selected = [row for row in rows if row["parameter"] == name and row["status"] == "passed"]
        cdfs = sorted(row["cdf"] for row in selected)
        count = len(cdfs)
        summaries[name] = {"completed_trials": count, "failed_trials": trials-count,
            "cdf_mean": math.fsum(cdfs)/count if count else None,
            "cdf_uniform_distance": max((max((i+1)/count-p, p-i/count) for i,p in enumerate(cdfs)), default=None),
            "cdf_simultaneous_95_band": min(1., math.sqrt(math.log(2*len(parameters)/.05)/(2*count))) if count else None,
            "band_applies_to_all_trials": count == trials,
            "central_interval_level": level,
            "coverage": _tail_summary(sum(row["lower"] <= row["truth"] <= row["upper"] for row in selected), count),
            "mean_absolute_median_error": math.fsum(abs(row["median"]-row["truth"]) for row in selected)/count if count else None,
            "mean_credible_width": math.fsum(row["upper"]-row["lower"] for row in selected)/count if count else None}
    return summaries


def run_calibration(*, model, observation_count, output_directory, trials=200, seed=0, interval_level=.95):
    """Simulate prior parameters/data, refit and evaluate posterior CDF at truth.

    This continuous-CDF form of simulation calibration tests marginal numerical
    inference. It does not audit an MCMC sampler, prove all joint dependencies,
    or establish model adequacy for external data. Failures are never discarded.
    """
    provider = get_evidence_provider(model, "calibration")
    for name, value in (("trials", trials), ("observation_count", observation_count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    interval_probabilities(interval_level)
    parameters = provider.calibration_parameters
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError("Calibration needs unique unknown parameter names")
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output/"trials").mkdir()
    manifest = {"format_version": 1, "kind": "simulation_calibration", "status": "running",
        "model": {"name": model.name, "configuration": model.configuration}, "trials": trials,
        "observation_count": observation_count, "seed": seed, "parameters": parameters,
        "interval_level": interval_level, "started_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy")},
        "method": "Continuous posterior CDF at simulated truth; independent prior-data trials, not sampled ranks",
        "seed_derivation": "PCG64(SeedSequence([seed,2,trial_index]))",
        "source_sha256": source_hashes(Path(__file__), Path(__file__).with_name("evidence.py"),
                                        *provider.metadata.implementation_files)}
    _write_json(output/"calibration.json", manifest)
    rows, inference = [], []
    for index in range(trials):
        truth, values, results, error, evidence = {}, (), {}, None, None
        try:
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2, index])))
            truth, values = provider.calibration_draw(observation_count, generator)
            if (set(truth) != set(parameters) or len(values) != observation_count
                    or not all(math.isfinite(value) for value in (*truth.values(), *values))):
                raise ValueError("Invalid simulated truth/data")
            _write_json(output/"trials"/f"{index:06d}.json", {"truth": truth, "values": values})
            start = datetime(2000,1,1,tzinfo=UTC)
            history = tuple(Observation(start+timedelta(days=i),value) for i,value in enumerate(values))
            request = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp+timedelta(days=1))
            results, evidence = provider.calibration_fit(request, truth, interval_level)
            if set(results) != set(parameters):
                raise ValueError("Calibration provider omitted a requested parameter")
            for item in results.values():
                if (any(not math.isfinite(item[key]) for key in ("cdf", "lower", "median", "upper"))
                        or not 0 <= item["cdf"] <= 1 or not item["lower"] <= item["median"] <= item["upper"]):
                    raise ValueError("Invalid posterior calibration summary")
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError) as failure:
            error = f"{type(failure).__name__}: {failure}"
            evidence = getattr(failure, "evidence", evidence)
        inference.append({"trial": index, "error": error, "inference": evidence})
        for name in parameters:
            row = {"trial": index, "parameter": name, "truth": truth.get(name),
                   "status": "failed" if error else "passed", "error": error,
                   "cdf": None, "lower": None, "median": None, "upper": None}
            if error is None:
                row.update(results[name])
            rows.append(row)
    _write_csv(output/"calibration.csv", tuple(rows[0]), rows)
    _write_json(output/"inference.json", inference)
    _write_json(output/"summary.json", summarize_calibration(rows, trials=trials, parameters=parameters, level=interval_level))
    passed = all(row["status"] == "passed" for row in rows)
    _write_json(output/"checks.json", [{"name": "all_calibration_trials_computed", "required": True,
                                       "status": "passed" if passed else "failed"}])
    _write_json(output/"review.json", {"status": "pending", "decision": None,
        "limitations": ["Marginal CDF calibration does not prove joint correctness or posterior concentration",
            "Simulation coverage averages over the prior, not every fixed parameter value",
            "Uniform-CDF bands use DKW with a union bound across parameters; numerical approximation remains",
            "Failed fits can induce selection bias; conditional summaries do not certify calibration",
            "No automatic uniformity or model-approval threshold; no external dataset or MCMC sampler is tested"]})
    manifest.update(status="completed" if passed else "failed", finished_at=datetime.now(UTC).isoformat())
    (output/"calibration.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return output


def plot_calibration(directory):
    """Read saved CDF and recovery evidence only."""
    import csv
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    manifest = json.loads((directory/"calibration.json").read_text())
    if manifest.get("kind") != "simulation_calibration" or manifest.get("status") not in ("completed", "failed"):
        raise ValueError("Finished simulation calibration required")
    destination = directory/"calibration.png"
    if destination.exists():
        raise FileExistsError(destination)
    with (directory/"calibration.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    summaries = json.loads((directory/"summary.json").read_text())
    figure = Figure(figsize=(11, 4*len(manifest["parameters"])), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(manifest["parameters"]),2,squeeze=False)
    for name, pair in zip(manifest["parameters"], axes):
        selected = [row for row in rows if row["parameter"] == name and row["status"] == "passed"]
        values = sorted(float(row["cdf"]) for row in selected)
        if values:
            grid = np.linspace(0,1,101)
            if summaries[name]["band_applies_to_all_trials"]:
                epsilon = summaries[name]["cdf_simultaneous_95_band"]
                pair[0].fill_between(grid,np.maximum(0,grid-epsilon),np.minimum(1,grid+epsilon),alpha=.2,label="Simultaneous 95% uniform reference")
            pair[0].step(values,np.arange(1,len(values)+1)/len(values),where="post")
            pair[0].plot([0,1],[0,1],color="grey")
            truth = [float(row["truth"]) for row in selected]
            median = [float(row["median"]) for row in selected]
            pair[1].scatter(truth,median,s=8,alpha=.6)
            limits=(min(*truth,*median),max(*truth,*median))
            pair[1].plot(limits,limits,color="grey")
        pair[0].set(title=name,xlabel="Posterior CDF at truth",ylabel="Empirical CDF",xlim=(0,1),ylim=(0,1))
        pair[1].set(title=f"{name}: recovery",xlabel="Simulated truth",ylabel="Posterior median")
    figure.suptitle(f"Simulation calibration — {manifest['status']}; human review pending")
    with destination.open("xb") as stream:
        figure.savefig(stream,format="png",dpi=150)
    return destination
