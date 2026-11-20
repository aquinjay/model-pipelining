"""Data-free posterior-CDF simulation calibration and parameter recovery."""

from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import json
import hashlib
import math
import platform
from numbers import Real
from pathlib import Path

import numpy as np

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.evidence import get_evidence_provider, source_hashes
from model_pipelining.models import ForecastRequest, Observation
from model_pipelining.predictive_checks import tail_summary
from model_pipelining.uncertainty import interval_probabilities


def calibration_transform(item, *, seed, trial, parameter, interval_level):
    """Randomized distributional transform within genuine posterior atoms.

    The provider supplies inclusive F(truth) and optional left limit F(truth-).
    Independent parameter-labelled streams preserve dataset and prefix seeds.
    Continuous providers need only supply their original CDF/interval fields.
    """
    if not isinstance(item,dict) or "cdf" not in item:
        raise ValueError("Calibration transform requires a CDF record")
    left=item.get("cdf_lower",item["cdf"])
    right=item.get("cdf_upper",item["cdf"])
    mass=item.get("interval_mass",interval_level)
    if (not all(isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)
                for value in (left,right,mass,item["cdf"]))
            or not 0<=left<=right<=1 or not 0<mass<=1+1e-12
            or abs(item["cdf"]-right)>1e-12):
        raise ValueError("Invalid posterior atom bounds or interval mass")
    randomized=right>left
    value=right
    if randomized:
        label=int.from_bytes(hashlib.sha256(parameter.encode()).digest()[:8],"little")
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,3,trial,label])))
        value=left+(right-left)*float(rng.random())
    return {**item,"cdf":value,"cdf_lower":left,"cdf_upper":right,"randomized":randomized,
            "interval_mass":min(1.,mass)}


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
            "randomized_trials":sum(row["randomized"] for row in selected),
            "mean_interval_posterior_mass":math.fsum(row["interval_mass"] for row in selected)/count if count else None,
            "coverage": tail_summary(sum(row["lower"] <= row["truth"] <= row["upper"] for row in selected), count),
            "mean_absolute_median_error": math.fsum(abs(row["median"]-row["truth"]) for row in selected)/count if count else None,
            "mean_credible_width": math.fsum(row["upper"]-row["lower"] for row in selected)/count if count else None}
    return summaries


def run_calibration(*, model, observation_count, output_directory, trials=200, seed=0, interval_level=.95):
    """Simulate prior parameters/data, refit and evaluate posterior CDF at truth.

    This CDF-based form of simulation calibration tests marginal numerical
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
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output/"trials").mkdir()
    manifest = {"format_version": 1, "kind": "simulation_calibration", "status": "running",
        "model": {"name": model.name, "configuration": model.configuration}, "trials": trials,
        "observation_count": observation_count, "seed": seed, "parameters": parameters,
        "interval_level": interval_level, "started_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy")},
        "method": "Posterior CDF at truth, randomized within genuine probability atoms; not sampled ranks",
        "seed_derivation": "PCG64(SeedSequence([seed,2,trial_index]))",
        "atom_seed_derivation":"PCG64(SeedSequence([seed,3,trial_index,first_64_bits_SHA256(parameter)])); little-endian label",
        "source_sha256": source_hashes(Path(__file__), Path(__file__).with_name("evidence.py"), Path(__file__).with_name("artifacts.py"),
                                        *provider.metadata.implementation_files)}
    write_json(output/"calibration.json", manifest)
    rows, inference = [], []
    for index in range(trials):
        truth, values, results, error, evidence = {}, (), {}, None, None
        try:
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2, index])))
            truth, values = provider.calibration_draw(observation_count, generator)
            if (set(truth) != set(parameters) or len(values) != observation_count
                    or not all(math.isfinite(value) for value in (*truth.values(), *values))):
                raise ValueError("Invalid simulated truth/data")
            write_json(output/"trials"/f"{index:06d}.json", {"truth": truth, "values": values})
            start = datetime(2000,1,1,tzinfo=UTC)
            history = tuple(Observation(start+timedelta(days=i),value) for i,value in enumerate(values))
            request = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp+timedelta(days=1))
            results, evidence = provider.calibration_fit(request, truth, interval_level)
            if set(results) != set(parameters):
                raise ValueError("Calibration provider omitted a requested parameter")
            for name,item in results.items():
                required={"cdf","lower","median","upper"}
                if (not isinstance(item,dict) or not required.issubset(item)
                        or set(item)-required-{"cdf_lower","cdf_upper","interval_mass"}
                        or any(not isinstance(item[key],Real) or isinstance(item[key],bool) or not math.isfinite(item[key]) for key in required)
                        or not 0 <= item["cdf"] <= 1 or not item["lower"] <= item["median"] <= item["upper"]):
                    raise ValueError("Invalid posterior calibration summary")
                results[name]=calibration_transform(item,seed=seed,trial=index,parameter=name,interval_level=interval_level)
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError) as failure:
            error = f"{type(failure).__name__}: {failure}"
            evidence = getattr(failure, "evidence", evidence)
        inference.append({"trial": index, "error": error, "inference": evidence})
        for name in parameters:
            row = {"trial": index, "parameter": name, "truth": truth.get(name),
                   "status": "failed" if error else "passed", "error": error,
                   "cdf": None, "lower": None, "median": None, "upper": None,
                   "cdf_lower":None,"cdf_upper":None,"randomized":None,"interval_mass":None}
            if error is None:
                row.update(results[name])
            rows.append(row)
    write_csv(output/"calibration.csv", tuple(rows[0]), rows)
    write_json(output/"inference.json", inference)
    write_json(output/"summary.json", summarize_calibration(rows, trials=trials, parameters=parameters, level=interval_level))
    passed = all(row["status"] == "passed" for row in rows)
    write_json(output/"checks.json", [{"name": "all_calibration_trials_computed", "required": True,
                                       "status": "passed" if passed else "failed"}])
    write_json(output/"review.json", {"status": "pending", "decision": None,
        "limitations": ["Marginal CDF calibration does not prove joint correctness or posterior concentration",
            "Simulation coverage averages over the prior, not every fixed parameter value",
            "Uniform-CDF bands use DKW with a union bound across parameters; numerical approximation remains",
            "CDF jumps are randomized; discrete/mixed closed credible intervals may have more than nominal mass",
            "Marginal calibration can pass for prior-only inference; independent likelihood/recovery checks remain necessary",
            "Failed fits can induce selection bias; conditional summaries do not certify calibration",
            "No automatic uniformity or model-approval threshold; no external dataset or MCMC sampler is tested"]})
    manifest.update(status="completed" if passed else "failed", finished_at=datetime.now(UTC).isoformat())
    write_json(output/"calibration.json", manifest, overwrite=True)
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
        pair[0].set(title=name,xlabel="Posterior CDF transform at truth",ylabel="Empirical CDF",xlim=(0,1),ylim=(0,1))
        pair[1].set(title=f"{name}: recovery",xlabel="Simulated truth",ylabel="Posterior median")
    figure.suptitle(f"Simulation calibration — {manifest['status']}; human review pending")
    with destination.open("xb") as stream:
        figure.savefig(stream,format="png",dpi=150)
    return destination
