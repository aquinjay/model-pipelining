"""Audit innovation moments and forecasts on frozen controls and abrupt-42."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
import warnings

import arviz as az
import numpy as np

from check_innovation_probability_bounds import write_json
from model_pipelining.innovation_gibbs_check import innovation_gibbs_probes, sample_innovation_gibbs
from model_pipelining.innovation_lattice import bound_innovation_probabilities_lattice
from model_pipelining.innovation_probability_bounds import ProbabilityBoundSettings
from model_pipelining.innovation_reference import InnovationMixtureReference


ROOT = Path(__file__).resolve().parents[1]
QUERIES = {
    "predictive_cdf": "predictive_points", "predictive_density": "predictive_points",
    "initial_cdf_lower": "state_points", "initial_cdf_upper": "state_points",
    "last_cdf_lower": "state_points", "last_cdf_upper": "state_points", "variance_cdf": "variance_points",
}
CONTROLS = ("short-abrupt", "short-dominant", "state-atoms", "zero-jump", "always-jump", "coincident-scales")
CASE = "abrupt-42"
SOURCE_FILES = (
    "scripts/check_innovation_moments.py", "scripts/check_innovation_probability_bounds.py",
    "src/model_pipelining/innovation_gibbs_check.py", "src/model_pipelining/innovation_lattice.py",
    "src/model_pipelining/innovation_particles.py", "src/model_pipelining/innovation_probability_bounds.py",
    "src/model_pipelining/innovation_reference.py", "src/model_pipelining/local_level_variance.py",
)
GIBBS_BUDGETS = {
    "control": {"warmup": 1024, "draws": 8192, "seeds": [231, 232, 233, 234]},
    "coarse": {"warmup": 2048, "draws": 8192, "seeds": [461, 462, 463, 464]},
    "refined": {"warmup": 4096, "draws": 16384, "seeds": [731, 732, 733, 734]},
}
TOLERANCES = {
    "initial_mean": .05, "last_mean": .05, "initial_variance": .05, "last_variance": .05,
    "initial_last_covariance": .05, "observation_variance_mean": .05,
    "variance_jump_count_covariance": .08, "jump_count": .20,
    "predictive_mean": .05, "predictive_variance": .08, "next_jump_probability": 1e-12,
    "predictive_cdf": .025, "predictive_density": .010,
    "initial_cdf_lower": .025, "initial_cdf_upper": .025,
    "last_cdf_lower": .025, "last_cdf_upper": .025, "variance_cdf": .025,
}
DIAGNOSTICS = {"max_rhat": 1.01, "min_bulk_ess": 400, "mcse_multiplier": 3.0}
LATTICE_SEEDS = (41, 43)


def _lookup(posterior, grids):
    out = {}
    for name, tolerance in TOLERANCES.items():
        if name == "jump_count" and name not in posterior:
            out[name] = sum(posterior["jump_probabilities"])
        elif name in QUERIES:
            out.update((f"{name}:{float(point)}", value)
                       for point, value in zip(grids[QUERIES[name]], posterior[name], strict=True))
        else:
            out[name] = posterior[name]
    return out


def _structural_constant(name, model):
    root = name.split(":")[0]
    return (root == "next_jump_probability"
            or model.jump_probability in (0., 1.) and root in {"jump_count", "variance_jump_count_covariance"}
            or model.initial_variance == 0 and root.startswith("initial"))


def _diagnose(features, model):
    diagnostics = {}
    for name, draws in features.items():
        draws = np.asarray(draws, dtype=float)
        tolerance = TOLERANCES[name.split(":")[0]]
        estimate = float(draws.mean())
        row = dict(estimate=estimate, mcse=None, rhat=None, ess_bulk=None)
        if not np.isfinite(draws).all():
            raise ValueError(f"Nonfinite Gibbs contributions for {name}")
        if _structural_constant(name, model):
            if np.ptp(draws) > 1e-12:
                raise ValueError(f"Expected structural constant for {name}")
            row.update(kind="structural_constant", mcse=0., passed=True)
        elif np.ptp(draws) == 0:
            # An observed zero is not an uncertainty bound on a tail event.
            row.update(kind="constant_sample", passed=None)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                metrics = (float(az.mcse(draws, method="mean")),
                           float(az.rhat(draws, method="rank")),
                           float(az.ess(draws, method="bulk")))
            if all(math.isfinite(value) for value in metrics):
                mcse, rhat, ess = metrics
                row.update(kind="mcmc", mcse=mcse, rhat=rhat, ess_bulk=ess,
                    passed=(0 < mcse <= tolerance / DIAGNOSTICS["mcse_multiplier"]
                            and rhat <= DIAGNOSTICS["max_rhat"] and ess >= DIAGNOSTICS["min_bulk_ess"]))
            else:
                row.update(kind="undefined_diagnostic", passed=False)
        diagnostics[name] = row
    return diagnostics


def _compare(diagnostics, target):
    rows = {}
    for name, row in diagnostics.items():
        tolerance = TOLERANCES[name.split(":")[0]]
        error = abs(row["estimate"] - target[name])
        allowance = DIAGNOSTICS["mcse_multiplier"] * (row["mcse"] or 0.)
        # For a constant indicator or an unavailable MCMC diagnostic, retain
        # the point comparison and show explicitly that MCSE is unavailable.
        rows[name] = dict(target=target[name], absolute_difference=error, tolerance=tolerance,
            difference_plus_mcse=error + allowance if row["mcse"] is not None else None,
            passed=error + allowance <= tolerance)
    return rows


def _required_chain_checks(diagnostics):
    # All nonindicator moments and densities require measured precision.
    return all(row["passed"] is True or
               (row["passed"] is None and "_cdf" in name) for name, row in diagnostics.items())


def _fit(out, name, case, plan, targets):
    model = InnovationMixtureReference(**case["model"])
    grids = {key: case[key] for key in set(QUERIES.values())}
    start = time.perf_counter()
    posterior_draws, labels = sample_innovation_gibbs(case["values"], model, **plan)
    fit_seconds = time.perf_counter() - start
    np.savez_compressed(out / f"{name}-draws.npz", samples=posterior_draws, labels=labels)
    diagnostics = _diagnose(innovation_gibbs_probes(posterior_draws, labels, model, **grids), model)
    comparisons = {label: _compare(diagnostics, _lookup(target, grids)) for label, target in targets.items()}
    return dict(diagnostics=diagnostics, comparisons=comparisons, fit_seconds=fit_seconds,
        checks=dict(chain_precision=_required_chain_checks(diagnostics),
                    comparisons=all(item["passed"] for rows in comparisons.values() for item in rows.values())))


def _lattice(out, name, case, seed, samples=None):
    settings = dict(case["settings"], seed=seed)
    if samples is not None:
        settings["samples"] = samples
    start = time.perf_counter()
    result = bound_innovation_probabilities_lattice(case["values"],
        model=InnovationMixtureReference(**case["model"]), settings=ProbabilityBoundSettings(**settings),
        **{key: case[key] for key in set(QUERIES.values())})
    return dict(result, fit_seconds=time.perf_counter() - start)


def _record(out, name, calculate):
    print(name, flush=True)
    try:
        result = calculate()
    except (ValueError, ArithmeticError) as error:
        result = dict(status="failed", error=str(error))
    write_json(out / f"{name}.json", result)
    return result


def _frozen_inputs(source):
    raw = {}
    names = ["study.json", "validation.json"]
    names += [f"cases/{name}.json" for name in (*CONTROLS, CASE)]
    names += [f"inputs/inputs/oracles/{name}.json" for name in CONTROLS]
    names += [f"inputs/inputs/cases/{CASE}.json"]
    for name in names:
        raw[name] = (source / name).read_bytes()
    inputs = {name: json.loads(value) for name, value in raw.items()}
    parent = inputs["study.json"]
    if parent["kind"] != "innovation_lattice_study" or parent["precision_status"] != "passed":
        raise ValueError("Use the completed state-grid study as the source")
    for name in names:
        if name.startswith("inputs/") and hashlib.sha256(raw[name]).hexdigest() != parent["input_sha256"][name[7:]]:
            raise ValueError(f"Frozen input hash mismatch for {name}")
    for name in (*CONTROLS, CASE):
        case = inputs[f"cases/{name}.json"]
        kind = "oracles" if name in CONTROLS else "cases"
        reference = inputs[f"inputs/inputs/{kind}/{name}.json"]
        if (case["values"], case["model"]) != (reference["values"], reference["model"]):
            raise ValueError(f"Frozen case identity mismatch for {name}")
    return raw, inputs


def _archive(out, folder, files):
    hashes = {}
    for name, raw in files.items():
        path = out / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--refine-control", choices=CONTROLS,
        help="Repeat one named oracle control with longer independent chains and the same criteria")
    options = parser.parse_args(argv)
    raw, inputs = _frozen_inputs(options.source_directory)
    out = options.output_directory
    out.mkdir(parents=True, exist_ok=False)
    design = dict(controls=CONTROLS, held_history=CASE, gibbs=GIBBS_BUDGETS,
        tolerances=TOLERANCES, diagnostics=DIAGNOSTICS, lattice_retry_seeds=LATTICE_SEEDS,
        insufficient_lattice_budget=32, comparison_scale="Raw observation units with probabilities on [0, 1]",
        environment={name: version(name) for name in ("arviz", "numpy", "scipy")},
        input_hashes=_archive(out, "inputs", raw),
        source_hashes=_archive(out, "sources", {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}))
    if options.refine_control:
        name = options.refine_control
        plan = dict(warmup=4096, draws=65536, seeds=[891, 892, 893, 894])
        design["control_refinement"] = dict(name=name, **plan)
        write_json(out / "design.json", design)
        case = inputs[f"cases/{name}.json"]
        oracle = inputs[f"inputs/inputs/oracles/{name}.json"]["batch_adaptive"]
        result = _record(out, name, lambda: _fit(out, name, case, plan, {"oracle": oracle}))
        passed = all(result.get("checks", {"failed": False}).values())
        write_json(out / "summary.json", dict(case=name, verification_passed=passed,
            **{key: result[key] for key in ("checks", "fit_seconds", "error") if key in result}))
        print(f"Saved {out}. Refinement passed: {passed}", flush=True)
        return 0 if passed else 1
    write_json(out / "design.json", design)
    summary = {}
    for name in CONTROLS:
        case = inputs[f"cases/{name}.json"]
        oracle = inputs[f"inputs/inputs/oracles/{name}.json"]["batch_adaptive"]
        result = _record(out, name, lambda: _fit(out, name, case, GIBBS_BUDGETS["control"], {"oracle": oracle}))
        summary[name] = {key: result[key] for key in ("checks", "fit_seconds", "error") if key in result}

    case = inputs[f"cases/{CASE}.json"]
    targets = {"lattice-retained": case["posterior_estimates"]}
    particle = inputs[f"inputs/inputs/cases/{CASE}.json"]["posterior"]
    for seed in LATTICE_SEEDS:
        name = f"lattice-{seed}"
        result = _record(out, name, lambda: _lattice(out, name, case, seed))
        summary[name] = {key: result[key] for key in ("status", "fit_seconds", "maximum_probability_width", "error")
                         if key in result}
        if result["status"] == "completed":
            targets[name] = result["posterior_estimates"]
    result = _record(out, "lattice-insufficient-budget",
        lambda: _lattice(out, "lattice-insufficient-budget", case, LATTICE_SEEDS[0], 32))
    summary["lattice-insufficient-budget"] = dict(expected_status="unresolved", actual_status=result["status"])
    for budget in ("coarse", "refined"):
        name = f"{CASE}-{budget}"
        result = _record(out, name, lambda: _fit(out, name, case, GIBBS_BUDGETS[budget], targets))
        summary[name] = {key: result[key] for key in ("checks", "fit_seconds", "error") if key in result}
    coarse = json.loads((out / f"{CASE}-coarse.json").read_text())
    refined = json.loads((out / f"{CASE}-refined.json").read_text())
    if "diagnostics" in coarse and "diagnostics" in refined:
        grids = {key: case[key] for key in set(QUERIES.values())}
        differences = _compare(refined["diagnostics"], {key: row["estimate"] for key, row in coarse["diagnostics"].items()})
        write_json(out / "refinement-comparison.json", differences)
        write_json(out / "particle-comparison.json", _compare(refined["diagnostics"], _lookup(particle, grids)))
        summary["refinement_passed"] = all(row["passed"] for row in differences.values())
    else:
        summary["refinement_passed"] = False
    summary["verification_passed"] = (
        all(all(summary[name].get("checks", {"failed": False}).values())
            for name in (*CONTROLS, f"{CASE}-coarse", f"{CASE}-refined"))
        and all(summary[f"lattice-{seed}"]["status"] == "completed" for seed in LATTICE_SEEDS)
        and summary["lattice-insufficient-budget"]["actual_status"] == "unresolved" and summary["refinement_passed"])
    write_json(out / "summary.json", summary)
    print(f'Saved {out}. Verification passed: {summary["verification_passed"]}', flush=True)
    return 0 if summary["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
