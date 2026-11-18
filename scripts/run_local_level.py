"""Run local-level mean/uncertainty forecasts and save prior draws for review.

Launch from the repository root with PYTHONPATH=src; see docs/local-level.md.
"""

import argparse
import csv
from dataclasses import replace
import json
import platform

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.local_level import LocalLevel
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings
from model_pipelining.reporting import plot_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--minimum-history", type=int, default=None)
    parser.add_argument("--experiment", help="Version-1 experiment JSON; omit --minimum-history")
    parser.add_argument("--evaluation-role", choices=("development", "final"), help="Explicit override of the experiment's evaluation role")
    parser.add_argument("--interval-levels", type=float, nargs="+", default=(0.8, 0.95))
    parser.add_argument("--initial-mean", type=float, default=0.0)
    parser.add_argument("--initial-variance", type=float, default=100.0)
    parser.add_argument("--observation-variance", type=float, help="Fixed-variance model only; default 1")
    parser.add_argument("--learn-observation-variance", action="store_true")
    parser.add_argument("--prior-shape", type=float, help="Learned-variance inverse-gamma shape; default 3")
    parser.add_argument("--prior-scale", type=float, help="Learned-variance inverse-gamma scale; default 2")
    parser.add_argument("--quadrature-config", help="Learned-variance numerical settings JSON")
    parser.add_argument("--evolution-variance", type=float, default=0.1)
    parser.add_argument("--prior-seed", type=int, default=0)
    options = vars(parser.parse_args())
    specification, role = options.pop("experiment"), options.pop("evaluation_role")
    if role is not None and specification is None:
        parser.error("--evaluation-role requires --experiment")
    if specification is not None:
        experiment = ExperimentSpec.load(specification)
        if role is not None:
            experiment = replace(experiment, evaluation=replace(experiment.evaluation, role=role))
        options["experiment"] = experiment
    learned = options.pop("learn_observation_variance")
    fixed_variance = options.pop("observation_variance")
    shape, scale, numerical = (options.pop(name) for name in ("prior_shape", "prior_scale", "quadrature_config"))
    shared = {name: options.pop(name) for name in ("initial_mean", "initial_variance", "evolution_variance")}
    if learned:
        if fixed_variance is not None:
            parser.error("--observation-variance conflicts with --learn-observation-variance")
        settings = {}
        if numerical:
            with open(numerical, encoding="utf-8") as stream:
                settings = json.load(stream)
        model = LocalLevelVariance(**shared, prior_shape=3 if shape is None else shape,
                                   prior_scale=2 if scale is None else scale, quadrature=QuadratureSettings(**settings))
    else:
        if any(value is not None for value in (shape, scale, numerical)):
            parser.error("Variance-prior/numerical flags require --learn-observation-variance")
        model = LocalLevel(**shared, observation_variance=1 if fixed_variance is None else fixed_variance)
    seed = options.pop("prior_seed")
    # Simulation is deliberately before fitting. Save it after the runner has
    # created its new output directory, without modifying runner-owned artifacts.
    if learned:
        from model_pipelining.predictive_checks import simulate_prior_predictive
        simulation = simulate_prior_predictive(model, observation_count=200, replications=1, seed=seed)
        if simulation.failures:
            raise ValueError(f"Prior simulation failed: {simulation.failures}")
        simulated = simulation.datasets[0]
        prior_rows = tuple(zip(simulated.latent_levels, simulated.values))
    else:
        prior_rows = tuple((draw.level, draw.value) for draw in model.simulate_prior(count=200, seed=seed))
    run = run_backtest(**options, model=model)
    with (run / "prior_predictive.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "level", "value"))
        writer.writerows((index, level, value) for index, (level, value) in enumerate(prior_rows, start=1))
    with (run / "prior_predictive.json").open("x", encoding="utf-8") as stream:
        json.dump({"seed": seed, "count": len(prior_rows), "python": platform.python_version(),
                   "model_configuration": model.configuration, "review_status": "pending",
                   "scope": "One prior trajectory; not held-out forecasts or evidence of prior plausibility"},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved run: {run}")
    print(f"Chart: {plot_run(run)}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    print((run / "scores.json").read_text(encoding="utf-8"))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
