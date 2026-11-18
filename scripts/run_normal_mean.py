"""Run exact unknown-mean forecasts, optionally auditing every fit with MCMC."""

import argparse
from dataclasses import replace
import json

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.normal_mean import NormalMean
from model_pipelining.reporting import plot_fit, plot_run
from model_pipelining.sampling import MCMCSettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--minimum-history", type=int)
    parser.add_argument("--experiment", help="Version-1 experiment JSON; omit --minimum-history")
    parser.add_argument("--evaluation-role", choices=("development", "final"))
    parser.add_argument("--interval-levels", type=float, nargs="+", default=(0.8, 0.95))
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-variance", type=float, default=100.0)
    parser.add_argument("--observation-variance", type=float, default=1.0)
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--mcmc", action="store_true", help="Audit every fit with default MCMC settings")
    inference.add_argument("--mcmc-config", help="JSON MCMCSettings; omitted fields use defaults")
    parser.add_argument("--plot-fit-origin", type=int, help="Also draw one saved sampler trace (zero-based origin index)")
    options = vars(parser.parse_args())
    specification, role = options.pop("experiment"), options.pop("evaluation_role")
    if role is not None and specification is None:
        parser.error("--evaluation-role requires --experiment")
    if specification is not None:
        experiment = ExperimentSpec.load(specification)
        if role is not None:
            experiment = replace(experiment, evaluation=replace(experiment.evaluation, role=role))
        options["experiment"] = experiment
    use_mcmc, config_path = options.pop("mcmc"), options.pop("mcmc_config")
    settings = None
    if config_path is not None:
        with open(config_path, encoding="utf-8") as stream:
            settings = MCMCSettings(**json.load(stream))
    elif use_mcmc:
        settings = MCMCSettings()
    origin = options.pop("plot_fit_origin")
    if origin is not None and (settings is None or origin < 0):
        parser.error("--plot-fit-origin requires MCMC and a nonnegative index")
    model = NormalMean(**{name: options.pop(name) for name in ("prior_mean", "prior_variance", "observation_variance")},
                       mcmc=settings)
    run = run_backtest(**options, model=model)
    print(f"Saved run: {run}")
    print(f"Chart: {plot_run(run)}")
    if origin is not None:
        print(f"Fit chart: {plot_fit(run, origin_index=origin)}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    print((run / "scores.json").read_text(encoding="utf-8"))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
