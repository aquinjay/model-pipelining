"""Run the training-mean baseline against a preserved CSV, then draw its chart.

Launch from the repository root with PYTHONPATH=src; see docs/forecasting.md.
"""

import argparse
from dataclasses import replace
import json

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
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
    options = vars(parser.parse_args())
    specification, role = options.pop("experiment"), options.pop("evaluation_role")
    if role is not None and specification is None:
        parser.error("--evaluation-role requires --experiment")
    if specification is not None:
        experiment = ExperimentSpec.load(specification)
        if role is not None:
            experiment = replace(experiment, evaluation=replace(experiment.evaluation, role=role))
        options["experiment"] = experiment
    run = run_backtest(**options)
    chart = plot_run(run)
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    print(f"Saved run: {run}")
    print(f"Chart: {chart}")
    print(json.dumps(scores, indent=2))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
